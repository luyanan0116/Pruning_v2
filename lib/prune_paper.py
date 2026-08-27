from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch

from .data import get_loaders
from .paper_pruning.collector import UNIT_TYPES, collect_gradient_response_cache, load_response_cache
from .paper_pruning.config import BudgetConfig, FrequencyConfig, GranularBallConfig, LCBConfig, PipelineConfig
from .paper_pruning.pipeline import score_layer
from .paper_pruning.reporting import LayerReportWriter, plot_granular_balls, plot_lcb_scores, plot_unit_local_granular_balls
from .paper_pruning.weight_budget import allocate_weight_budget, write_weight_budget
from .paper_pruning.wanda_mask import sequential_wanda_prune_


PAPER_METHODS = {"paper_mi", "paper_mi_gb", "paper_mi_gb_lcb", "paper_full"}
METHODS = ("paper_mi", "paper_mi_gb", "paper_mi_gb_lcb", "paper_full")


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated float")
    return values


def _parse_targets(raw: str) -> tuple[str, ...]:
    aliases = {"attn": "attention", "head": "attention", "heads": "attention", "ffn": "mlp"}
    result = []
    for part in raw.split(","):
        value = aliases.get(part.strip().lower(), part.strip().lower())
        if not value:
            continue
        if value not in UNIT_TYPES:
            raise ValueError(f"unknown target {value}; choose mlp,attention")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("at least one target is required")
    return tuple(result)


def _plot_layer_ids(raw: str, num_layers: int) -> set[int]:
    result = set()
    for token in (part.strip().lower() for part in raw.split(",") if part.strip()):
        if token in {"middle", "mid"}:
            result.add(num_layers // 2)
        elif token == "last":
            result.add(num_layers - 1)
        elif token == "first":
            result.add(0)
        else:
            value = int(token)
            if value < 0:
                value += num_layers
            if not 0 <= value < num_layers:
                raise ValueError(f"plot layer {token} is out of range")
            result.add(value)
    return result


def _pipeline_config(args) -> PipelineConfig:
    fine_bins = min(int(args.paper_num_bins), int(args.paper_response_length))
    target_bands = min(int(args.paper_num_bands), fine_bins)
    return PipelineConfig(
        frequency=FrequencyConfig(
            fine_bins=fine_bins,
            target_bands=target_bands,
            mi_neighbors=args.paper_mi_neighbors,
            probe_units=args.paper_probe_units,
            kde_bandwidth_scale=args.paper_kde_bandwidth_scale,
            random_state=args.seed,
            fast_small_mi=args.paper_fast_small_mi,
        ),
        granular_ball=GranularBallConfig(
            purity_thresholds=_parse_float_tuple(args.paper_purity_thresholds),
            min_ball_size=args.paper_min_ball_size,
            max_balls=args.paper_max_balls,
            max_depth=args.paper_max_ball_depth,
            min_purity_gain=args.paper_min_purity_gain,
            min_radius_reduction=args.paper_min_radius_reduction,
            compactness_ratio=args.paper_compactness_ratio,
            min_event_classes=args.paper_min_event_classes,
            min_event_count_per_class=args.paper_min_event_count_per_ball,
            localization_mode=args.paper_gb_localization,
            workers=args.paper_gb_workers,
            worker_chunk_size=args.paper_gb_chunk_size,
            kde_scope=args.paper_kde_scope,
            fusion_mode=args.paper_gb_fusion_mode,
            fusion_max_ratio=args.paper_gb_fusion_max_ratio,
            random_state=args.seed,
        ),
        lcb=LCBConfig(
            repeats=args.paper_lcb_repeats,
            sample_fraction=args.paper_lcb_sample_fraction,
            scenario_fraction=args.paper_lcb_scenario_fraction,
            lcb_lambda=args.lcb_lambda,
            stratify_by_scenario=True,
            cluster_by_base_sample=True,
            random_state=args.seed,
            workers=1,
        ),
        budget=BudgetConfig(
            coverage_ratio=args.paper_band_coverage_ratio,
            coverage_ratios=(
                None if args.paper_band_coverage_ratios is None
                else _parse_float_tuple(args.paper_band_coverage_ratios)
            ),
            coverage_alpha=args.paper_coverage_alpha,
            greedy_batches=args.paper_greedy_batches,
        ),
    )


def _score_config_payload(args, config: PipelineConfig, cache, targets: tuple[str, ...]) -> dict:
    payload = {
        "pipeline": asdict(config),
        "targets": list(targets),
        "response_cache": str(cache.root.resolve()),
        "response_cache_version": int(cache.cache_version),
        "num_observations": int(cache.num_observations),
        "response_length": int(cache.response_length),
        "unit_counts": cache.unit_counts,
        "attention_layouts": cache.attention_layouts,
        "ablation": "MI -> granular multi-scale MI -> repeated LCB -> LCB+band-coverage budget",
    }
    return json.loads(json.dumps(payload))


def _save_evidence(path: Path, evidence: Mapping[str, Mapping[str, Mapping[int, Mapping[str, np.ndarray]]]]) -> None:
    arrays = {}
    for method, type_map in evidence.items():
        for unit_type, layer_map in type_map.items():
            for layer_id, payload in layer_map.items():
                arrays[f"{method}__{unit_type}__{int(layer_id)}__score"] = np.asarray(payload["score"], dtype=np.float32)
                arrays[f"{method}__{unit_type}__{int(layer_id)}__bands"] = np.asarray(payload["bands"], dtype=np.float32)
    np.savez_compressed(path, **arrays)


def _load_evidence(path: Path) -> Dict[str, Dict[str, Dict[int, Dict[str, np.ndarray]]]]:
    result: Dict[str, Dict[str, Dict[int, Dict[str, np.ndarray]]]] = {
        method: {} for method in METHODS
    }
    with np.load(path) as archive:
        for key in archive.files:
            method, unit_type, layer_raw, field = key.split("__", 3)
            result.setdefault(method, {}).setdefault(unit_type, {}).setdefault(int(layer_raw), {})[field] = np.asarray(
                archive[key], dtype=np.float32
            )
    return result


def _score_or_load(args, cache, report_dir: Path, targets: tuple[str, ...]):
    config = _pipeline_config(args)
    payload = _score_config_payload(args, config, cache, targets)
    score_cache_dir = Path(args.paper_score_cache_dir) if args.paper_score_cache_dir else report_dir
    score_cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = score_cache_dir / "score_config.json"
    evidence_path = score_cache_dir / "unit_evidence.npz"
    if evidence_path.exists() and config_path.exists() and not args.paper_overwrite_scores:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous == payload:
            print(f"reusing paper contribution evidence from {score_cache_dir}")
            return _load_evidence(evidence_path)
        print("paper contribution configuration changed; recomputing evidence")

    report_dir.mkdir(parents=True, exist_ok=True)
    plot_layers = _plot_layer_ids(args.paper_plot_layers, cache.num_layers)
    writer = LayerReportWriter(report_dir, band_count=config.frequency.target_bands, prune_step=10)
    evidence: Dict[str, Dict[str, Dict[int, Dict[str, np.ndarray]]]] = {
        method: {unit_type: {} for unit_type in targets} for method in METHODS
    }
    print(
        f"[paper v8] scoring MI -> granular -> repeated LCB; bands={config.frequency.target_bands}; "
        f"repeats={config.lcb.repeats}",
        flush=True,
    )
    try:
        for layer_id in range(cache.num_layers):
            print(f"[paper scoring] layer {layer_id + 1}/{cache.num_layers}", flush=True)
            for target_offset, unit_type in enumerate(targets):
                responses = np.asarray(cache.load_layer(layer_id, unit_type), dtype=np.float32)
                scores = score_layer(
                    responses,
                    cache.events,
                    cache.scenario_ids,
                    config,
                    layer_id=layer_id * len(targets) + target_offset,
                    base_sample_ids=cache.base_sample_ids,
                )
                per_method = {
                    "paper_mi": (scores.mi_score, scores.global_spectrum.band_mi),
                    "paper_mi_gb": (scores.granular_score, scores.granular_band_mi),
                    "paper_mi_gb_lcb": (scores.lcb_score, scores.lcb_band_mean),
                    "paper_full": (scores.lcb_score, scores.lcb_band_mean),
                }
                for method, (scalar, bands) in per_method.items():
                    evidence[method][unit_type][layer_id] = {
                        "score": np.asarray(scalar, dtype=np.float64).copy(),
                        "bands": np.asarray(bands, dtype=np.float64).copy(),
                    }
                writer.add_layer(layer_id, unit_type, scores, {})
                ball_counts = [len(item.balls) for item in scores.granularities]
                fusion = [float(item.fusion_weight) for item in scores.granularities]
                print(
                    f"  {unit_type}: units={scores.mi_score.size}, balls={ball_counts}, "
                    f"fusion={[round(v, 4) for v in fusion]}, lcb_std_mean={scores.lcb_std.mean():.6g}",
                    flush=True,
                )
                if config.lcb.repeats > 1 and np.allclose(scores.lcb_std, 0.0):
                    print("  WARNING: repeated LCB has zero variance; inspect resampling diversity.", flush=True)
                if layer_id in plot_layers:
                    plot_granular_balls(
                        scores, cache.events, layer_id, unit_type,
                        report_dir / f"layer_{layer_id:03d}_{unit_type}_balls.png",
                    )
                    if config.granular_ball.localization_mode == "unit_local":
                        order = np.argsort(scores.lcb_score, kind="stable")
                        for label, unit_id in {
                            "low": int(order[0]),
                            "middle": int(order[len(order) // 2]),
                            "high": int(order[-1]),
                        }.items():
                            plot_unit_local_granular_balls(
                                scores, cache.events, layer_id, unit_type, unit_id,
                                config.granular_ball,
                                report_dir / f"layer_{layer_id:03d}_{unit_type}_unit_{unit_id:05d}_{label}_balls.png",
                            )
                    plot_lcb_scores(
                        scores, layer_id, unit_type,
                        report_dir / f"layer_{layer_id:03d}_{unit_type}_lcb.png",
                    )
                del responses, scores
    finally:
        writer.close()
    _save_evidence(evidence_path, evidence)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence


def _validate_cache_against_model(args, cache, model, targets: tuple[str, ...]) -> None:
    layers = getattr(getattr(model, "model", model), "layers")
    if len(layers) != cache.num_layers:
        raise ValueError(f"response cache has {cache.num_layers} layers but model has {len(layers)}")
    if cache.response_length != min(int(args.paper_response_length), int(args.paper_calib_seqlen)):
        # Older caches already encode their pooled response length; this check
        # protects accidental reuse under a different scoring configuration.
        if cache.response_length != int(args.paper_response_length):
            raise ValueError(
                f"response cache length {cache.response_length} != requested {args.paper_response_length}"
            )
    for layer_id, layer in enumerate(layers):
        if "mlp" in targets:
            expected = int(layer.mlp.down_proj.in_features)
            actual = int(cache.unit_counts["mlp"][layer_id])
            if expected != actual:
                raise ValueError(f"MLP unit count mismatch at layer {layer_id}: {actual} vs {expected}")
        if "attention" in targets:
            expected = int(
                getattr(layer.self_attn, "num_heads", 0)
                or getattr(model.config, "num_attention_heads", 0)
            )
            actual = int(cache.unit_counts["attention"][layer_id])
            if expected != actual:
                raise ValueError(f"attention unit count mismatch at layer {layer_id}: {actual} vs {expected}")


def prune_paper(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if prune_n or prune_m:
        raise ValueError("v8 supports exact unstructured weight zeroing only; N:M masks are not used")
    method = args.prune_method
    if method not in METHODS:
        raise ValueError(f"unsupported paper method: {method}")
    targets = _parse_targets(args.paper_prune_targets)

    # Uniform reference: skip all paper scoring and run the same sequential
    # per-output Wanda mask used by the explicit --prune_method wanda baseline.
    if args.paper_weight_allocation == "uniform":
        dataloader, _ = get_loaders(
            "c4", nsamples=args.wanda_nsamples, seed=args.seed,
            seqlen=args.wanda_calib_seqlen or model.seqlen, tokenizer=tokenizer,
        )
        return sequential_wanda_prune_(
            model, dataloader, nsamples=args.wanda_nsamples,
            seqlen=args.wanda_calib_seqlen or model.seqlen,
            sparsity=args.sparsity_ratio, budget=None,
            storage_mode=args.wanda_activation_storage,
            prune_order=args.prune_order,
            frozen_stats_cache_dir=args.wanda_stats_cache_dir,
        )

    cache_dir = Path(args.paper_cache_dir)
    if (cache_dir / "metadata.json").exists() and not args.paper_overwrite_cache:
        print(f"loading paper gradient-response cache from {cache_dir}")
        cache = load_response_cache(cache_dir)
    else:
        print("collecting paper task-gradient responses")
        dataloader, _ = get_loaders(
            args.paper_calib_dataset,
            nsamples=args.paper_score_nsamples,
            seed=args.seed,
            seqlen=args.paper_calib_seqlen,
            tokenizer=tokenizer,
        )
        cache = collect_gradient_response_cache(
            model,
            dataloader,
            cache_dir,
            scenario_ratios=args.paper_scenario_ratios,
            response_length=args.paper_response_length,
            event_bins=args.paper_event_bins,
            overwrite=args.paper_overwrite_cache,
        )

    _validate_cache_against_model(args, cache, model, targets)
    report_dir = Path(args.paper_report_dir)
    evidence = _score_or_load(args, cache, report_dir, targets)

    coverage_ratios = None
    if args.paper_band_coverage_ratios is not None:
        coverage_ratios = _parse_float_tuple(args.paper_band_coverage_ratios)

    effective_coverage_alpha = (
        float(args.paper_coverage_alpha) if bool(args.paper_use_band_gradient) else 0.0
    )
    print(
        f"[paper band gradient] enabled={bool(args.paper_use_band_gradient)}, "
        f"coverage_alpha={effective_coverage_alpha:.6g}",
        flush=True,
    )

    budget = allocate_weight_budget(
        model,
        method,
        evidence[method],
        targets,
        target_sparsity=args.sparsity_ratio,
        allocation=args.paper_weight_allocation,
        min_unit_sparsity=args.paper_weight_min_unit_sparsity,
        max_unit_sparsity=args.paper_weight_max_unit_sparsity,
        projection_temperature=args.paper_budget_temperature,
        coverage_ratio=args.paper_band_coverage_ratio,
        coverage_ratios=coverage_ratios,
        coverage_alpha=effective_coverage_alpha,
        greedy_batches=args.paper_greedy_batches,
    )
    print(
        f"[paper weight budget] method={method}, allocation={args.paper_weight_allocation}, "
        f"target={args.sparsity_ratio:.6f}, exact={budget.actual_sparsity:.9f}, "
        f"weights={budget.target_pruned_weights}/{budget.total_weights}",
        flush=True,
    )
    # Final element positions are selected by Wanda's activation-aware metric.
    # The paper contribution score only determines the 45%-55% unit quota.
    wanda_loader, _ = get_loaders(
        "c4", nsamples=args.wanda_nsamples, seed=args.seed,
        seqlen=args.wanda_calib_seqlen or model.seqlen, tokenizer=tokenizer,
    )
    summaries = sequential_wanda_prune_(
        model, wanda_loader, nsamples=args.wanda_nsamples,
        seqlen=args.wanda_calib_seqlen or model.seqlen,
        sparsity=args.sparsity_ratio, budget=budget,
        storage_mode=args.wanda_activation_storage,
        prune_order=args.prune_order,
        frozen_stats_cache_dir=args.wanda_stats_cache_dir,
    )
    budget_path = report_dir / f"weight_budget_{method}.csv"
    write_weight_budget(budget_path, budget, summaries)
    print(f"saved paper weight budget: {budget_path}", flush=True)
    return summaries
