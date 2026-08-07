from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .data import get_loaders
from .paper_pruning.apply_strict import apply_structured_pruning_
from .paper_pruning.budget_strict import select_keep_set_eq11_13
from .paper_pruning.collector_strict import (
    UNIT_TYPES,
    collect_gradient_response_cache,
    load_response_cache,
)
from .paper_pruning.config import (
    BudgetConfig,
    FrequencyConfig,
    GranularBallConfig,
    LCBConfig,
    PipelineConfig,
)
from .paper_pruning.pipeline import score_layer


PAPER_METHODS = {"paper_mi", "paper_mi_gb", "paper_mi_gb_lcb", "lcb", "paper_full"}
METHODS = ("paper_mi", "paper_mi_gb", "paper_mi_gb_lcb")


def _canonical_method(method: str) -> str:
    if method in {"lcb", "paper_full"}:
        return "paper_mi_gb_lcb"
    return method


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise ValueError("expected comma-separated floats")
    return values


def _parse_targets(raw: str) -> tuple[str, ...]:
    aliases = {"attn": "attention", "head": "attention", "heads": "attention", "ffn": "mlp"}
    result = []
    for item in raw.split(","):
        item = aliases.get(item.strip().lower(), item.strip().lower())
        if not item:
            continue
        if item not in UNIT_TYPES:
            raise ValueError("paper_prune_targets must contain mlp and/or attention")
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError("at least one paper target is required")
    return tuple(result)


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
            localization_mode=args.paper_gb_localization,
            workers=args.paper_gb_workers,
            worker_chunk_size=args.paper_gb_chunk_size,
            kde_scope=args.paper_kde_scope,
            random_state=args.seed,
        ),
        lcb=LCBConfig(
            repeats=args.n_samples_lcb,
            sample_fraction=args.paper_sample_fraction,
            scenario_fraction=args.paper_scenario_fraction,
            lcb_lambda=args.lcb_lambda,
            stratify_by_scenario=True,
            cluster_by_base_sample=True,
            random_state=args.seed,
            workers=args.paper_lcb_workers,
        ),
        budget=BudgetConfig(
            coverage_ratio=args.paper_band_coverage_ratio,
            coverage_alpha=args.paper_coverage_alpha,
            greedy_batches=args.paper_greedy_batches,
        ),
    )


def _validate_cache(model, cache, targets: tuple[str, ...]) -> None:
    layers = getattr(getattr(model, "model", model), "layers")
    if len(layers) != cache.num_layers:
        raise ValueError("response cache layer count does not match model")
    for layer_id, layer in enumerate(layers):
        if "mlp" in targets:
            current = int(layer.mlp.down_proj.in_features)
            if cache.unit_counts["mlp"].get(layer_id) != current:
                raise ValueError(f"MLP cache/model mismatch at layer {layer_id}")
        if "attention" in targets:
            expected = int(cache.attention_layouts[layer_id]["structural_units"])
            if cache.unit_counts["attention"].get(layer_id) != expected:
                raise ValueError(f"attention cache mismatch at layer {layer_id}")


def _unit_cost(model, cache, unit_type: str, layer_id: int, metric: str) -> float:
    if metric == "uniform":
        return 1.0
    if metric != "params":
        raise ValueError("paper_budget_metric must be params or uniform")
    layers = getattr(getattr(model, "model", model), "layers")
    layer = layers[layer_id]
    hidden = int(layer.self_attn.q_proj.in_features)
    if unit_type == "mlp":
        # gate row + up row + down column
        return float(3 * hidden)
    layout = cache.attention_layouts[layer_id]
    head_dim = int(layout["head_dim"])
    q_per_kv = int(layout["query_heads_per_kv"])
    # q rows + k row + v row + o columns for one complete GQA bundle.
    return float(2 * hidden * head_dim * (q_per_kv + 1))


def _score_bank_paths(report_dir: Path):
    return (
        report_dir / "strict_score_bank.npz",
        report_dir / "strict_score_config.json",
    )


def _score_config_payload(args, config, cache, targets):
    return {
        "pipeline": asdict(config),
        "targets": list(targets),
        "num_layers": cache.num_layers,
        "unit_counts": cache.unit_counts,
        "attention_layouts": cache.attention_layouts,
        "budget_metric": args.paper_budget_metric,
        "structural_sparsity_ratio": float(args.sparsity_ratio),
        "min_keep_per_block": int(args.paper_min_keep_per_block),
        "collector": "strict-gradient-only-v7",
    }


def _save_blocks(path: Path, blocks: Dict[tuple[str, int], dict]) -> None:
    arrays = {}
    for (unit_type, layer), block in blocks.items():
        prefix = f"{unit_type}__{layer}__"
        for key, value in block.items():
            if isinstance(value, np.ndarray):
                arrays[prefix + key] = value
    np.savez_compressed(path, **arrays)


def _load_blocks(path: Path) -> Dict[tuple[str, int], dict]:
    blocks: Dict[tuple[str, int], dict] = {}
    with np.load(path) as archive:
        for key in archive.files:
            unit_type, layer, field = key.split("__", 2)
            blocks.setdefault((unit_type, int(layer)), {})[field] = np.asarray(archive[key])
    return blocks


def _compute_or_load_blocks(args, model, cache, report_dir: Path, targets, config):
    bank_path, config_path = _score_bank_paths(report_dir)
    payload = _score_config_payload(args, config, cache, targets)
    if (
        bank_path.exists()
        and config_path.exists()
        and not args.paper_overwrite_scores
        and json.loads(config_path.read_text(encoding="utf-8")) == payload
    ):
        print(f"reusing strict paper score bank from {report_dir}")
        return _load_blocks(bank_path)

    blocks: Dict[tuple[str, int], dict] = {}
    for layer_id in range(cache.num_layers):
        print(f"[strict paper scoring] layer {layer_id + 1}/{cache.num_layers}")
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
            blocks[(unit_type, layer_id)] = {
                "mi_score": np.asarray(scores.mi_score, dtype=np.float64),
                "granular_score": np.asarray(scores.granular_score, dtype=np.float64),
                "lcb_mean": np.asarray(scores.lcb_mean, dtype=np.float64),
                "lcb_std": np.asarray(scores.lcb_std, dtype=np.float64),
                "lcb_score": np.asarray(scores.lcb_score, dtype=np.float64),
                "global_band_mi": np.asarray(scores.global_spectrum.band_mi, dtype=np.float64),
                "granular_band_mi": np.asarray(scores.granular_band_mi, dtype=np.float64),
                # Mean band contribution over dual-source repeats: used by Eq.(11)
                # for the full method; LCB itself remains the Eq.(11) objective.
                "repeat_band_mean": np.asarray(scores.lcb_band_mean, dtype=np.float64),
                "repeat_band_std": np.asarray(scores.lcb_band_std, dtype=np.float64),
            }
            del responses, scores

    report_dir.mkdir(parents=True, exist_ok=True)
    _save_blocks(bank_path, blocks)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return blocks


def _method_arrays(method: str, block: dict) -> tuple[np.ndarray, np.ndarray]:
    if method == "paper_mi":
        return block["mi_score"], block["global_band_mi"]
    if method == "paper_mi_gb":
        return block["granular_score"], block["granular_band_mi"]
    if method == "paper_mi_gb_lcb":
        # Eq.(11) objective is LCB. Coverage C_b(K) is built from robust
        # per-band contribution, not from an LCB-penalized band value.
        return block["lcb_score"], block["repeat_band_mean"]
    raise ValueError(method)


def _write_scores_csv(path: Path, method: str, blocks, keep_refs, costs_by_block) -> None:
    keep_set = set(keep_refs)
    band_count = next(iter(blocks.values()))["granular_band_mi"].shape[1]
    fields = [
        "method", "unit_type", "layer", "unit", "cost", "keep_in_K", "pruned",
        "mi_score", "granular_score", "lcb_mean", "lcb_std", "lcb_score",
    ]
    fields += [f"robust_band_{b}" for b in range(band_count)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (unit_type, layer), block in sorted(blocks.items(), key=lambda x: (x[0][1], x[0][0])):
            robust = _method_arrays(method, block)[1]
            cost = float(costs_by_block[(unit_type, layer)])
            for unit in range(block["lcb_score"].size):
                ref = (unit_type, int(layer), int(unit))
                row = {
                    "method": method, "unit_type": unit_type, "layer": layer, "unit": unit,
                    "cost": cost, "keep_in_K": ref in keep_set, "pruned": ref not in keep_set,
                    "mi_score": float(block["mi_score"][unit]),
                    "granular_score": float(block["granular_score"][unit]),
                    "lcb_mean": float(block["lcb_mean"][unit]),
                    "lcb_std": float(block["lcb_std"][unit]),
                    "lcb_score": float(block["lcb_score"][unit]),
                }
                for b in range(band_count):
                    row[f"robust_band_{b}"] = float(robust[unit, b])
                writer.writerow(row)


def prune_paper(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if prune_n or prune_m:
        raise ValueError("strict paper path uses complete structural units, not N:M masks")
    if not 0 < float(args.sparsity_ratio) < 1:
        raise ValueError("paper structural sparsity_ratio must be in (0,1)")

    method = _canonical_method(args.prune_method)
    if method not in METHODS:
        raise ValueError(f"unsupported strict paper method: {args.prune_method}")
    targets = _parse_targets(args.paper_prune_targets)
    config = _pipeline_config(args)
    config.validate()

    cache_dir = Path(args.paper_cache_dir)
    if (cache_dir / "metadata.json").exists() and not args.paper_overwrite_cache:
        print(f"loading strict gradient-response cache from {cache_dir}")
        cache = load_response_cache(cache_dir)
    else:
        print("loading calibration data for strict paper gradient responses")
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

    _validate_cache(model, cache, targets)
    report_dir = Path(args.paper_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    blocks = _compute_or_load_blocks(args, model, cache, report_dir, targets, config)

    scores = []
    bands = []
    costs = []
    refs = []
    block_global_indices: Dict[tuple[str, int], np.ndarray] = {}
    costs_by_block = {}
    cursor = 0

    for layer_id in range(cache.num_layers):
        for unit_type in targets:
            block = blocks[(unit_type, layer_id)]
            s, b = _method_arrays(method, block)
            unit_cost = _unit_cost(model, cache, unit_type, layer_id, args.paper_budget_metric)
            n = int(s.size)
            scores.append(np.asarray(s, dtype=np.float64))
            bands.append(np.asarray(b, dtype=np.float64))
            costs.append(np.full(n, unit_cost, dtype=np.float64))
            refs.extend((unit_type, layer_id, unit) for unit in range(n))
            block_global_indices[(unit_type, layer_id)] = np.arange(cursor, cursor + n, dtype=np.int64)
            costs_by_block[(unit_type, layer_id)] = unit_cost
            cursor += n

    score_all = np.concatenate(scores)
    band_all = np.concatenate(bands, axis=0)
    cost_all = np.concatenate(costs)
    total_cost = float(cost_all.sum())
    keep_budget = total_cost * (1.0 - float(args.sparsity_ratio))

    mandatory = []
    min_keep = max(1, int(args.paper_min_keep_per_block))
    for key, global_idx in block_global_indices.items():
        local_scores = score_all[global_idx]
        take = min(min_keep, global_idx.size)
        top = np.argsort(local_scores, kind="stable")[-take:]
        mandatory.extend(global_idx[top].tolist())

    selection = select_keep_set_eq11_13(
        score_all,
        band_all,
        cost_all,
        keep_budget,
        config.budget,
        mandatory_indices=np.asarray(mandatory, dtype=np.int64),
    )
    keep_global = set(selection.keep_indices.tolist())
    keep_refs = [refs[i] for i in selection.keep_indices]

    keep_by_type: Dict[str, Dict[int, list[int]]] = {t: {} for t in targets}
    prune_by_type: Dict[str, Dict[int, list[int]]] = {t: {} for t in targets}
    for (unit_type, layer_id), global_idx in block_global_indices.items():
        keep_local = [
            local for local, gi in enumerate(global_idx.tolist()) if gi in keep_global
        ]
        keep_local_set = set(keep_local)
        prune_local = [i for i in range(global_idx.size) if i not in keep_local_set]
        keep_by_type[unit_type][layer_id] = keep_local
        prune_by_type[unit_type][layer_id] = prune_local

    # K is the final keep set in the paper's notation.
    (report_dir / "keep_set_K.json").write_text(
        json.dumps(
            {
                t: {str(layer): values for layer, values in layer_map.items()}
                for t, layer_map in keep_by_type.items()
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    (report_dir / "prune_indices.json").write_text(
        json.dumps(
            {
                method: {
                    t: {str(layer): values for layer, values in layer_map.items()}
                    for t, layer_map in prune_by_type.items()
                }
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    _write_scores_csv(
        report_dir / "all_contribution_scores_strict.csv",
        method,
        blocks,
        keep_refs,
        costs_by_block,
    )

    before_params = int(sum(p.numel() for p in model.parameters()))
    apply_rows = apply_structured_pruning_(
        model, prune_by_type, mode=args.paper_apply_mode
    )
    after_params = int(sum(p.numel() for p in model.parameters()))

    total_units = sum(row["total"] for row in apply_rows)
    pruned_units = sum(row["pruned"] for row in apply_rows)
    by_type = {}
    for unit_type in targets:
        rows = [row for row in apply_rows if row["unit_type"] == unit_type]
        total = sum(row["total"] for row in rows)
        pruned = sum(row["pruned"] for row in rows)
        by_type[unit_type] = {
            "total_units": total,
            "pruned_units": pruned,
            "structural_unit_prune_rate": 0.0 if total == 0 else pruned / total,
        }

    stats = {
        "method": method,
        "apply_mode": args.paper_apply_mode,
        "budget_metric": args.paper_budget_metric,
        "requested_structural_cost_prune_ratio": float(args.sparsity_ratio),
        "actual_candidate_cost_prune_ratio": 1.0 - selection.spent_cost / selection.total_cost,
        "structural_unit_prune_rate": 0.0 if total_units == 0 else pruned_units / total_units,
        "whole_model_parameter_prune_rate": (
            0.0 if before_params == 0 else (before_params - after_params) / before_params
        ),
        "before_model_parameters": before_params,
        "after_model_parameters": after_params,
        "keep_budget": selection.budget,
        "spent_keep_cost": selection.spent_cost,
        "total_candidate_cost": selection.total_cost,
        "band_coverage_ratio": selection.achieved_coverage.tolist(),
        "hard_coverage_target_ratio": float(config.budget.coverage_ratio),
        "by_type": by_type,
        "K_size": len(keep_refs),
        "candidate_count": len(refs),
    }
    manifest = {
        "stats": stats,
        "apply_rows": apply_rows,
        "notes": {
            "attention_unit": "one KV head plus all query heads sharing it; equals one head for MHA",
            "save_reload": (
                "shrink mode changes per-layer tensor shapes. Evaluate in-memory; "
                "standard Hugging Face reload requires a structure-aware loader."
            ),
            "wanda_in_paper_path": False,
        },
    }
    (report_dir / "structural_pruning_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "budget_selection_eq11_13.json").write_text(
        json.dumps(
            {
                "budget": selection.budget,
                "spent_cost": selection.spent_cost,
                "total_cost": selection.total_cost,
                "achieved_coverage": selection.achieved_coverage.tolist(),
                "target_coverage": selection.target_coverage.tolist(),
                "final_coverage": selection.final_coverage.tolist(),
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[strict structured] method={method}, K={len(keep_refs)}/{len(refs)}, "
        f"cost-prune={stats['actual_candidate_cost_prune_ratio']:.6f}, "
        f"unit-prune={stats['structural_unit_prune_rate']:.6f}, "
        f"model-param-prune={stats['whole_model_parameter_prune_rate']:.6f}"
    )
    print(
        "[strict structured] band coverage="
        + "/".join(f"{x:.4f}" for x in selection.achieved_coverage)
    )
    return {
        "method": method,
        "K": keep_by_type,
        "prune": prune_by_type,
        "stats": stats,
        "manifest": manifest,
    }
