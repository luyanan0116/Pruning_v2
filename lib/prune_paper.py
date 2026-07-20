from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from .data import get_loaders
from .paper_pruning.apply import zero_mlp_channels_
from .paper_pruning.collector import (
    collect_gradient_response_cache,
    load_response_cache,
)
from .paper_pruning.config import (
    FrequencyConfig,
    GranularBallConfig,
    LCBConfig,
    PipelineConfig,
)
from .paper_pruning.pipeline import score_layer
from .paper_pruning.reporting import (
    LayerReportWriter,
    plot_granular_balls,
    plot_lcb_scores,
    write_selection_files,
)
from .paper_pruning.selection import resolve_prune_count, select_bottom_k


PAPER_METHODS = {"paper_mi", "paper_mi_gb", "paper_mi_gb_lcb", "lcb"}


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated float")
    return values


def _canonical_method(method: str) -> str:
    return "paper_mi_gb_lcb" if method == "lcb" else method


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
            random_state=args.seed,
        ),
        granular_ball=GranularBallConfig(
            purity_thresholds=_parse_float_tuple(args.paper_purity_thresholds),
            min_ball_size=args.paper_min_ball_size,
            max_balls=args.paper_max_balls,
            min_purity_gain=args.paper_min_purity_gain,
            min_radius_reduction=args.paper_min_radius_reduction,
            compactness_ratio=args.paper_compactness_ratio,
            min_event_classes=args.paper_min_event_classes,
            random_state=args.seed,
        ),
        lcb=LCBConfig(
            repeats=args.n_samples_lcb,
            sample_fraction=args.paper_sample_fraction,
            lcb_lambda=args.lcb_lambda,
            stratify_by_scenario=True,
            random_state=args.seed,
        ),
    )


def _config_payload(args, config: PipelineConfig, cache) -> dict:
    payload = {
        "pipeline": asdict(config),
        "sparsity_ratio": float(args.sparsity_ratio),
        "prune_per_layer": int(args.prune_per_layer),
        "prune_step": int(args.paper_prune_step),
        "response_cache": str(cache.root.resolve()),
        "num_observations": cache.num_observations,
        "response_length": cache.response_length,
        "unit_counts": cache.unit_counts,
    }
    # JSON normalization turns tuples into lists so saved/current configs compare exactly.
    return json.loads(json.dumps(payload))


def _load_selection_file(path: Path) -> Dict[str, Dict[int, np.ndarray]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        method: {int(layer): np.asarray(indices, dtype=np.int64) for layer, indices in layer_map.items()}
        for method, layer_map in raw.items()
    }


def _score_or_load(args, cache, report_dir: Path) -> Dict[str, Dict[int, np.ndarray]]:
    config = _pipeline_config(args)
    payload = _config_payload(args, config, cache)
    config_path = report_dir / "score_config.json"
    indices_path = report_dir / "prune_indices.json"

    if indices_path.exists() and config_path.exists() and not args.paper_overwrite_scores:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous == payload:
            print(f"reusing paper-aligned scores from {report_dir}")
            return _load_selection_file(indices_path)
        print("paper score configuration changed; recomputing scores")

    report_dir.mkdir(parents=True, exist_ok=True)
    plot_layers = _plot_layer_ids(args.paper_plot_layers, cache.num_layers)
    writer = LayerReportWriter(
        report_dir,
        band_count=config.frequency.target_bands,
        prune_step=args.paper_prune_step,
    )
    selections: Dict[str, Dict[int, np.ndarray]] = {
        "paper_mi": {},
        "paper_mi_gb": {},
        "paper_mi_gb_lcb": {},
    }

    try:
        for layer_id in range(cache.num_layers):
            print(f"[paper scoring] layer {layer_id + 1}/{cache.num_layers}")
            responses = np.asarray(cache.load_layer(layer_id), dtype=np.float32)
            scores = score_layer(
                responses,
                cache.events,
                cache.scenario_ids,
                config,
                layer_id=layer_id,
            )
            prune_count = resolve_prune_count(
                scores.mi_score.size,
                args.sparsity_ratio,
                args.prune_per_layer,
            )
            layer_selections = {
                "paper_mi": select_bottom_k(scores.mi_score, prune_count),
                "paper_mi_gb": select_bottom_k(scores.granular_score, prune_count),
                "paper_mi_gb_lcb": select_bottom_k(scores.lcb_score, prune_count),
            }
            for method, indices in layer_selections.items():
                selections[method][layer_id] = indices
            writer.add_layer(layer_id, scores, layer_selections)

            ball_counts = [len(item.balls) for item in scores.granularities]
            print(
                f"  units={scores.mi_score.size}, prune={prune_count}, "
                f"balls={ball_counts}, lcb_std_mean={scores.lcb_std.mean():.6g}"
            )
            if len(set(ball_counts)) == 1:
                print("  WARNING: purity thresholds produced identical ball counts; consider lowering min_ball_size or min_event_classes.")
            if np.allclose(scores.lcb_std, 0.0):
                print("  WARNING: all LCB standard deviations are zero; check repeat count and scenario/event diversity.")
            if layer_id in plot_layers:
                plot_granular_balls(
                    scores,
                    cache.events,
                    layer_id,
                    report_dir / f"layer_{layer_id:03d}_balls.png",
                )
                plot_lcb_scores(
                    scores,
                    layer_id,
                    report_dir / f"layer_{layer_id:03d}_lcb.png",
                )
            del responses, scores
    finally:
        writer.close()

    write_selection_files(report_dir, selections, args.paper_prune_step)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return selections


def _print_mask_overlap(selections: Mapping[str, Mapping[int, np.ndarray]]) -> None:
    def flatten(mask):
        return {(layer, int(unit)) for layer, values in mask.items() for unit in values}

    pairs = [
        ("paper_mi", "paper_mi_gb"),
        ("paper_mi_gb", "paper_mi_gb_lcb"),
        ("paper_mi", "paper_mi_gb_lcb"),
    ]
    for left, right in pairs:
        a, b = flatten(selections[left]), flatten(selections[right])
        union = a | b
        jaccard = 1.0 if not union else len(a & b) / len(union)
        changed = len(a ^ b)
        print(f"mask overlap {left} vs {right}: Jaccard={jaccard:.6f}, changed={changed}")
        if changed == 0:
            print("  WARNING: masks are identical, so identical PPL is expected.")



def _validate_cache_against_model(cache, model) -> None:
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None or len(layers) != cache.num_layers:
        raise ValueError(
            "response cache does not match the current model layer count; "
            "use --paper_overwrite_cache or a model-specific --paper_cache_dir"
        )
    for layer_id, layer in enumerate(layers):
        unit_count = int(layer.mlp.down_proj.in_features)
        if cache.unit_counts.get(layer_id) != unit_count:
            raise ValueError(
                f"response cache layer {layer_id} has {cache.unit_counts.get(layer_id)} units, "
                f"current model has {unit_count}; rebuild the cache"
            )

def prune_paper(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if prune_n or prune_m:
        raise ValueError("paper-aligned pruning supports structured MLP channels, not N:M weight sparsity")
    method = _canonical_method(args.prune_method)
    if method not in {"paper_mi", "paper_mi_gb", "paper_mi_gb_lcb"}:
        raise ValueError(f"unsupported paper method: {args.prune_method}")

    cache_dir = Path(args.paper_cache_dir)
    if (cache_dir / "metadata.json").exists() and not args.paper_overwrite_cache:
        print(f"loading gradient-response cache from {cache_dir}")
        cache = load_response_cache(cache_dir)
    else:
        print("loading calibration data for paper-aligned gradient responses")
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

    _validate_cache_against_model(cache, model)
    report_dir = Path(args.paper_report_dir)
    selections = _score_or_load(args, cache, report_dir)
    _print_mask_overlap(selections)
    print(f"applying {method} structured MLP channel mask")
    zero_mlp_channels_(model, selections[method])
    return selections[method]
