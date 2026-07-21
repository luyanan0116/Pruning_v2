from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import torch

from .data import get_loaders
from .paper_pruning.apply import zero_structured_units_
from .paper_pruning.collector import (
    UNIT_TYPES,
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
from .paper_pruning.wanda_weight import (
    ALL_LINEAR_MODULES,
    apply_paper_wanda_weight_masks_,
    write_weight_mask_summary,
)


PAPER_METHODS = {"paper_mi", "paper_mi_gb", "paper_mi_gb_lcb", "lcb"}
METHODS = ("paper_mi", "paper_mi_gb", "paper_mi_gb_lcb")


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated float")
    return values


def _parse_targets(raw: str) -> tuple[str, ...]:
    aliases = {"attn": "attention", "head": "attention", "heads": "attention", "ffn": "mlp"}
    targets = []
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        value = aliases.get(value, value)
        if value not in UNIT_TYPES:
            raise ValueError(f"unknown paper prune target: {value}; choose from mlp,attention")
        if value not in targets:
            targets.append(value)
    if not targets:
        raise ValueError("at least one paper prune target is required")
    return tuple(targets)


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


def _target_budget(args, unit_type: str) -> tuple[float, int]:
    if unit_type == "mlp":
        ratio = args.mlp_sparsity_ratio
        exact = args.prune_per_layer
    elif unit_type == "attention":
        ratio = args.attention_sparsity_ratio
        exact = args.attention_prune_per_layer
    else:
        raise ValueError(unit_type)
    if ratio is None:
        ratio = args.sparsity_ratio
    return float(ratio), int(exact)


def _config_payload(args, config: PipelineConfig, cache, targets: tuple[str, ...]) -> dict:
    payload = {
        "pipeline": asdict(config),
        "targets": list(targets),
        "budgets": {
            unit_type: {
                "sparsity_ratio": _target_budget(args, unit_type)[0],
                "prune_per_layer": _target_budget(args, unit_type)[1],
            }
            for unit_type in targets
        },
        "prune_step": int(args.paper_prune_step),
        "mask_style": args.paper_mask_style,
        "wanda_weight": {
            "score_floor": float(args.paper_wanda_score_floor),
            "row_spread": float(args.paper_wanda_row_spread),
            "temperature": float(args.paper_wanda_temperature),
            "chunk_rows": int(args.paper_wanda_chunk_rows),
        },
        "response_cache": str(cache.root.resolve()),
        "num_observations": cache.num_observations,
        "response_length": cache.response_length,
        "unit_counts": cache.unit_counts,
        "attention_layouts": cache.attention_layouts,
    }
    return json.loads(json.dumps(payload))


def _load_selection_file(path: Path) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        method: {
            unit_type: {
                int(layer): np.asarray(indices, dtype=np.int64)
                for layer, indices in layer_map.items()
            }
            for unit_type, layer_map in target_map.items()
        }
        for method, target_map in raw.items()
    }


def _save_unit_scores(
    path: Path,
    scores: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
) -> None:
    arrays = {}
    for method, type_map in scores.items():
        for unit_type, layer_map in type_map.items():
            for layer_id, values in layer_map.items():
                arrays[f"{method}__{unit_type}__{int(layer_id)}"] = np.asarray(values, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def _load_unit_scores(path: Path) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
    result: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {method: {} for method in METHODS}
    with np.load(path) as archive:
        for key in archive.files:
            method, unit_type, layer_raw = key.split("__", 2)
            result.setdefault(method, {}).setdefault(unit_type, {})[int(layer_raw)] = np.asarray(
                archive[key], dtype=np.float32
            )
    return result


def _score_or_load(
    args,
    cache,
    report_dir: Path,
    targets: tuple[str, ...],
) -> Tuple[
    Dict[str, Dict[str, Dict[int, np.ndarray]]],
    Dict[str, Dict[str, Dict[int, np.ndarray]]],
]:
    config = _pipeline_config(args)
    payload = _config_payload(args, config, cache, targets)
    config_path = report_dir / "score_config.json"
    indices_path = report_dir / "prune_indices.json"
    unit_scores_path = report_dir / "unit_scores.npz"

    if indices_path.exists() and unit_scores_path.exists() and config_path.exists() and not args.paper_overwrite_scores:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous == payload:
            print(f"reusing paper-aligned scores from {report_dir}")
            return _load_selection_file(indices_path), _load_unit_scores(unit_scores_path)
        print("paper score configuration changed; recomputing scores")

    report_dir.mkdir(parents=True, exist_ok=True)
    plot_layers = _plot_layer_ids(args.paper_plot_layers, cache.num_layers)
    writer = LayerReportWriter(
        report_dir,
        band_count=config.frequency.target_bands,
        prune_step=args.paper_prune_step,
    )
    selections: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {
        method: {unit_type: {} for unit_type in targets}
        for method in METHODS
    }
    unit_scores: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {
        method: {unit_type: {} for unit_type in targets}
        for method in METHODS
    }

    try:
        for layer_id in range(cache.num_layers):
            print(f"[paper scoring] layer {layer_id + 1}/{cache.num_layers}")
            for target_offset, unit_type in enumerate(targets):
                responses = np.asarray(cache.load_layer(layer_id, unit_type), dtype=np.float32)
                scores = score_layer(
                    responses,
                    cache.events,
                    cache.scenario_ids,
                    config,
                    layer_id=layer_id * len(targets) + target_offset,
                )
                ratio, exact_count = _target_budget(args, unit_type)
                prune_count = resolve_prune_count(
                    scores.mi_score.size,
                    ratio,
                    exact_count,
                )
                layer_score_vectors = {
                    "paper_mi": np.asarray(scores.mi_score, dtype=np.float32),
                    "paper_mi_gb": np.asarray(scores.granular_score, dtype=np.float32),
                    "paper_mi_gb_lcb": np.asarray(scores.lcb_score, dtype=np.float32),
                }
                layer_selections = {
                    method: select_bottom_k(values, prune_count)
                    for method, values in layer_score_vectors.items()
                }
                for method, values in layer_score_vectors.items():
                    unit_scores[method][unit_type][layer_id] = values.copy()
                for method, indices in layer_selections.items():
                    selections[method][unit_type][layer_id] = indices
                writer.add_layer(layer_id, unit_type, scores, layer_selections)

                ball_counts = [len(item.balls) for item in scores.granularities]
                print(
                    f"  {unit_type}: units={scores.mi_score.size}, prune={prune_count}, "
                    f"ratio={prune_count / scores.mi_score.size:.4f}, "
                    f"balls={ball_counts}, lcb_std_mean={scores.lcb_std.mean():.6g}"
                )
                if len(set(ball_counts)) == 1:
                    print(
                        "  WARNING: purity thresholds produced identical ball counts; "
                        "consider lowering min_ball_size or min_event_classes."
                    )
                if np.allclose(scores.lcb_std, 0.0):
                    print(
                        "  WARNING: all LCB standard deviations are zero; "
                        "check repeat count and scenario/event diversity."
                    )
                if layer_id in plot_layers:
                    plot_granular_balls(
                        scores,
                        cache.events,
                        layer_id,
                        unit_type,
                        report_dir / f"layer_{layer_id:03d}_{unit_type}_balls.png",
                    )
                    plot_lcb_scores(
                        scores,
                        layer_id,
                        unit_type,
                        report_dir / f"layer_{layer_id:03d}_{unit_type}_lcb.png",
                    )
                del responses, scores
    finally:
        writer.close()

    write_selection_files(report_dir, selections, args.paper_prune_step)
    _save_unit_scores(unit_scores_path, unit_scores)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return selections, unit_scores


def _print_mask_overlap(
    selections: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]]
) -> None:
    def flatten(mask):
        return {
            (unit_type, layer, int(unit))
            for unit_type, layer_map in mask.items()
            for layer, values in layer_map.items()
            for unit in values
        }

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


def _validate_cache_against_model(args, cache, model, targets: tuple[str, ...]) -> None:
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None or len(layers) != cache.num_layers:
        raise ValueError(
            "response cache does not match the current model layer count; "
            "use --paper_overwrite_cache or a model-specific --paper_cache_dir"
        )
    for layer_id, layer in enumerate(layers):
        if "mlp" in targets:
            unit_count = int(layer.mlp.down_proj.in_features)
            if cache.unit_counts["mlp"].get(layer_id) != unit_count:
                raise ValueError(
                    f"response cache layer {layer_id} has {cache.unit_counts['mlp'].get(layer_id)} MLP units, "
                    f"current model has {unit_count}; rebuild the cache"
                )
        if "attention" in targets:
            attn = layer.self_attn
            num_heads = int(
                getattr(attn, "num_heads", 0)
                or getattr(model.config, "num_attention_heads", 0)
            )
            num_kv_heads = int(
                getattr(attn, "num_key_value_heads", 0)
                or getattr(model.config, "num_key_value_heads", num_heads)
            )
            if num_heads != num_kv_heads:
                raise ValueError(
                    "paper MLP+attention scoring currently expects standard multi-head attention "
                    f"with equal query/KV head counts; layer {layer_id} has {num_heads}/{num_kv_heads}."
                )
            if cache.unit_counts["attention"].get(layer_id) != num_heads:
                raise ValueError(
                    f"response cache layer {layer_id} has "
                    f"{cache.unit_counts['attention'].get(layer_id)} attention heads, "
                    f"current model has {num_heads}; rebuild the cache"
                )
    if args.paper_mask_style == "wanda_weight":
        required = set()
        if "attention" in targets:
            required.update(name for name in ALL_LINEAR_MODULES if name.startswith("self_attn."))
        if "mlp" in targets:
            required.update(name for name in ALL_LINEAR_MODULES if name.startswith("mlp."))
        for layer_id in range(cache.num_layers):
            for module_name in required:
                path = cache.activation_scale_path(layer_id, module_name)
                if not path.exists():
                    raise ValueError(
                        f"response cache lacks Wanda activation statistics for {module_name}; "
                        "rebuild it with --paper_overwrite_cache"
                    )


def _print_applied_budget(
    model,
    selected: Mapping[str, Mapping[int, np.ndarray]],
) -> None:
    layers = getattr(getattr(model, "model", model), "layers")
    for unit_type, layer_map in selected.items():
        first_layer = min(layer_map) if layer_map else None
        if first_layer is None:
            continue
        pruned = len(layer_map[first_layer])
        if unit_type == "mlp":
            total = int(layers[first_layer].mlp.down_proj.in_features)
        else:
            attn = layers[first_layer].self_attn
            total = int(
                getattr(attn, "num_heads", 0)
                or getattr(model.config, "num_attention_heads", 0)
            )
        print(
            f"applied {unit_type} structured sparsity: "
            f"{pruned}/{total} = {pruned / total:.6f} per layer"
        )


def prune_paper(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if prune_n or prune_m:
        raise ValueError("paper-aligned pruning currently supports unstructured Wanda-style weights or structured units, not N:M masks")
    method = _canonical_method(args.prune_method)
    if method not in METHODS:
        raise ValueError(f"unsupported paper method: {args.prune_method}")
    targets = _parse_targets(args.paper_prune_targets)

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

    _validate_cache_against_model(args, cache, model, targets)
    report_dir = Path(args.paper_report_dir)
    selections, unit_scores = _score_or_load(args, cache, report_dir, targets)
    _print_mask_overlap(selections)
    selected = selections[method]

    if args.paper_mask_style == "structured_unit":
        print(f"applying {method} structured masks for targets={','.join(targets)}")
        zero_structured_units_(model, selected)
        _print_applied_budget(model, selected)
        return selected

    mlp_ratio, _ = _target_budget(args, "mlp")
    attention_ratio, _ = _target_budget(args, "attention")
    print(
        f"applying {method} paper-guided Wanda weight masks for targets={','.join(targets)}; "
        f"mlp_ratio={mlp_ratio:.4f}, attention_ratio={attention_ratio:.4f}"
    )
    summaries = apply_paper_wanda_weight_masks_(
        model,
        unit_scores[method],
        cache,
        targets,
        mlp_ratio=mlp_ratio,
        attention_ratio=attention_ratio,
        score_floor=args.paper_wanda_score_floor,
        row_spread=args.paper_wanda_row_spread,
        temperature=args.paper_wanda_temperature,
        chunk_rows=args.paper_wanda_chunk_rows,
    )
    summary_path = report_dir / f"weight_mask_summary_{method}.csv"
    write_weight_mask_summary(summary_path, summaries)
    print(f"saved Wanda-style weight mask summary: {summary_path}")
    return summaries
