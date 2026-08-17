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
    BudgetConfig,
    FrequencyConfig,
    GranularBallConfig,
    LCBConfig,
    PipelineConfig,
)
from .paper_pruning.budget import select_keep_indices
from .paper_pruning.pipeline import score_layer
from .paper_pruning.reporting import (
    LayerReportWriter,
    plot_granular_balls,
    plot_unit_local_granular_balls,
    plot_lcb_scores,
    write_selection_files,
    write_global_selection_status,
)
from .paper_pruning.selection import resolve_prune_count, select_bottom_k
from .paper_pruning.wanda_sequential import apply_sequential_paper_wanda_masks_
from .paper_pruning.wanda_weight import (
    ALL_LINEAR_MODULES,
    apply_paper_wanda_weight_masks_,
    apply_paper_unit_budget_weight_masks_,
    write_weight_mask_summary,
)


PAPER_METHODS = {"paper_mi", "paper_mi_gb", "paper_mi_gb_lcb", "paper_full", "lcb"}
METHODS = ("paper_mi", "paper_mi_gb", "paper_mi_gb_lcb", "paper_full")


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
            band_selection_weights=(
                None if args.paper_band_selection_weights is None
                else _parse_float_tuple(args.paper_band_selection_weights)
            ),
            coverage_alpha=args.paper_coverage_alpha,
            greedy_batches=args.paper_greedy_batches,
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
        "final_keep_strategy": args.paper_final_keep_strategy,
        "global_budget": bool(args.paper_global_budget),
        "global_layer_spread": float(args.paper_global_layer_spread),
        "ablation_design": "MI(score-only)->GB(score-only)->LCB(score-only)->Full(LCB+coverage)",
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
    """Score every layer first, then allocate the pruning budget globally.

    The clean ablation is intentionally fixed as:
      paper_mi         = MI score only
      paper_mi_gb      = granular-ball score only
      paper_mi_gb_lcb  = true repeated-estimation LCB score only
      paper_full       = LCB + frequency-coverage marginal gain

    With ``--paper_global_budget`` (default), the total budget for each unit
    type is pooled across all layers.  The total number of pruned units remains
    exactly the sum of the legacy per-layer budgets, but sensitive layers are no
    longer forced to prune the same fraction as redundant layers.
    """
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

    # Only compact vectors are retained for the later cross-layer optimizer.
    # The large [samples, units, bands] tensors inside LayerAblationScores are
    # released immediately after per-layer diagnostics are written.
    compact: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {
        unit_type: {} for unit_type in targets
    }

    coverage_targets = (
        config.budget.coverage_ratios
        if config.budget.coverage_ratios is not None
        else tuple([config.budget.coverage_ratio] * config.frequency.target_bands)
    )
    print(
        f"[paper ablation] MI=score-only, GB=score-only, LCB=score-only, "
        f"Full={args.paper_final_keep_strategy}; global_budget={args.paper_global_budget}; "
        f"bands={config.frequency.target_bands}, coverage_targets={coverage_targets}",
        flush=True,
    )

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
                    base_sample_ids=cache.base_sample_ids,
                )
                compact[unit_type][layer_id] = {
                    "mi_score": np.asarray(scores.mi_score, dtype=np.float64).copy(),
                    "gb_score": np.asarray(scores.granular_score, dtype=np.float64).copy(),
                    "lcb_score": np.asarray(scores.lcb_score, dtype=np.float64).copy(),
                    "mi_bands": np.asarray(scores.global_spectrum.band_mi, dtype=np.float64).copy(),
                    "gb_bands": np.asarray(scores.granular_band_mi, dtype=np.float64).copy(),
                    "lcb_bands": np.asarray(scores.lcb_band_mean, dtype=np.float64).copy(),
                }

                # Cross-layer masks are not known yet.  Keep the contribution
                # report truthful by leaving prune flags blank; final flags are
                # written after global selection to global_selection_status.csv.
                writer.add_layer(layer_id, unit_type, scores, {})

                ratio, exact_count = _target_budget(args, unit_type)
                nominal_prune = resolve_prune_count(scores.mi_score.size, ratio, exact_count)
                ball_counts = [len(item.balls) for item in scores.granularities]
                fusion = [float(item.fusion_weight) for item in scores.granularities]
                print(
                    f"  {unit_type}: units={scores.mi_score.size}, nominal_prune={nominal_prune}, "
                    f"balls={ball_counts}, fusion={[round(v, 4) for v in fusion]}, "
                    f"lcb_std_mean={scores.lcb_std.mean():.6g}"
                )
                if len(set(ball_counts)) == 1:
                    print(
                        "  NOTE: purity thresholds produced identical ball counts; "
                        "the fusion can still differ by local MI, but inspect the ball report."
                    )
                if config.lcb.repeats > 1 and np.allclose(scores.lcb_std, 0.0):
                    print(
                        "  WARNING: repeated LCB still has zero variance; inspect bootstrap diversity."
                    )
                if layer_id in plot_layers:
                    plot_granular_balls(
                        scores,
                        cache.events,
                        layer_id,
                        unit_type,
                        report_dir / f"layer_{layer_id:03d}_{unit_type}_balls_diagnostic.png",
                    )
                    if config.granular_ball.localization_mode == "unit_local":
                        order = np.argsort(scores.lcb_score, kind="stable")
                        representative = {
                            "low": int(order[0]),
                            "boundary": int(order[min(nominal_prune, order.size - 1)]),
                            "high": int(order[-1]),
                        }
                        for label, unit_id in representative.items():
                            plot_unit_local_granular_balls(
                                scores, cache.events, layer_id, unit_type, unit_id,
                                config.granular_ball,
                                report_dir / (
                                    f"layer_{layer_id:03d}_{unit_type}_unit_{unit_id:05d}_"
                                    f"{label}_local_balls.png"
                                ),
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

    method_score_key = {
        "paper_mi": "mi_score",
        "paper_mi_gb": "gb_score",
        "paper_mi_gb_lcb": "lcb_score",
        "paper_full": "lcb_score",
    }
    method_band_key = {
        "paper_mi": "mi_bands",
        "paper_mi_gb": "gb_bands",
        "paper_mi_gb_lcb": "lcb_bands",
        "paper_full": "lcb_bands",
    }
    budget_summary = {
        "global_budget": bool(args.paper_global_budget),
        "global_layer_spread": float(args.paper_global_layer_spread),
        "ablation": {
            "paper_mi": "MI scalar score only",
            "paper_mi_gb": "multi-granularity local-MI scalar score only",
            "paper_mi_gb_lcb": "bootstrap LCB scalar score only",
            "paper_full": f"LCB + {args.paper_final_keep_strategy}",
        },
        "unit_types": {},
    }

    for unit_type in targets:
        layer_ids = sorted(compact[unit_type])
        ratio, exact_count = _target_budget(args, unit_type)
        nominal_prune = {
            layer_id: resolve_prune_count(
                compact[unit_type][layer_id]["mi_score"].size, ratio, exact_count
            )
            for layer_id in layer_ids
        }

        if args.paper_global_budget:
            sizes = [compact[unit_type][layer_id]["mi_score"].size for layer_id in layer_ids]
            offsets = np.cumsum([0] + sizes)
            total_units = int(offsets[-1])
            prune_total = int(sum(nominal_prune.values()))
            keep_total = total_units - prune_total
            all_indices = np.arange(total_units, dtype=np.int64)
            type_summary = {
                "total_units": total_units,
                "prune_total": prune_total,
                "keep_total": keep_total,
                "nominal_per_layer_prune": {str(k): int(v) for k, v in nominal_prune.items()},
                "methods": {},
            }

            for method in METHODS:
                score_vector = np.concatenate([
                    compact[unit_type][layer_id][method_score_key[method]]
                    for layer_id in layer_ids
                ])
                band_vector = np.concatenate([
                    compact[unit_type][layer_id][method_band_key[method]]
                    for layer_id in layer_ids
                ], axis=0)
                strategy = args.paper_final_keep_strategy if method == "paper_full" else "lcb_only"
                keep, priority, achieved = select_keep_indices(
                    strategy,
                    score_vector,
                    band_vector,
                    keep_total,
                    config.budget,
                )
                prune_global = np.setdiff1d(all_indices, keep, assume_unique=True)
                per_layer_counts = {}
                for position, layer_id in enumerate(layer_ids):
                    start, end = int(offsets[position]), int(offsets[position + 1])
                    local_prune = prune_global[(prune_global >= start) & (prune_global < end)] - start
                    selections[method][unit_type][layer_id] = local_prune.astype(np.int64)
                    unit_scores[method][unit_type][layer_id] = priority[start:end].astype(np.float32)
                    per_layer_counts[str(layer_id)] = int(local_prune.size)
                type_summary["methods"][method] = {
                    "strategy": strategy,
                    "achieved_band_coverage": [float(v) for v in achieved],
                    "per_layer_prune": per_layer_counts,
                }
                print(
                    f"[global budget] {unit_type} {method}: prune={prune_total}/{total_units}, "
                    f"layer_prune_range={min(per_layer_counts.values())}-{max(per_layer_counts.values())}, "
                    f"coverage={'/'.join(f'{v:.3f}' for v in achieved)}",
                    flush=True,
                )
            budget_summary["unit_types"][unit_type] = type_summary
        else:
            type_summary = {"methods": {}, "nominal_per_layer_prune": {str(k): int(v) for k, v in nominal_prune.items()}}
            for method in METHODS:
                strategy = args.paper_final_keep_strategy if method == "paper_full" else "lcb_only"
                method_coverage = {}
                for layer_id in layer_ids:
                    block = compact[unit_type][layer_id]
                    unit_count = block["mi_score"].size
                    keep_count = unit_count - nominal_prune[layer_id]
                    keep, priority, achieved = select_keep_indices(
                        strategy,
                        block[method_score_key[method]],
                        block[method_band_key[method]],
                        keep_count,
                        config.budget,
                    )
                    selections[method][unit_type][layer_id] = np.setdiff1d(
                        np.arange(unit_count, dtype=np.int64), keep, assume_unique=True
                    )
                    unit_scores[method][unit_type][layer_id] = priority.astype(np.float32)
                    method_coverage[str(layer_id)] = [float(v) for v in achieved]
                type_summary["methods"][method] = {
                    "strategy": strategy,
                    "achieved_band_coverage_by_layer": method_coverage,
                }
            budget_summary["unit_types"][unit_type] = type_summary

    write_selection_files(report_dir, selections, args.paper_prune_step)
    status_path = write_global_selection_status(
        report_dir, selections, {unit_type: cache.unit_counts[unit_type] for unit_type in targets}
    )
    _save_unit_scores(unit_scores_path, unit_scores)
    (report_dir / "global_budget_summary.json").write_text(
        json.dumps(budget_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved final selection flags: {status_path}")
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
        ("paper_mi_gb_lcb", "paper_full"),
        ("paper_mi", "paper_full"),
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
    if args.paper_mask_style == "wanda_weight" and not args.paper_wanda_sequential:
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


def _derive_global_layer_weight_ratios(
    model,
    selected: Mapping[str, Mapping[int, np.ndarray]],
    targets: tuple[str, ...],
    mlp_ratio: float,
    attention_ratio: float,
    spread: float,
    unit_budget: bool = False,
    min_unit_sparsity: float = 0.30,
    max_unit_sparsity: float = 0.70,
) -> Dict[str, Dict[int, float]]:
    """Map global structural allocation to bounded layer-wise weight budgets.

    The raw signal is each layer's fraction of globally-pruned structural units.
    We center that signal using parameter-count weights and scale it around the
    requested global weight sparsity.  The weighted mean therefore remains the
    original target, while layer ratios can move by at most ``spread``.
    """
    layers = getattr(getattr(model, "model", model), "layers")
    result: Dict[str, Dict[int, float]] = {}
    eps = 1e-4
    for unit_type in targets:
        base_ratio = float(attention_ratio if unit_type == "attention" else mlp_ratio)
        raw = []
        parameter_weights = []
        layer_ids = sorted(selected.get(unit_type, {}))
        if not layer_ids:
            continue
        for layer_id in layer_ids:
            layer = layers[layer_id]
            if unit_type == "mlp":
                unit_count = int(layer.mlp.down_proj.in_features)
                parameter_count = sum(
                    int(module.weight.numel())
                    for module in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
                )
            else:
                attn = layer.self_attn
                unit_count = int(
                    getattr(attn, "num_heads", 0)
                    or getattr(model.config, "num_attention_heads", 0)
                )
                parameter_count = sum(
                    int(module.weight.numel())
                    for module in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj)
                )
            raw.append(len(selected[unit_type][layer_id]) / float(unit_count))
            parameter_weights.append(float(parameter_count))

        raw_arr = np.asarray(raw, dtype=np.float64)
        weight_arr = np.asarray(parameter_weights, dtype=np.float64)
        center = float(np.average(raw_arr, weights=weight_arr))
        dev = raw_arr - center

        low = max(eps, base_ratio - float(spread))
        high = min(1.0 - eps, base_ratio + float(spread))
        if unit_budget:
            low = max(low, float(min_unit_sparsity))
            high = min(high, float(max_unit_sparsity))
        if not low <= base_ratio <= high:
            raise ValueError(
                f"global layer budget bounds [{low:.4f},{high:.4f}] do not contain "
                f"target {base_ratio:.4f} for {unit_type}"
            )

        scale = 1.0
        positive = dev[dev > 0]
        negative = dev[dev < 0]
        if positive.size:
            scale = min(scale, (high - base_ratio) / float(positive.max()))
        if negative.size:
            scale = min(scale, (base_ratio - low) / float(-negative.min()))
        scale = max(0.0, min(1.0, scale))
        ratios = base_ratio + scale * dev

        # Floating-point correction keeps the parameter-weighted mean at the
        # requested target without changing ordering or exceeding bounds.
        correction = base_ratio - float(np.average(ratios, weights=weight_arr))
        ratios = np.clip(ratios + correction, low, high)
        result[unit_type] = {
            int(layer_id): float(ratio)
            for layer_id, ratio in zip(layer_ids, ratios)
        }
        weighted_mean = float(np.average(ratios, weights=weight_arr))
        print(
            f"[global weight budget] {unit_type}: target={base_ratio:.4f}, "
            f"layer_range={ratios.min():.4f}-{ratios.max():.4f}, "
            f"parameter_weighted_mean={weighted_mean:.6f}",
            flush=True,
        )
    return result


def _print_applied_budget(
    model,
    selected: Mapping[str, Mapping[int, np.ndarray]],
) -> None:
    layers = getattr(getattr(model, "model", model), "layers")
    for unit_type, layer_map in selected.items():
        if not layer_map:
            continue
        prune_counts = []
        total_units = 0
        total_pruned = 0
        for layer_id, indices in sorted(layer_map.items()):
            if unit_type == "mlp":
                count = int(layers[layer_id].mlp.down_proj.in_features)
            else:
                attn = layers[layer_id].self_attn
                count = int(
                    getattr(attn, "num_heads", 0)
                    or getattr(model.config, "num_attention_heads", 0)
                )
            pruned = int(len(indices))
            prune_counts.append(pruned)
            total_units += count
            total_pruned += pruned
        print(
            f"applied {unit_type} structured sparsity: {total_pruned}/{total_units} "
            f"= {total_pruned / total_units:.6f} globally; "
            f"per-layer prune range={min(prune_counts)}-{max(prune_counts)}"
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
    unit_budget = args.paper_mask_style == "unit_budget_weight"
    layer_ratios = None
    if args.paper_global_budget:
        layer_ratios = _derive_global_layer_weight_ratios(
            model,
            selected,
            targets,
            mlp_ratio=mlp_ratio,
            attention_ratio=attention_ratio,
            spread=args.paper_global_layer_spread,
            unit_budget=unit_budget,
            min_unit_sparsity=args.paper_unit_min_sparsity,
            max_unit_sparsity=args.paper_unit_max_sparsity,
        )
        (report_dir / f"weight_layer_budget_{method}.json").write_text(
            json.dumps(
                {unit_type: {str(k): v for k, v in layer_map.items()}
                 for unit_type, layer_map in layer_ratios.items()},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
    if unit_budget:
        print(
            f"applying {method} score-driven unit-budget weight masks; "
            f"mlp_ratio={mlp_ratio:.4f}, attention_ratio={attention_ratio:.4f}, "
            f"unit_sparsity=[{args.paper_unit_min_sparsity:.2f},"
            f"{args.paper_unit_max_sparsity:.2f}]"
        )
    else:
        print(
            f"applying {method} paper-guided Wanda weight masks for targets={','.join(targets)}; "
            f"mlp_ratio={mlp_ratio:.4f}, attention_ratio={attention_ratio:.4f}"
        )

    if args.paper_wanda_sequential:
        summaries = apply_sequential_paper_wanda_masks_(
            model,
            tokenizer,
            unit_scores[method],
            targets,
            mlp_ratio=mlp_ratio,
            attention_ratio=attention_ratio,
            calib_dataset=args.paper_calib_dataset,
            nsamples=args.paper_wanda_nsamples,
            seqlen=args.paper_wanda_seqlen,
            seed=args.seed,
            row_spread=args.paper_wanda_row_spread,
            row_temperature=args.paper_wanda_temperature,
            guidance_strength=args.paper_wanda_guidance_strength,
            chunk_rows=args.paper_wanda_chunk_rows,
            unit_budget=unit_budget,
            min_unit_sparsity=args.paper_unit_min_sparsity,
            max_unit_sparsity=args.paper_unit_max_sparsity,
            layer_ratios=layer_ratios,
        )
    elif unit_budget:
        summaries = apply_paper_unit_budget_weight_masks_(
            model,
            unit_scores[method],
            cache,
            targets,
            mlp_ratio=mlp_ratio,
            attention_ratio=attention_ratio,
            min_unit_sparsity=args.paper_unit_min_sparsity,
            max_unit_sparsity=args.paper_unit_max_sparsity,
            chunk_rows=args.paper_wanda_chunk_rows,
            layer_ratios=layer_ratios,
        )
    else:
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
            guidance_strength=args.paper_wanda_guidance_strength,
            chunk_rows=args.paper_wanda_chunk_rows,
            layer_ratios=layer_ratios,
        )
    summary_path = report_dir / f"weight_mask_summary_{method}.csv"
    write_weight_mask_summary(summary_path, summaries)
    print(f"saved weight mask summary: {summary_path}")
    return summaries
