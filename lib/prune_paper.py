from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from .data import get_loaders
from .paper_pruning.apply import zero_structured_units_
from .paper_pruning.collector import collect_gradient_response_cache, load_response_cache
from .paper_pruning.config import (
    BudgetConfig,
    FrequencyConfig,
    GranularBallConfig,
    LCBConfig,
    PipelineConfig,
)
from .paper_pruning.global_budget import (
    CostComponents,
    build_cost_matrix,
    budget_limits_from_keep_ratios,
    coverage_aware_multi_budget_indices,
    unit_cost_components,
)
from .paper_pruning.materialize import materialize_structured_units_
from .paper_pruning.pipeline import score_layer_from_fine_energy
from .paper_pruning.reporting import LayerReportWriter, plot_lcb_scores
from .paper_pruning.scenarios import load_scenario_manifest
from .paper_pruning.stability import compare_score_reports


METHODS = ("paper_mi", "paper_mi_gb", "paper_mi_gb_lcb")
PAPER_METHODS = set(METHODS)
UNIT_TYPES = ("mlp", "attention")


def _json_digest(payload) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_float_tuple(raw: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(raw, str):
        return tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    return tuple(float(value) for value in raw)


def _parse_targets(raw: str | Sequence[str]) -> tuple[str, ...]:
    values = (
        [part.strip().lower() for part in raw.split(",") if part.strip()]
        if isinstance(raw, str)
        else [str(value).strip().lower() for value in raw]
    )
    aliases = {"ffn": "mlp", "heads": "attention", "head": "attention"}
    values = [aliases.get(value, value) for value in values]
    if not values or any(value not in UNIT_TYPES for value in values):
        raise ValueError("paper_prune_targets must contain mlp and/or attention")
    return tuple(dict.fromkeys(values))


def _parse_metrics(raw: str | Sequence[str]) -> tuple[str, ...]:
    values = (
        [part.strip().lower() for part in raw.split(",") if part.strip()]
        if isinstance(raw, str)
        else [str(value).strip().lower() for value in raw]
    )
    aliases = {"mem": "memory", "kv": "kv_cache", "time": "latency"}
    return tuple(dict.fromkeys(aliases.get(value, value) for value in values))


def _plot_layer_ids(raw: str, num_layers: int) -> set[int]:
    result: set[int] = set()
    for token in str(raw).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "first":
            result.add(0)
        elif token == "middle":
            result.add(num_layers // 2)
        elif token == "last":
            result.add(num_layers - 1)
        else:
            result.add(int(token))
    return {index for index in result if 0 <= index < num_layers}


def _model_signature(model) -> dict:
    config = getattr(model, "config", None)
    keys = (
        "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "vocab_size",
    )
    return {
        "name_or_path": str(getattr(config, "_name_or_path", "")),
        "config": {key: getattr(config, key, None) for key in keys},
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def _path_fingerprint(raw_path: str | Path | None, max_files: int = 512) -> dict | None:
    """Cheap reproducibility fingerprint for local model/data resources.

    Full model-shard hashing is unnecessarily expensive. File sizes and
    nanosecond modification times for configuration, tokenizer, index, shard
    and calibration-data files are enough to reject accidental stale-cache
    reuse while keeping cache creation fast. Remote model identifiers are
    recorded verbatim as non-local resources.
    """
    if raw_path is None or str(raw_path).strip() == "":
        return None
    source = Path(str(raw_path)).expanduser()
    if not source.exists():
        return {"value": str(raw_path), "local": False}
    if source.is_file():
        stat = source.stat()
        return {
            "path": str(source.resolve()),
            "local": True,
            "kind": "file",
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    allowed_suffixes = {
        ".json", ".jsonl", ".txt", ".model", ".safetensors", ".bin",
        ".arrow", ".parquet", ".csv", ".raw", ".gz",
    }
    important_names = {
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "special_tokens_map.json", "dataset_dict.json",
        "state.json",
    }
    files = []
    for candidate in source.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.name in important_names or candidate.suffix.lower() in allowed_suffixes:
            stat = candidate.stat()
            files.append({
                "relative_path": str(candidate.relative_to(source)),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
            if len(files) >= max_files:
                break
    files.sort(key=lambda item: item["relative_path"])
    return {
        "path": str(source.resolve()),
        "local": True,
        "kind": "directory",
        "files": files,
        "truncated": len(files) >= max_files,
    }


def _cache_signature(args, model, tokenizer) -> dict:
    data_identity = {
        "calibration_dataset": args.paper_calib_dataset,
        "scenario_manifest": _path_fingerprint(args.paper_scenario_manifest),
        "c4_path": _path_fingerprint(getattr(args, "c4_path", None)),
        "wikitext2_path": _path_fingerprint(getattr(args, "wikitext2_path", None)),
    }
    payload = {
        "model": {
            **_model_signature(model),
            "resource": _path_fingerprint(getattr(args, "model", None)),
        },
        "tokenizer": {
            "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
            "vocab_size": int(len(tokenizer)),
            "resource": _path_fingerprint(getattr(tokenizer, "name_or_path", None)),
        },
        "data": data_identity,
        "seed": int(args.seed),
        "score_nsamples": int(args.paper_score_nsamples),
        "calib_seqlen": int(args.paper_calib_seqlen),
        "fine_bins": int(args.paper_num_bins),
        "scenario_ratios": str(args.paper_scenario_ratios),
        "scenario_crops": str(args.paper_scenario_crops),
        "event_bins": int(args.paper_event_bins),
    }
    return {**payload, "digest": _json_digest(payload)}


def _pipeline_config(args) -> PipelineConfig:
    band_weights = _parse_float_tuple(args.paper_band_weights) if args.paper_band_weights else None
    granularity_weights = (
        _parse_float_tuple(args.paper_granularity_weights)
        if args.paper_granularity_weights
        else None
    )
    return PipelineConfig(
        frequency=FrequencyConfig(
            fine_bins=args.paper_num_bins,
            target_bands=args.paper_num_bands,
            mi_neighbors=args.paper_mi_neighbors,
            probe_units=args.paper_probe_units,
            band_weights=band_weights,
            kde_bandwidth_scale=args.paper_kde_bandwidth_scale,
            max_merge_loss=args.paper_max_merge_loss,
            random_state=args.seed,
        ),
        granular_ball=GranularBallConfig(
            purity_thresholds=_parse_float_tuple(args.paper_purity_thresholds),
            granularity_weights=granularity_weights,
            granularity_weight_mode=args.paper_granularity_weight_mode,
            min_ball_size=args.paper_min_ball_size,
            max_balls=args.paper_max_balls,
            max_depth=args.paper_max_ball_depth,
            min_purity_gain=args.paper_min_purity_gain,
            min_radius_reduction=args.paper_min_radius_reduction,
            compactness_ratio=args.paper_compactness_ratio,
            min_event_classes=args.paper_min_event_classes,
            workers=args.paper_gb_workers,
            worker_chunk_size=args.paper_gb_chunk_size,
            kde_scope=args.paper_kde_scope,
            random_state=args.seed,
        ),
        lcb=LCBConfig(
            repeats=args.paper_lcb_repeats,
            sample_fraction=args.paper_sample_fraction,
            scenario_fraction=args.paper_scenario_fraction,
            lcb_lambda=args.paper_lcb_lambda,
            stratify_by_scenario=True,
            cluster_by_base_sample=True,
            random_state=args.seed,
            workers=args.paper_lcb_workers,
        ),
        budget=BudgetConfig(
            coverage_ratio=args.paper_band_coverage_ratio,
            coverage_alpha=args.paper_coverage_alpha,
            fill_budget=args.paper_fill_budget,
        ),
    )


def _score_config_payload(
    args,
    config: PipelineConfig,
    cache,
    targets: tuple[str, ...],
    method: str,
) -> dict:
    return {
        "method": method,
        "pipeline": asdict(config),
        "targets": list(targets),
        "response_cache": str(cache.root.resolve()),
        "cache_signature": cache.cache_signature,
        "num_observations": cache.num_observations,
        "fine_bins": cache.fine_bins,
        "unit_counts": cache.unit_counts,
        "attention_layouts": cache.attention_layouts,
    }


def _save_nested_scores(
    path: Path,
    scores: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
) -> None:
    arrays = {}
    for method, type_map in scores.items():
        for unit_type, layer_map in type_map.items():
            for layer, values in layer_map.items():
                arrays[f"{method}__{unit_type}__{int(layer)}"] = np.asarray(values, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def _load_nested_scores(path: Path) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
    result: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {method: {} for method in METHODS}
    with np.load(path) as archive:
        for key in archive.files:
            method, unit_type, layer = key.split("__", 2)
            result.setdefault(method, {}).setdefault(unit_type, {})[int(layer)] = np.asarray(
                archive[key], dtype=np.float64
            )
    return result


def _calibration_loader(args, tokenizer):
    if args.paper_scenario_manifest:
        return load_scenario_manifest(
            args.paper_scenario_manifest,
            tokenizer,
            max_length=args.paper_calib_seqlen,
            limit=args.paper_score_nsamples,
        )
    dataloader, _ = get_loaders(
        args.paper_calib_dataset,
        nsamples=args.paper_score_nsamples,
        seed=args.seed,
        seqlen=args.paper_calib_seqlen,
        tokenizer=tokenizer,
    )
    return dataloader


def _load_or_collect_cache(args, model, tokenizer, cache_dir: Path, signature: dict):
    if (cache_dir / "metadata.json").exists() and not args.paper_overwrite_cache:
        print(f"loading response cache from {cache_dir}")
        return load_response_cache(cache_dir, expected_signature=signature)
    dataloader = _calibration_loader(args, tokenizer)
    return collect_gradient_response_cache(
        model,
        dataloader,
        cache_dir,
        scenario_ratios=args.paper_scenario_ratios,
        scenario_crops=args.paper_scenario_crops,
        fine_bins=args.paper_num_bins,
        event_bins=args.paper_event_bins,
        eps=_pipeline_config(args).frequency.eps,
        overwrite=args.paper_overwrite_cache,
        cache_signature=signature,
    )


def _validate_cache(cache, model, targets: tuple[str, ...]) -> None:
    layers = getattr(getattr(model, "model", model), "layers")
    if len(layers) != cache.num_layers:
        raise ValueError("response cache layer count does not match the model")
    for layer_id, layer in enumerate(layers):
        if "mlp" in targets and cache.unit_counts["mlp"][layer_id] != int(layer.mlp.down_proj.in_features):
            raise ValueError(f"layer {layer_id} FFN width differs from the response cache")
        if "attention" in targets:
            heads = int(getattr(layer.self_attn, "num_heads", 0) or getattr(model.config, "num_attention_heads", 0))
            if cache.unit_counts["attention"][layer_id] != heads:
                raise ValueError(f"layer {layer_id} head count differs from the response cache")


def _score_or_load(
    args,
    cache,
    report_dir: Path,
    targets: tuple[str, ...],
    method: str,
):
    """Score exactly one ablation stage and cache only that stage."""
    config = _pipeline_config(args)
    config.validate()
    payload = _score_config_payload(args, config, cache, targets, method)
    config_path = report_dir / "score_config.json"
    unit_path = report_dir / "unit_scores.npz"
    band_path = report_dir / "band_scores.npz"
    if (
        config_path.exists() and unit_path.exists() and band_path.exists()
        and not args.paper_overwrite_scores
        and json.loads(config_path.read_text(encoding="utf-8")) == payload
    ):
        print(f"reusing {method} contribution scores from {report_dir}")
        return _load_nested_scores(unit_path), _load_nested_scores(band_path), config

    report_dir.mkdir(parents=True, exist_ok=True)
    writer = LayerReportWriter(
        report_dir,
        band_count=config.frequency.target_bands,
        method=method,
    )
    plot_layers = _plot_layer_ids(args.paper_plot_layers, cache.num_layers)
    unit_scores: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {
        method: {unit_type: {} for unit_type in targets}
    }
    band_scores: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {
        method: {unit_type: {} for unit_type in targets}
    }
    stage_manifest = {
        "method": method,
        "granularity_weight_mode": config.granular_ball.granularity_weight_mode,
        "layers": [],
    }
    try:
        for layer_id in range(cache.num_layers):
            print(f"[scoring:{method}] layer {layer_id + 1}/{cache.num_layers}")
            for target_offset, unit_type in enumerate(targets):
                responses = np.asarray(cache.load_layer(layer_id, unit_type), dtype=np.float32)
                result = score_layer_from_fine_energy(
                    responses,
                    cache.events,
                    cache.scenario_ids,
                    config,
                    layer_id=layer_id * len(targets) + target_offset,
                    base_sample_ids=cache.base_sample_ids,
                    method=method,
                )
                unit_scores[method][unit_type][layer_id] = np.asarray(
                    result.selected_score, dtype=np.float64
                )
                band_scores[method][unit_type][layer_id] = np.asarray(
                    result.selected_band_score, dtype=np.float64
                )
                writer.add_layer(layer_id, unit_type, result, config.lcb.lcb_lambda)
                if method == "paper_mi_gb_lcb" and layer_id in plot_layers:
                    plot_lcb_scores(
                        result,
                        layer_id,
                        unit_type,
                        report_dir / f"layer_{layer_id:03d}_{unit_type}_lcb.png",
                    )
                stage_manifest["layers"].append({
                    "layer": layer_id,
                    "unit_type": unit_type,
                    "executed_stages": list(result.executed_stages),
                    "units": int(result.selected_score.size),
                })
                print(
                    f"  {unit_type}: units={result.selected_score.size}, "
                    f"selected_mean={np.mean(result.selected_score):.6g}, "
                    f"stages={','.join(result.executed_stages)}"
                )
    finally:
        writer.close()
    _save_nested_scores(unit_path, unit_scores)
    _save_nested_scores(band_path, band_scores)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "executed_stage_manifest.json").write_text(
        json.dumps(stage_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return unit_scores, band_scores, config

def _load_latency_profile(path: str | None) -> tuple[dict[tuple[str, int, int], float], dict[str, float]]:
    if not path:
        return {}, {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    values: dict[tuple[str, int, int], float] = {}
    defaults: dict[str, float] = {}
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                latency = float(row["latency"])
                unit_type = row["unit_type"].strip().lower()
                if row.get("layer", "").strip() and row.get("unit", "").strip():
                    values[(unit_type, int(row["layer"]), int(row["unit"]))] = latency
                else:
                    defaults[unit_type] = latency
    else:
        raw = json.loads(source.read_text(encoding="utf-8"))
        defaults = {str(key): float(value) for key, value in raw.get("defaults", {}).items()}
        for item in raw.get("units", []):
            values[(str(item["unit_type"]), int(item["layer"]), int(item["unit"]))] = float(item["latency"])
    return values, defaults


def _global_configuration(
    args,
    model,
    method: str,
    targets: tuple[str, ...],
    unit_scores,
    band_scores,
    config: PipelineConfig,
    output_path: Path,
) -> dict:
    metrics = _parse_metrics(args.paper_budget_metrics)
    latency_values, latency_defaults = _load_latency_profile(args.paper_latency_profile)
    if "latency" in metrics and not (latency_values or latency_defaults):
        raise ValueError("a latency budget requires --paper_latency_profile with measured costs")

    metadata: list[tuple[str, int, int]] = []
    score_rows: list[float] = []
    band_rows: list[np.ndarray] = []
    components: list[CostComponents] = []
    for unit_type in targets:
        for layer_id in sorted(unit_scores[method][unit_type]):
            values = np.asarray(unit_scores[method][unit_type][layer_id], dtype=np.float64)
            bands = np.asarray(band_scores[method][unit_type][layer_id], dtype=np.float64)
            for unit in range(values.size):
                metadata.append((unit_type, layer_id, unit))
                score_rows.append(float(values[unit]))
                band_rows.append(bands[unit])
                latency = latency_values.get(
                    (unit_type, layer_id, unit),
                    latency_defaults.get(unit_type, 0.0),
                )
                components.append(
                    unit_cost_components(
                        model.config,
                        unit_type,
                        sequence_length=args.seqlen,
                        dtype_bytes=args.paper_dtype_bytes,
                        latency=latency,
                    )
                )

    score_array = np.asarray(score_rows, dtype=np.float64)
    band_array = np.stack(band_rows, axis=0)
    cost_matrix = build_cost_matrix(components, metrics)
    if args.paper_budget_keep_ratios:
        keep_ratios = _parse_float_tuple(args.paper_budget_keep_ratios)
        if len(keep_ratios) == 1:
            keep_ratios = keep_ratios * len(metrics)
    else:
        keep_ratios = (1.0 - float(args.prune_ratio),) * len(metrics)
    limits = budget_limits_from_keep_ratios(cost_matrix, keep_ratios)
    keep, spent, achieved, feasible = coverage_aware_multi_budget_indices(
        score_array,
        band_array,
        cost_matrix,
        limits,
        config.budget,
        metric_weights=(
            _parse_float_tuple(args.paper_budget_metric_weights)
            if args.paper_budget_metric_weights else None
        ),
    )
    keep_set = set(keep.tolist())
    keep_indices = {unit_type: {} for unit_type in targets}
    prune_indices = {unit_type: {} for unit_type in targets}
    for global_index, (unit_type, layer_id, unit) in enumerate(metadata):
        target = keep_indices if global_index in keep_set else prune_indices
        target[unit_type].setdefault(layer_id, []).append(unit)
    for target in (keep_indices, prune_indices):
        for unit_type in targets:
            for layer_id in unit_scores[method][unit_type]:
                target[unit_type].setdefault(layer_id, [])

    payload = {
        "format_version": 2,
        "method": method,
        "keep_indices": {
            unit_type: {str(layer): indices for layer, indices in layer_map.items()}
            for unit_type, layer_map in keep_indices.items()
        },
        "prune_indices": {
            unit_type: {str(layer): indices for layer, indices in layer_map.items()}
            for unit_type, layer_map in prune_indices.items()
        },
        "budget": {
            "metrics": list(metrics),
            "keep_ratios": list(keep_ratios),
            "limits": limits.tolist(),
            "spent": spent.tolist(),
            "spent_ratios": (spent / np.maximum(limits, 1e-12)).tolist(),
            "coverage_target": config.budget.coverage_ratio,
            "coverage_achieved": achieved.tolist(),
            "coverage_feasible": feasible,
            "kept_units": int(keep.size),
            "total_units": int(len(metadata)),
            "unit_keep_ratio": float(keep.size / max(1, len(metadata))),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.paper_require_coverage and not feasible:
        raise RuntimeError(
            "the requested budgets cannot satisfy every frequency-band coverage constraint; "
            "increase keep ratios or lower paper_band_coverage_ratio"
        )
    return payload


def _load_selection_file(path: str | Path, method: str) -> Dict[str, Dict[int, np.ndarray]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "prune_indices" in raw:
        selected = raw["prune_indices"]
    elif method in raw:
        selected = raw[method]
    else:
        selected = raw
    return {
        unit_type: {
            int(layer): np.asarray(indices, dtype=np.int64)
            for layer, indices in layer_map.items()
        }
        for unit_type, layer_map in selected.items()
    }


def _apply_structured(args, model, selected, report_dir: Path) -> dict:
    if args.paper_mask_style == "structured_surgery":
        summary = materialize_structured_units_(
            model,
            selected,
            require_uniform=args.paper_require_uniform_surgery,
        )
    else:
        zero_structured_units_(model, selected)
        summary = {
            "mode": "structured_zero",
            "pruned_units": {
                unit_type: int(sum(len(values) for values in layer_map.values()))
                for unit_type, layer_map in selected.items()
            },
            "physical_shape_change": False,
        }
    (report_dir / "structural_application_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _run_post_prune_validation(
    args,
    model,
    tokenizer,
    targets: tuple[str, ...],
    pre_report_dir: Path,
    base_signature: dict,
) -> None:
    if not args.paper_post_prune_validate:
        return
    if args.paper_mask_style == "structured_surgery":
        print("post-prune spectrum comparison is skipped after physical surgery because unit indices change")
        return
    post_cache_dir = Path(args.paper_post_cache_dir)
    post_report_dir = Path(args.paper_post_report_dir)
    signature = {
        **base_signature,
        "stage": "post_prune",
        "method": args.prune_method,
        "prune_ratio": float(args.prune_ratio),
    }
    dataloader = _calibration_loader(args, tokenizer)
    post_cache = collect_gradient_response_cache(
        model,
        dataloader,
        post_cache_dir,
        scenario_ratios=args.paper_scenario_ratios,
        scenario_crops=args.paper_scenario_crops,
        fine_bins=args.paper_num_bins,
        event_bins=args.paper_event_bins,
        eps=_pipeline_config(args).frequency.eps,
        overwrite=args.paper_overwrite_post,
        cache_signature=signature,
    )
    _validate_cache(post_cache, model, targets)
    original = args.paper_overwrite_scores
    try:
        args.paper_overwrite_scores = bool(args.paper_overwrite_post)
        _score_or_load(args, post_cache, post_report_dir, targets, args.prune_method)
    finally:
        args.paper_overwrite_scores = original
    before = pre_report_dir / "all_contribution_scores.csv"
    after = post_report_dir / "all_contribution_scores.csv"
    if before.exists() and after.exists():
        compare_score_reports(
            before, after, post_report_dir / "pre_post_spectrum_stability.csv",
            method=args.prune_method,
        )


def prune_paper(args, model, tokenizer):
    method = args.prune_method
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    targets = _parse_targets(args.paper_prune_targets)
    report_dir = Path(args.paper_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.paper_selection_file:
        selected = _load_selection_file(args.paper_selection_file, method)
        summary = _apply_structured(args, model, selected, report_dir)
        return {"prune_indices": selected, "application": summary, "selection_file": args.paper_selection_file}

    signature = _cache_signature(args, model, tokenizer)
    cache = _load_or_collect_cache(args, model, tokenizer, Path(args.paper_cache_dir), signature)
    _validate_cache(cache, model, targets)
    unit_scores, band_scores, config = _score_or_load(args, cache, report_dir, targets, method)
    if args.paper_score_only:
        return {"score_only": True, "report_dir": str(report_dir)}

    selection_path = report_dir / f"global_selection_{method}.json"
    selection_payload = _global_configuration(
        args,
        model,
        method,
        targets,
        unit_scores,
        band_scores,
        config,
        selection_path,
    )
    selected = _load_selection_file(selection_path, method)
    application = _apply_structured(args, model, selected, report_dir)
    _run_post_prune_validation(args, model, tokenizer, targets, report_dir, signature)
    return {
        "selection_file": str(selection_path),
        "selection": selection_payload["budget"],
        "prune_indices": selected,
        "application": application,
    }
