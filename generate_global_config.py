from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lib.paper_pruning.config import BudgetConfig
from lib.paper_pruning.global_budget import (
    CostComponents,
    budget_limits_from_keep_ratios,
    build_cost_matrix,
    coverage_aware_multi_budget_indices,
    normalized_effective_cost,
    unit_cost_components,
)


METHOD_COLUMNS = {
    "paper_mi": ("mi_total", "global_band_mi"),
    "paper_mi_gb": ("granular_total", "local_band_mi"),
    "paper_mi_gb_lcb": ("lcb_score", "lcb_band_score"),
}


def _tuple(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _metrics(raw: str) -> tuple[str, ...]:
    aliases = {"mem": "memory", "kv": "kv_cache", "time": "latency"}
    return tuple(
        dict.fromkeys(
            aliases.get(part.strip().lower(), part.strip().lower())
            for part in raw.split(",")
            if part.strip()
        )
    )


def _load_config(args):
    if args.config_json:
        return SimpleNamespace(**json.loads(Path(args.config_json).read_text(encoding="utf-8")))
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(args.model, cache_dir=args.cache_dir)


def _load_latency(path: str | None):
    if not path:
        return {}, {}
    source = Path(path)
    exact, defaults = {}, {}
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                unit_type = row["unit_type"].strip().lower()
                value = float(row["latency"])
                if row.get("layer", "").strip() and row.get("unit", "").strip():
                    exact[(unit_type, int(row["layer"]), int(row["unit"]))] = value
                else:
                    defaults[unit_type] = value
    else:
        raw = json.loads(source.read_text(encoding="utf-8"))
        defaults = {str(key): float(value) for key, value in raw.get("defaults", {}).items()}
        for item in raw.get("units", []):
            exact[(str(item["unit_type"]), int(item["layer"]), int(item["unit"]))] = float(item["latency"])
    return exact, defaults


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the global LCB/coverage/deployment-budget structural configuration from a score report."
    )
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--output", default=None)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--config_json")
    parser.add_argument("--cache_dir", default="llm_weights")
    parser.add_argument("--method", choices=tuple(METHOD_COLUMNS), default="paper_mi_gb_lcb")
    parser.add_argument("--prune_ratio", type=float, default=0.15)
    parser.add_argument("--budget_metrics", default="params,flops,memory")
    parser.add_argument("--budget_keep_ratios", default=None)
    parser.add_argument("--budget_metric_weights", default=None)
    parser.add_argument("--latency_profile", default=None)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--dtype_bytes", type=int, default=2)
    parser.add_argument("--coverage_ratio", type=float, default=0.90)
    parser.add_argument("--coverage_alpha", type=float, default=0.25)
    parser.add_argument("--fill_budget", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_coverage", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not 0 <= args.prune_ratio < 1:
        parser.error("--prune_ratio must be in [0,1)")
    report_dir = Path(args.report_dir)
    score_path = report_dir / "all_contribution_scores.csv"
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    output = Path(args.output) if args.output else report_dir / f"global_selection_{args.method}.json"
    with score_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty score report")

    model_config = _load_config(args)
    score_column, band_prefix = METHOD_COLUMNS[args.method]
    band_count = sum(key.startswith("global_band_mi_") for key in rows[0])
    metrics = _metrics(args.budget_metrics)
    latency_exact, latency_defaults = _load_latency(args.latency_profile)
    if "latency" in metrics and not (latency_exact or latency_defaults):
        raise ValueError("latency is active but no measured latency profile was provided")

    keys: list[tuple[str, int, int]] = []
    scores, bands, components = [], [], []
    for row in rows:
        unit_type = row["unit_type"]
        layer = int(row["layer"])
        unit = int(row["unit"])
        keys.append((unit_type, layer, unit))
        scores.append(float(row[score_column]))
        bands.append([float(row[f"{band_prefix}_{band}"]) for band in range(band_count)])
        latency = latency_exact.get((unit_type, layer, unit), latency_defaults.get(unit_type, 0.0))
        components.append(
            unit_cost_components(
                model_config,
                unit_type,
                sequence_length=args.seqlen,
                dtype_bytes=args.dtype_bytes,
                latency=latency,
            )
        )

    cost_matrix = build_cost_matrix(components, metrics)
    ratios = _tuple(args.budget_keep_ratios)
    if ratios is None:
        ratios = (1.0 - args.prune_ratio,) * len(metrics)
    elif len(ratios) == 1:
        ratios = ratios * len(metrics)
    limits = budget_limits_from_keep_ratios(cost_matrix, ratios)
    weights = _tuple(args.budget_metric_weights)
    keep, spent, achieved, feasible = coverage_aware_multi_budget_indices(
        np.asarray(scores, dtype=np.float64),
        np.asarray(bands, dtype=np.float64),
        cost_matrix,
        limits,
        BudgetConfig(
            coverage_ratio=args.coverage_ratio,
            coverage_alpha=args.coverage_alpha,
            fill_budget=args.fill_budget,
        ),
        metric_weights=weights,
    )
    keep_set = set(keep.tolist())
    keep_map: dict[str, dict[str, list[int]]] = {}
    prune_map: dict[str, dict[str, list[int]]] = {}
    effective_cost = normalized_effective_cost(cost_matrix, limits, weights)
    detail = []
    for index, (unit_type, layer, unit) in enumerate(keys):
        target = keep_map if index in keep_set else prune_map
        target.setdefault(unit_type, {}).setdefault(str(layer), []).append(unit)
        component = components[index].as_dict()
        detail.append(
            {
                "unit_type": unit_type,
                "layer": layer,
                "unit": unit,
                "score": scores[index],
                "effective_cost": effective_cost[index],
                **component,
                "kept": index in keep_set,
            }
        )

    payload = {
        "format_version": 2,
        "method": args.method,
        "keep_indices": keep_map,
        "prune_indices": prune_map,
        "budget": {
            "metrics": list(metrics),
            "keep_ratios": list(ratios),
            "metric_weights": None if weights is None else list(weights),
            "limits": limits.tolist(),
            "spent": spent.tolist(),
            "coverage_target": args.coverage_ratio,
            "coverage_achieved": achieved.tolist(),
            "coverage_feasible": feasible,
            "kept_units": int(keep.size),
            "total_units": len(keys),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with output.with_name(output.stem + "_units.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    print(json.dumps(payload["budget"], ensure_ascii=False, indent=2))
    if args.require_coverage and not feasible:
        raise RuntimeError("the requested budgets cannot meet all frequency-band coverage constraints")


if __name__ == "__main__":
    main()
