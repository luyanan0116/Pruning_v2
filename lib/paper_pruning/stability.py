from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import kendalltau, spearmanr


def ranking_stability_summary(
    repeated_scores: np.ndarray,
    top_fraction: float = 0.10,
) -> Dict[str, float]:
    values = np.asarray(repeated_scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("repeated_scores must be [repeats,units] with at least two repeats")
    top_k = min(values.shape[1], max(1, int(round(values.shape[1] * top_fraction))))
    taus, rhos, jaccards = [], [], []
    for left, right in combinations(range(values.shape[0]), 2):
        tau = kendalltau(values[left], values[right], nan_policy="omit").statistic
        rho = spearmanr(values[left], values[right], nan_policy="omit").statistic
        left_top = set(np.argsort(values[left], kind="stable")[-top_k:].tolist())
        right_top = set(np.argsort(values[right], kind="stable")[-top_k:].tolist())
        union = left_top | right_top
        taus.append(float(tau) if np.isfinite(tau) else 0.0)
        rhos.append(float(rho) if np.isfinite(rho) else 0.0)
        jaccards.append(1.0 if not union else len(left_top & right_top) / len(union))
    return {
        "pair_count": float(len(taus)),
        "mean_kendall_tau": float(np.mean(taus)),
        "min_kendall_tau": float(np.min(taus)),
        "mean_spearman_rho": float(np.mean(rhos)),
        "min_spearman_rho": float(np.min(rhos)),
        "mean_top_jaccard": float(np.mean(jaccards)),
        "min_top_jaccard": float(np.min(jaccards)),
        "top_k": float(top_k),
    }


def compare_score_reports(
    before_csv: str | Path,
    after_csv: str | Path,
    output_csv: str | Path,
    top_fraction: float = 0.10,
    method: str | None = None,
) -> Path:
    """Compare the selected stage score before and after pruning."""

    def load(path):
        rows = {}
        methods = set()
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row_method = row.get("method", "")
                if row_method:
                    methods.add(row_method)
                key = (row["unit_type"], int(row["layer"]), int(row["unit"]))
                rows[key] = row
        return rows, methods

    before, before_methods = load(before_csv)
    after, after_methods = load(after_csv)
    inferred = before_methods & after_methods
    if method is None:
        if len(inferred) != 1:
            raise ValueError("could not infer one common scoring method from the reports")
        method = next(iter(inferred))
    common = sorted(set(before) & set(after))
    if not common:
        raise ValueError("the two score reports have no common structural units")
    left = np.asarray([float(before[key]["selected_score"]) for key in common], dtype=np.float64)
    right = np.asarray([float(after[key]["selected_score"]) for key in common], dtype=np.float64)
    tau = kendalltau(left, right, nan_policy="omit").statistic
    rho = spearmanr(left, right, nan_policy="omit").statistic
    top_k = min(left.size, max(1, int(round(left.size * top_fraction))))
    a = set(np.argsort(left, kind="stable")[-top_k:].tolist())
    b = set(np.argsort(right, kind="stable")[-top_k:].tolist())
    result = [{
        "method": method,
        "common_units": len(common),
        "kendall_tau": float(tau) if np.isfinite(tau) else "",
        "spearman_rho": float(rho) if np.isfinite(rho) else "",
        "top_k": top_k,
        "top_jaccard": 1.0 if not (a | b) else len(a & b) / len(a | b),
        "mean_abs_score_change": float(np.mean(np.abs(left - right))),
    }]
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    return output
