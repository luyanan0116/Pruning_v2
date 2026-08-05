from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr

from .pipeline import LayerAblationScores
from .stability import ranking_stability_summary


METHOD_KEYS = ("paper_mi", "paper_mi_gb", "paper_mi_gb_lcb")


def _finite_or_blank(value) -> float | str:
    value = float(value)
    return value if np.isfinite(value) else ""


def _safe_rank_stats(primary: np.ndarray, auxiliary: np.ndarray) -> tuple[float | str, float | str]:
    left = np.asarray(primary, dtype=np.float64)
    right = np.asarray(auxiliary, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return "", ""
    tau = kendalltau(left[valid], right[valid], nan_policy="omit").statistic
    rho = spearmanr(left[valid], right[valid], nan_policy="omit").statistic
    return _finite_or_blank(tau), _finite_or_blank(rho)


class LayerReportWriter:
    def __init__(self, output_dir: str | Path, band_count: int, method: str):
        if method not in METHOD_KEYS:
            raise ValueError(f"unsupported report method: {method}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.band_count = int(band_count)
        self.method = method
        self.score_file = (self.output_dir / "all_contribution_scores.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.band_file = (self.output_dir / "frequency_band_boundaries.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.ball_file = (self.output_dir / "granular_ball_summary.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.consistency_file = (self.output_dir / "mi_estimator_consistency.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.stability_file = (self.output_dir / "bootstrap_ranking_stability.csv").open(
            "w", newline="", encoding="utf-8"
        )

        score_fields = [
            "method", "executed_stages", "unit_type", "layer", "unit",
            "selected_score", "mi_total", "granular_total",
            "lcb_mean", "lcb_std", "lcb_score",
        ]
        for band in range(self.band_count):
            score_fields.extend([
                f"selected_band_score_{band}",
                f"global_band_mi_{band}", f"global_band_kde_{band}",
                f"local_band_mi_{band}", f"local_band_kde_{band}",
                f"lcb_band_mean_{band}", f"lcb_band_std_{band}",
                f"lcb_band_score_{band}",
            ])
        self.score_writer = csv.DictWriter(self.score_file, fieldnames=score_fields)
        self.band_writer = csv.DictWriter(
            self.band_file,
            fieldnames=[
                "method", "unit_type", "layer", "band", "start_fine_bin",
                "end_fine_bin", "probe_count", "probe_indices", "fine_relevance",
            ],
        )
        self.ball_writer = csv.DictWriter(
            self.ball_file,
            fieldnames=[
                "method", "unit_type", "layer", "granularity", "purity_threshold",
                "fusion_weight", "repeat_variance", "mean_ball_count",
                "mean_ball_size", "mean_purity", "mean_radius", "mean_max_depth",
            ],
        )
        self.consistency_writer = csv.DictWriter(
            self.consistency_file,
            fieldnames=["method", "unit_type", "layer", "scope", "kendall_tau", "spearman_rho"],
        )
        self.stability_writer = csv.DictWriter(
            self.stability_file,
            fieldnames=[
                "method", "unit_type", "layer", "pair_count", "mean_kendall_tau",
                "min_kendall_tau", "mean_spearman_rho", "min_spearman_rho",
                "mean_top_jaccard", "min_top_jaccard", "top_k",
            ],
        )
        for writer in (
            self.score_writer, self.band_writer, self.ball_writer,
            self.consistency_writer, self.stability_writer,
        ):
            writer.writeheader()

    def add_layer(self, layer_id: int, unit_type: str, scores: LayerAblationScores, lcb_lambda: float) -> None:
        if scores.method != self.method:
            raise ValueError("score method does not match report method")
        spectrum = scores.global_spectrum
        probes = np.asarray(spectrum.probe_indices, dtype=np.int64)
        relevance = np.asarray(spectrum.fine_relevance, dtype=np.float64)
        for band, (start, end) in enumerate(spectrum.band_ranges):
            self.band_writer.writerow({
                "method": self.method,
                "unit_type": unit_type,
                "layer": layer_id,
                "band": band,
                "start_fine_bin": start,
                "end_fine_bin": end,
                "probe_count": int(probes.size),
                "probe_indices": json.dumps(probes.tolist()),
                "fine_relevance": json.dumps(relevance[start:end].tolist()),
            })

        tau, rho = _safe_rank_stats(
            spectrum.band_mi.mean(axis=1),
            spectrum.band_mi_kde.mean(axis=1),
        )
        self.consistency_writer.writerow({
            "method": self.method,
            "unit_type": unit_type,
            "layer": layer_id,
            "scope": "global",
            "kendall_tau": tau,
            "spearman_rho": rho,
        })
        if np.isfinite(scores.granular_band_mi).any():
            tau, rho = _safe_rank_stats(
                scores.granular_band_mi.mean(axis=1),
                scores.granular_band_mi_kde.mean(axis=1),
            )
            self.consistency_writer.writerow({
                "method": self.method,
                "unit_type": unit_type,
                "layer": layer_id,
                "scope": "granular",
                "kendall_tau": tau,
                "spearman_rho": rho,
            })

        if scores.bootstrap_scores.ndim == 2 and scores.bootstrap_scores.shape[0] >= 2:
            self.stability_writer.writerow({
                "method": self.method,
                "unit_type": unit_type,
                "layer": layer_id,
                **ranking_stability_summary(scores.bootstrap_scores),
            })

        robust_band = scores.lcb_band_mean - float(lcb_lambda) * scores.lcb_band_std
        stages = json.dumps(list(scores.executed_stages), ensure_ascii=False)
        for unit in range(scores.mi_score.size):
            row = {
                "method": self.method,
                "executed_stages": stages,
                "unit_type": unit_type,
                "layer": layer_id,
                "unit": unit,
                "selected_score": _finite_or_blank(scores.selected_score[unit]),
                "mi_total": _finite_or_blank(scores.mi_score[unit]),
                "granular_total": _finite_or_blank(scores.granular_score[unit]),
                "lcb_mean": _finite_or_blank(scores.lcb_mean[unit]),
                "lcb_std": _finite_or_blank(scores.lcb_std[unit]),
                "lcb_score": _finite_or_blank(scores.lcb_score[unit]),
            }
            for band in range(self.band_count):
                row[f"selected_band_score_{band}"] = _finite_or_blank(
                    scores.selected_band_score[unit, band]
                )
                row[f"global_band_mi_{band}"] = _finite_or_blank(spectrum.band_mi[unit, band])
                row[f"global_band_kde_{band}"] = _finite_or_blank(spectrum.band_mi_kde[unit, band])
                row[f"local_band_mi_{band}"] = _finite_or_blank(scores.granular_band_mi[unit, band])
                row[f"local_band_kde_{band}"] = _finite_or_blank(scores.granular_band_mi_kde[unit, band])
                row[f"lcb_band_mean_{band}"] = _finite_or_blank(scores.lcb_band_mean[unit, band])
                row[f"lcb_band_std_{band}"] = _finite_or_blank(scores.lcb_band_std[unit, band])
                row[f"lcb_band_score_{band}"] = _finite_or_blank(robust_band[unit, band])
            self.score_writer.writerow(row)

        for granularity_id, granularity in enumerate(scores.granularities):
            self.ball_writer.writerow({
                "method": self.method,
                "unit_type": unit_type,
                "layer": layer_id,
                "granularity": granularity_id,
                "purity_threshold": granularity.purity_threshold,
                "fusion_weight": _finite_or_blank(granularity.fusion_weight),
                "repeat_variance": _finite_or_blank(granularity.variance),
                "mean_ball_count": granularity.mean_ball_count,
                "mean_ball_size": granularity.mean_ball_size,
                "mean_purity": granularity.mean_purity,
                "mean_radius": granularity.mean_radius,
                "mean_max_depth": granularity.mean_max_depth,
            })
        for handle in (
            self.score_file, self.band_file, self.ball_file,
            self.consistency_file, self.stability_file,
        ):
            handle.flush()

    def close(self) -> None:
        for handle in (
            self.score_file, self.band_file, self.ball_file,
            self.consistency_file, self.stability_file,
        ):
            handle.close()


def plot_lcb_scores(
    scores: LayerAblationScores,
    layer_id: int,
    unit_type: str,
    output_path: str | Path,
) -> Path:
    if not np.isfinite(scores.lcb_score).any():
        raise ValueError("LCB plot requested for a stage that did not compute LCB")
    order = np.argsort(scores.lcb_score)
    positions = np.arange(order.size)
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.errorbar(
        positions,
        scores.lcb_mean[order],
        yerr=scores.lcb_std[order],
        fmt=".",
        markersize=2,
        capsize=1,
    )
    axis.plot(positions, scores.lcb_score[order], linewidth=1.0, label="LCB")
    axis.set_title(f"Layer {layer_id} {unit_type}: contribution mean, std and LCB")
    axis.set_xlabel(f"{unit_type} units sorted by LCB")
    axis.set_ylabel("Contribution")
    axis.legend()
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
