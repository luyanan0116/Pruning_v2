from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA

from .config import GranularBallConfig
from .granular_ball import build_multigranularity_hierarchy, layer_localization_features, unit_localization_features
from .pipeline import LayerAblationScores


class LayerReportWriter:
    def __init__(self, output_dir: str | Path, band_count: int, prune_step: int = 10):
        del prune_step
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.band_count = int(band_count)
        self.score_file = (self.output_dir / "all_contribution_scores.csv").open("w", newline="", encoding="utf-8")
        self.band_file = (self.output_dir / "frequency_band_boundaries.csv").open("w", newline="", encoding="utf-8")
        self.consistency_file = (self.output_dir / "mi_estimator_consistency.csv").open("w", newline="", encoding="utf-8")
        self.ball_file = (self.output_dir / "granular_ball_summary.csv").open("w", newline="", encoding="utf-8")
        self.score_writer = csv.DictWriter(self.score_file, fieldnames=self._score_fields())
        self.band_writer = csv.DictWriter(
            self.band_file,
            fieldnames=["unit_type", "layer", "band", "start_fine_bin", "end_fine_bin", "probe_count", "probe_indices", "fine_relevance"],
        )
        self.consistency_writer = csv.DictWriter(
            self.consistency_file,
            fieldnames=["unit_type", "layer", "scope", "kendall_tau", "spearman_rho"],
        )
        self.ball_writer = csv.DictWriter(
            self.ball_file,
            fieldnames=[
                "unit_type", "layer", "granularity", "purity_threshold", "fusion_weight", "dispersion",
                "ball_id", "size", "purity", "radius", "depth", "dominant_event",
            ],
        )
        self.score_writer.writeheader()
        self.band_writer.writeheader()
        self.consistency_writer.writeheader()
        self.ball_writer.writeheader()

    def _score_fields(self):
        fields = ["unit_type", "layer", "unit", "mi_total", "granular_total", "lcb_mean", "lcb_std", "lcb_score"]
        for band in range(self.band_count):
            fields.extend([
                f"global_band_mi_{band}", f"global_band_kde_{band}",
                f"local_band_mi_{band}", f"local_band_kde_{band}",
                f"lcb_band_mean_{band}", f"lcb_band_std_{band}",
            ])
        return fields

    def add_layer(self, layer_id: int, unit_type: str, scores: LayerAblationScores, selections=None) -> None:
        del selections
        probes = np.asarray(scores.global_spectrum.probe_indices, dtype=np.int64)
        relevance = np.asarray(scores.global_spectrum.fine_relevance, dtype=np.float64)
        for band, (start, end) in enumerate(scores.global_spectrum.band_ranges):
            self.band_writer.writerow({
                "unit_type": unit_type, "layer": layer_id, "band": band,
                "start_fine_bin": start, "end_fine_bin": end,
                "probe_count": int(probes.size), "probe_indices": json.dumps(probes.tolist()),
                "fine_relevance": json.dumps(relevance[start:end].tolist()),
            })
        for scope, primary, auxiliary in (
            ("global", scores.global_spectrum.band_mi.mean(axis=1), scores.global_spectrum.band_mi_kde.mean(axis=1)),
            ("granular", scores.granular_band_mi.mean(axis=1), scores.granular_band_mi_kde.mean(axis=1)),
        ):
            tau = kendalltau(primary, auxiliary, nan_policy="omit").statistic
            rho = spearmanr(primary, auxiliary, nan_policy="omit").statistic
            self.consistency_writer.writerow({
                "unit_type": unit_type, "layer": layer_id, "scope": scope,
                "kendall_tau": float(tau) if np.isfinite(tau) else "",
                "spearman_rho": float(rho) if np.isfinite(rho) else "",
            })
        for unit in range(scores.mi_score.size):
            row = {
                "unit_type": unit_type, "layer": layer_id, "unit": unit,
                "mi_total": float(scores.mi_score[unit]),
                "granular_total": float(scores.granular_score[unit]),
                "lcb_mean": float(scores.lcb_mean[unit]),
                "lcb_std": float(scores.lcb_std[unit]),
                "lcb_score": float(scores.lcb_score[unit]),
            }
            for band in range(self.band_count):
                row[f"global_band_mi_{band}"] = float(scores.global_spectrum.band_mi[unit, band])
                row[f"global_band_kde_{band}"] = float(scores.global_spectrum.band_mi_kde[unit, band])
                row[f"local_band_mi_{band}"] = float(scores.granular_band_mi[unit, band])
                row[f"local_band_kde_{band}"] = float(scores.granular_band_mi_kde[unit, band])
                row[f"lcb_band_mean_{band}"] = float(scores.lcb_band_mean[unit, band])
                row[f"lcb_band_std_{band}"] = float(scores.lcb_band_std[unit, band])
            self.score_writer.writerow(row)
        for granularity_id, granularity in enumerate(scores.granularities):
            for ball_id, ball in enumerate(granularity.balls):
                self.ball_writer.writerow({
                    "unit_type": unit_type, "layer": layer_id, "granularity": granularity_id,
                    "purity_threshold": granularity.purity_threshold, "fusion_weight": granularity.fusion_weight,
                    "dispersion": granularity.dispersion, "ball_id": ball_id, "size": ball.size,
                    "purity": ball.purity, "radius": ball.radius, "depth": ball.depth,
                    "dominant_event": ball.dominant_event,
                })
        self.score_file.flush(); self.band_file.flush(); self.consistency_file.flush(); self.ball_file.flush()

    def close(self) -> None:
        self.score_file.close(); self.band_file.close(); self.consistency_file.close(); self.ball_file.close()


def plot_granular_balls(scores: LayerAblationScores, events: np.ndarray, layer_id: int, unit_type: str, output_path: str | Path, granularity_index: int = -1) -> Path:
    features = layer_localization_features(scores.global_spectrum.band_energy)
    projection = PCA(n_components=2, random_state=0).fit_transform(features)
    granularity = scores.granularities[granularity_index]
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=np.asarray(events), s=22, alpha=0.75)
    for ball_id, ball in enumerate(granularity.balls):
        points = projection[ball.indices]
        center = points.mean(axis=0)
        radius = float(np.quantile(np.linalg.norm(points - center, axis=1), 0.95)) if points.shape[0] > 1 else 0.0
        ax.add_patch(plt.Circle(center, radius, fill=False, linewidth=1.0, alpha=0.8))
        ax.text(center[0], center[1], f"g{ball_id}\nn={ball.size}\np={ball.purity:.2f}", fontsize=7)
    ax.set_title(f"Layer {layer_id} {unit_type}: local granular balls, purity={granularity.purity_threshold:.2f}")
    ax.set_xlabel("PCA-1"); ax.set_ylabel("PCA-2")
    fig.colorbar(scatter, ax=ax, label="task event")
    fig.tight_layout()
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)
    return path


def plot_unit_local_granular_balls(scores: LayerAblationScores, events: np.ndarray, layer_id: int, unit_type: str, unit_id: int, gb_config: GranularBallConfig, output_path: str | Path, granularity_index: int = -1) -> Path:
    values = np.asarray(scores.global_spectrum.band_energy[:, unit_id, :], dtype=np.float64)
    features = unit_localization_features(values)
    hierarchy = build_multigranularity_hierarchy(features, events, gb_config)
    threshold, balls = hierarchy[granularity_index]
    n_components = 2 if features.shape[1] >= 2 else 1
    projection = PCA(n_components=n_components, random_state=0).fit_transform(features)
    if n_components == 1:
        projection = np.column_stack([projection[:, 0], np.zeros(projection.shape[0])])
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=np.asarray(events), s=24, alpha=0.78)
    for ball_id, ball in enumerate(balls):
        points = projection[ball.indices]
        center = points.mean(axis=0)
        radius = float(np.quantile(np.linalg.norm(points - center, axis=1), 0.95)) if points.shape[0] > 1 else 0.0
        ax.add_patch(plt.Circle(center, radius, fill=False, linewidth=1.0, alpha=0.85))
        ax.text(center[0], center[1], f"g{ball_id}\nn={ball.size}\np={ball.purity:.2f}", fontsize=7)
    ax.set_title(f"Layer {layer_id} {unit_type} unit {unit_id}: local balls, purity={threshold:.2f}")
    ax.set_xlabel("PCA-1"); ax.set_ylabel("PCA-2")
    fig.colorbar(scatter, ax=ax, label="task event")
    fig.tight_layout()
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)
    return path


def plot_lcb_scores(scores: LayerAblationScores, layer_id: int, unit_type: str, output_path: str | Path) -> Path:
    order = np.argsort(scores.lcb_score)
    x = np.arange(order.size)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(x, scores.lcb_mean[order], yerr=scores.lcb_std[order], fmt=".", markersize=2, capsize=1)
    ax.plot(x, scores.lcb_score[order], linewidth=1.0, label="LCB")
    ax.set_title(f"Layer {layer_id} {unit_type}: contribution mean, std and LCB")
    ax.set_xlabel(f"{unit_type} units sorted by LCB"); ax.set_ylabel("contribution"); ax.legend()
    fig.tight_layout()
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)
    return path
