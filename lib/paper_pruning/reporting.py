from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from .granular_ball import layer_localization_features
from .pipeline import LayerAblationScores
from .selection import split_into_steps


METHOD_KEYS = ("paper_mi", "paper_mi_gb", "paper_mi_gb_lcb")


class LayerReportWriter:
    def __init__(self, output_dir: str | Path, band_count: int, prune_step: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.band_count = int(band_count)
        self.prune_step = int(prune_step)
        self.score_file = (self.output_dir / "all_contribution_scores.csv").open("w", newline="", encoding="utf-8")
        self.ball_file = (self.output_dir / "granular_ball_summary.csv").open("w", newline="", encoding="utf-8")
        self.score_writer = csv.DictWriter(self.score_file, fieldnames=self._score_fields())
        self.ball_writer = csv.DictWriter(
            self.ball_file,
            fieldnames=[
                "unit_type", "layer", "granularity", "purity_threshold", "ball_id", "size",
                "purity", "radius", "depth", "dominant_event",
            ],
        )
        self.score_writer.writeheader()
        self.ball_writer.writeheader()

    def _score_fields(self):
        fields = [
            "unit_type", "layer", "unit", "mi_total", "granular_total", "lcb_mean", "lcb_std", "lcb_score",
            "mi_pruned", "granular_pruned", "lcb_pruned",
        ]
        for band in range(self.band_count):
            fields.extend([f"global_band_mi_{band}", f"local_band_mi_{band}"])
        return fields

    def add_layer(
        self,
        layer_id: int,
        unit_type: str,
        scores: LayerAblationScores,
        selections: Mapping[str, np.ndarray],
    ) -> None:
        selected_sets = {key: set(np.asarray(value, dtype=np.int64).tolist()) for key, value in selections.items()}
        for unit in range(scores.mi_score.size):
            row = {
                "unit_type": unit_type,
                "layer": layer_id,
                "unit": unit,
                "mi_total": float(scores.mi_score[unit]),
                "granular_total": float(scores.granular_score[unit]),
                "lcb_mean": float(scores.lcb_mean[unit]),
                "lcb_std": float(scores.lcb_std[unit]),
                "lcb_score": float(scores.lcb_score[unit]),
                "mi_pruned": unit in selected_sets["paper_mi"],
                "granular_pruned": unit in selected_sets["paper_mi_gb"],
                "lcb_pruned": unit in selected_sets["paper_mi_gb_lcb"],
            }
            for band in range(self.band_count):
                row[f"global_band_mi_{band}"] = float(scores.global_spectrum.band_mi[unit, band])
                row[f"local_band_mi_{band}"] = float(scores.granular_band_mi[unit, band])
            self.score_writer.writerow(row)

        for granularity_id, granularity in enumerate(scores.granularities):
            for ball_id, ball in enumerate(granularity.balls):
                self.ball_writer.writerow(
                    {
                        "unit_type": unit_type,
                        "layer": layer_id,
                        "granularity": granularity_id,
                        "purity_threshold": granularity.purity_threshold,
                        "ball_id": ball_id,
                        "size": ball.size,
                        "purity": ball.purity,
                        "radius": ball.radius,
                        "depth": ball.depth,
                        "dominant_event": ball.dominant_event,
                    }
                )
        self.score_file.flush()
        self.ball_file.flush()

    def close(self) -> None:
        self.score_file.close()
        self.ball_file.close()


def plot_granular_balls(
    scores: LayerAblationScores,
    events: np.ndarray,
    layer_id: int,
    unit_type: str,
    output_path: str | Path,
    granularity_index: int = -1,
) -> Path:
    features = layer_localization_features(scores.global_spectrum.band_energy)
    projection = PCA(n_components=2, random_state=0).fit_transform(features)
    granularity = scores.granularities[granularity_index]

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=np.asarray(events), s=22, alpha=0.75)
    for ball_id, ball in enumerate(granularity.balls):
        points = projection[ball.indices]
        center = points.mean(axis=0)
        radius = float(np.quantile(np.linalg.norm(points - center, axis=1), 0.95)) if points.shape[0] > 1 else 0.0
        circle = plt.Circle(center, radius, fill=False, linewidth=1.0, alpha=0.8)
        ax.add_patch(circle)
        ax.text(center[0], center[1], f"g{ball_id}\nn={ball.size}\np={ball.purity:.2f}", fontsize=7)
    ax.set_title(
        f"Layer {layer_id} {unit_type}: local granular balls, "
        f"purity={granularity.purity_threshold:.2f}"
    )
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    fig.colorbar(scatter, ax=ax, label="task event")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_lcb_scores(
    scores: LayerAblationScores,
    layer_id: int,
    unit_type: str,
    output_path: str | Path,
) -> Path:
    order = np.argsort(scores.lcb_score)
    x = np.arange(order.size)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(x, scores.lcb_mean[order], yerr=scores.lcb_std[order], fmt=".", markersize=2, capsize=1)
    ax.plot(x, scores.lcb_score[order], linewidth=1.0, label="LCB")
    ax.set_title(f"Layer {layer_id} {unit_type}: contribution mean, std and LCB")
    ax.set_xlabel(f"{unit_type} units sorted by LCB")
    ax.set_ylabel("contribution")
    ax.legend()
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _flatten_mask(mask: Mapping[str, Mapping[int, np.ndarray]]):
    return {
        (unit_type, int(layer), int(unit))
        for unit_type, layer_map in mask.items()
        for layer, values in layer_map.items()
        for unit in np.asarray(values)
    }


def write_selection_files(
    output_dir: str | Path,
    selections: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    prune_step: int,
) -> None:
    output = Path(output_dir)
    serializable = {
        method: {
            unit_type: {
                str(layer): np.asarray(indices, dtype=np.int64).tolist()
                for layer, indices in layer_map.items()
            }
            for unit_type, layer_map in target_map.items()
        }
        for method, target_map in selections.items()
    }
    (output / "prune_indices.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    batches = {
        method: {
            unit_type: {
                str(layer): [batch.tolist() for batch in split_into_steps(indices, prune_step)]
                for layer, indices in layer_map.items()
            }
            for unit_type, layer_map in target_map.items()
        }
        for method, target_map in selections.items()
    }
    (output / "prune_batches_step10.json").write_text(
        json.dumps(batches, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = []
    method_names = list(selections)
    for left_pos, left in enumerate(method_names):
        for right in method_names[left_pos + 1 :]:
            a, b = _flatten_mask(selections[left]), _flatten_mask(selections[right])
            union = a | b
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "jaccard": 1.0 if not union else len(a & b) / len(union),
                    "symmetric_difference_count": len(a ^ b),
                }
            )
    with (output / "mask_overlap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["left", "right", "jaccard", "symmetric_difference_count"])
        writer.writeheader()
        writer.writerows(rows)
