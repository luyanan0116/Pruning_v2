from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import FrequencyConfig, GranularBallConfig
from .mi import encode_events, estimate_mi_matrix


@dataclass
class GranularBall:
    indices: np.ndarray
    center: np.ndarray
    radius: float
    purity: float
    dominant_event: int
    depth: int

    @property
    def size(self) -> int:
        return int(self.indices.size)


@dataclass
class GranularityResult:
    purity_threshold: float
    balls: List[GranularBall]
    local_band_mi: np.ndarray  # [units, bands]


def event_purity(events: np.ndarray) -> Tuple[float, int]:
    classes, counts = np.unique(events, return_counts=True)
    position = int(np.argmax(counts))
    return float(counts[position] / counts.sum()), int(classes[position])


def make_ball(features: np.ndarray, events: np.ndarray, indices: np.ndarray, depth: int) -> GranularBall:
    points = features[indices]
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    radius = float(distances.max(initial=0.0))
    purity, dominant_event = event_purity(events[indices])
    return GranularBall(
        indices=np.asarray(indices, dtype=np.int64),
        center=center,
        radius=radius,
        purity=purity,
        dominant_event=dominant_event,
        depth=int(depth),
    )


def layer_localization_features(band_energy: np.ndarray) -> np.ndarray:
    """Build one local response space per layer, not one global ball partition.

    The input is [samples, MLP channels, frequency bands]. Four robust summaries
    over channels preserve each sample's low/mid/high-frequency profile while
    keeping granular-ball construction inexpensive.
    """
    x = np.asarray(band_energy, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("band_energy must be [samples, units, bands]")
    summaries = [
        x.mean(axis=1),
        x.std(axis=1),
        np.quantile(x, 0.25, axis=1),
        np.quantile(x, 0.75, axis=1),
    ]
    features = np.concatenate(summaries, axis=1)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / (std + 1e-8)


def _two_means(points: np.ndarray, iterations: int) -> Optional[np.ndarray]:
    if points.shape[0] < 2:
        return None
    mean = points.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(points - mean, axis=1)))
    second = int(np.argmax(np.linalg.norm(points - points[first], axis=1)))
    if first == second:
        return None
    centers = np.stack([points[first], points[second]], axis=0)
    labels = np.full(points.shape[0], -1, dtype=np.int64)
    for _ in range(max(2, int(iterations))):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        if np.unique(labels).size < 2:
            return None
        centers = np.stack([points[labels == group].mean(axis=0) for group in (0, 1)], axis=0)
    return labels if np.unique(labels).size == 2 else None


def split_ball(
    ball: GranularBall,
    features: np.ndarray,
    events: np.ndarray,
    cfg: GranularBallConfig,
) -> Optional[Tuple[GranularBall, GranularBall]]:
    if ball.size < 2 * cfg.min_ball_size:
        return None
    labels = _two_means(features[ball.indices], cfg.kmeans_iterations)
    if labels is None:
        return None
    child_indices = [ball.indices[labels == group] for group in (0, 1)]
    if any(indices.size < cfg.min_ball_size for indices in child_indices):
        return None
    children = [make_ball(features, events, indices, ball.depth + 1) for indices in child_indices]

    # Local I(E;Y|g) is undefined/use-less when every child contains only one
    # event. Keeping at least two event classes prevents pure-ball MI collapse.
    if cfg.min_event_classes > 1:
        if any(np.unique(events[child.indices]).size < cfg.min_event_classes for child in children):
            return None

    weighted_purity = sum(child.size * child.purity for child in children) / ball.size
    purity_gain = weighted_purity - ball.purity
    weighted_radius = sum(child.size * child.radius for child in children) / ball.size
    radius_reduction = 1.0 - weighted_radius / max(ball.radius, 1e-12)
    if purity_gain < cfg.min_purity_gain and radius_reduction < cfg.min_radius_reduction:
        return None
    return children[0], children[1]


def build_multigranularity_hierarchy(
    features: np.ndarray,
    events: np.ndarray,
    cfg: GranularBallConfig,
) -> List[Tuple[float, List[GranularBall]]]:
    """Purity-driven nested hierarchy.

    Higher event-purity thresholds continue splitting the previous partition, so
    the number of balls is non-decreasing across coarse-to-fine granularities.
    """
    cfg.validate()
    y = encode_events(events)
    root = make_ball(features, y, np.arange(features.shape[0]), depth=0)
    root_radius = max(root.radius, 1e-12)
    balls: List[GranularBall] = [root]
    terminal: Set[Tuple[int, ...]] = set()
    snapshots: List[Tuple[float, List[GranularBall]]] = []

    for threshold in cfg.purity_thresholds:
        while len(balls) < cfg.max_balls:
            candidates = []
            for position, ball in enumerate(balls):
                key = tuple(ball.indices.tolist())
                if key in terminal:
                    continue
                purity_gap = max(0.0, threshold - ball.purity)
                radius_gap = max(0.0, ball.radius / root_radius - cfg.compactness_ratio)
                if purity_gap > 0 or radius_gap > 0:
                    # Event purity is primary, compactness breaks local-density ties.
                    priority = 2.0 * purity_gap + radius_gap
                    candidates.append((priority, ball.size, position))
            if not candidates:
                break
            _, _, position = max(candidates)
            parent = balls[position]
            children = split_ball(parent, features, y, cfg)
            if children is None:
                terminal.add(tuple(parent.indices.tolist()))
                continue
            balls[position : position + 1] = [children[0], children[1]]

        snapshots.append(
            (
                float(threshold),
                [
                    GranularBall(
                        indices=ball.indices.copy(),
                        center=ball.center.copy(),
                        radius=ball.radius,
                        purity=ball.purity,
                        dominant_event=ball.dominant_event,
                        depth=ball.depth,
                    )
                    for ball in balls
                ],
            )
        )
    return snapshots


def local_mi_for_partition(
    band_energy: np.ndarray,
    events: np.ndarray,
    balls: Sequence[GranularBall],
    mi_cfg: FrequencyConfig,
) -> np.ndarray:
    y = encode_events(events)
    n_samples, n_units, n_bands = band_energy.shape
    result = np.zeros((n_units, n_bands), dtype=np.float64)
    for ball_id, ball in enumerate(balls):
        indices = ball.indices
        classes, counts = np.unique(y[indices], return_counts=True)
        if classes.size < 2 or counts.min() < 2:
            continue
        sample_weight = indices.size / n_samples
        for band in range(n_bands):
            result[:, band] += sample_weight * estimate_mi_matrix(
                band_energy[indices, :, band],
                y[indices],
                mi_cfg,
                seed_offset=ball_id * 1000 + band,
            )
    return result


def multi_granularity_local_mi(
    band_energy: np.ndarray,
    events: np.ndarray,
    mi_cfg: FrequencyConfig,
    gb_cfg: GranularBallConfig,
) -> Tuple[np.ndarray, List[GranularityResult]]:
    features = layer_localization_features(band_energy)
    hierarchy = build_multigranularity_hierarchy(features, events, gb_cfg)
    if gb_cfg.granularity_weights is None:
        weights = np.ones(len(hierarchy), dtype=np.float64)
    else:
        weights = np.asarray(gb_cfg.granularity_weights, dtype=np.float64)
    weights /= weights.sum()

    fused = np.zeros((band_energy.shape[1], band_energy.shape[2]), dtype=np.float64)
    details: List[GranularityResult] = []
    for weight, (threshold, balls) in zip(weights, hierarchy):
        local = local_mi_for_partition(band_energy, events, balls, mi_cfg)
        fused += weight * local
        details.append(GranularityResult(threshold, balls, local))
    return fused, details
