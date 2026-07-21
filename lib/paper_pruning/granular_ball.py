from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import FrequencyConfig, GranularBallConfig
from .mi import (
    encode_events,
    estimate_kde_mi_matrix,
    estimate_knn_mi_matrix,
    normalized_band_weights,
)


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
    local_band_mi: np.ndarray       # primary kNN [units, bands]
    local_band_mi_kde: np.ndarray   # auxiliary KDE [units, bands]
    fusion_weight: float = 0.0
    dispersion: float = 0.0


def event_purity(events: np.ndarray) -> Tuple[float, int]:
    classes, counts = np.unique(events, return_counts=True)
    position = int(np.argmax(counts))
    return float(counts[position] / counts.sum()), int(classes[position])


def make_ball(
    features: np.ndarray, events: np.ndarray, indices: np.ndarray, depth: int
) -> GranularBall:
    points = features[indices]
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    radius = float(distances.max(initial=0.0))
    purity, dominant_event = event_purity(events[indices])
    return GranularBall(
        indices=np.asarray(indices, dtype=np.int64),
        center=np.asarray(center, dtype=np.float64),
        radius=radius,
        purity=purity,
        dominant_event=dominant_event,
        depth=int(depth),
    )


def layer_localization_features(band_energy: np.ndarray) -> np.ndarray:
    """Practical layer-local response space used for diagnostics/shared mode.

    ``unit_local`` mode follows the detailed proposal literally by using each
    unit's own B-dimensional response vector. ``layer_shared`` mode retains a
    much faster layer-local partition and is useful for debugging.
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


def unit_localization_features(unit_band_energy: np.ndarray) -> np.ndarray:
    x = np.asarray(unit_band_energy, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("unit_band_energy must be [samples, bands]")
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + 1e-8)


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
        centers = np.stack(
            [points[labels == group].mean(axis=0) for group in (0, 1)], axis=0
        )
    return labels if np.unique(labels).size == 2 else None


def split_ball(
    ball: GranularBall,
    features: np.ndarray,
    events: np.ndarray,
    cfg: GranularBallConfig,
) -> Optional[Tuple[GranularBall, GranularBall]]:
    if ball.depth >= cfg.max_depth or ball.size < 2 * cfg.min_ball_size:
        return None
    labels = _two_means(features[ball.indices], cfg.kmeans_iterations)
    if labels is None:
        return None
    child_indices = [ball.indices[labels == group] for group in (0, 1)]
    if any(indices.size < cfg.min_ball_size for indices in child_indices):
        return None
    children = [
        make_ball(features, events, indices, ball.depth + 1)
        for indices in child_indices
    ]
    if cfg.min_event_classes > 1 and any(
        np.unique(events[child.indices]).size < cfg.min_event_classes
        for child in children
    ):
        return None

    weighted_purity = sum(child.size * child.purity for child in children) / ball.size
    purity_gain = weighted_purity - ball.purity
    weighted_radius = sum(child.size * child.radius for child in children) / ball.size
    radius_reduction = 1.0 - weighted_radius / max(ball.radius, 1e-12)
    # The proposal uses geometry and event purity as a joint quality rule.
    # Do not accept a split that improves one dimension by degrading the other.
    if purity_gain < cfg.min_purity_gain:
        return None
    if radius_reduction < cfg.min_radius_reduction:
        return None
    return children[0], children[1]


def build_multigranularity_hierarchy(
    features: np.ndarray,
    events: np.ndarray,
    cfg: GranularBallConfig,
) -> List[Tuple[float, List[GranularBall]]]:
    """Nested coarse-to-fine partitions driven by event purity and compactness."""
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
                if key in terminal or ball.depth >= cfg.max_depth:
                    continue
                purity_gap = max(0.0, threshold - ball.purity)
                radius_gap = max(
                    0.0, ball.radius / root_radius - cfg.compactness_ratio
                )
                if purity_gap > 0 or radius_gap > 0:
                    priority = 2.0 * purity_gap + radius_gap
                    candidates.append((priority, ball.size, -ball.depth, position))
            if not candidates:
                break
            *_, position = max(candidates)
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


def _ball_bandwidth(ball: GranularBall, feature_dim: int) -> float:
    return max(1e-4, ball.radius / max(1.0, np.sqrt(float(feature_dim))))


def local_mi_for_partition(
    band_energy: np.ndarray,
    events: np.ndarray,
    balls: Sequence[GranularBall],
    mi_cfg: FrequencyConfig,
    compute_kde: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Equations (7)/(20)-(22): local MI and sample-share aggregation."""
    y = encode_events(events)
    n_samples, n_units, n_bands = band_energy.shape
    knn_result = np.zeros((n_units, n_bands), dtype=np.float64)
    kde_result = np.zeros((n_units, n_bands), dtype=np.float64)
    ball_totals = []
    ball_weights = []

    for ball in balls:
        indices = ball.indices
        classes, counts = np.unique(y[indices], return_counts=True)
        if classes.size < 2 or counts.min() < 2:
            continue
        sample_weight = indices.size / n_samples
        local_knn = np.zeros((n_units, n_bands), dtype=np.float64)
        local_kde = np.zeros((n_units, n_bands), dtype=np.float64)
        bandwidth = _ball_bandwidth(ball, band_energy.shape[-1])
        for band in range(n_bands):
            values = band_energy[indices, :, band]
            local_knn[:, band] = estimate_knn_mi_matrix(values, y[indices], mi_cfg)
            if compute_kde:
                local_kde[:, band] = estimate_kde_mi_matrix(
                    values, y[indices], mi_cfg, bandwidth=bandwidth
                )
        knn_result += sample_weight * local_knn
        kde_result += sample_weight * local_kde
        ball_totals.append(local_knn @ normalized_band_weights(n_bands, mi_cfg))
        ball_weights.append(sample_weight)

    if len(ball_totals) <= 1:
        dispersion = 0.0
    else:
        stacked = np.stack(ball_totals, axis=0)
        weights = np.asarray(ball_weights, dtype=np.float64)
        weights /= weights.sum()
        center = np.average(stacked, axis=0, weights=weights)
        dispersion = float(
            np.average(np.square(stacked - center[None, :]), axis=0, weights=weights).mean()
        )
    return knn_result, kde_result, dispersion


def _fusion_weights(
    dispersions: Sequence[float], gb_cfg: GranularBallConfig
) -> np.ndarray:
    if gb_cfg.granularity_weights is not None:
        weights = np.asarray(gb_cfg.granularity_weights, dtype=np.float64)
    else:
        # Detailed proposal: weights are adaptively determined by estimation
        # variance or ranking stability. Inverse dispersion is the direct
        # variance-based realization.
        weights = 1.0 / (
            np.asarray(dispersions, dtype=np.float64) + gb_cfg.variance_floor
        )
        # If every dispersion is numerically zero, use equal weights.
        if not np.isfinite(weights).all() or weights.max() / max(weights.min(), 1e-30) > 1e12:
            finite = np.asarray(dispersions, dtype=np.float64)
            if np.all(finite <= gb_cfg.variance_floor):
                weights = np.ones_like(finite)
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    return weights / weights.sum()


def _shared_partition_mi(
    band_energy: np.ndarray,
    events: np.ndarray,
    mi_cfg: FrequencyConfig,
    gb_cfg: GranularBallConfig,
    compute_kde: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[GranularityResult]]:
    features = layer_localization_features(band_energy)
    hierarchy = build_multigranularity_hierarchy(features, events, gb_cfg)
    raw = []
    for threshold, balls in hierarchy:
        knn, kde, dispersion = local_mi_for_partition(
            band_energy, events, balls, mi_cfg, compute_kde=compute_kde
        )
        raw.append((threshold, balls, knn, kde, dispersion))
    weights = _fusion_weights([item[4] for item in raw], gb_cfg)
    fused_knn = np.zeros_like(raw[0][2])
    fused_kde = np.zeros_like(raw[0][3])
    details: List[GranularityResult] = []
    for weight, (threshold, balls, knn, kde, dispersion) in zip(weights, raw):
        fused_knn += weight * knn
        fused_kde += weight * kde
        details.append(
            GranularityResult(
                threshold, balls, knn, kde,
                fusion_weight=float(weight),
                dispersion=float(dispersion),
            )
        )
    return fused_knn, fused_kde, details


def _single_unit_local_mi(
    unit_values: np.ndarray,
    events: np.ndarray,
    mi_cfg: FrequencyConfig,
    gb_cfg: GranularBallConfig,
    compute_kde: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = unit_localization_features(unit_values)
    hierarchy = build_multigranularity_hierarchy(features, events, gb_cfg)
    per_level_knn = []
    per_level_kde = []
    dispersions = []
    for _, balls in hierarchy:
        knn, kde, dispersion = local_mi_for_partition(
            unit_values[:, None, :], events, balls, mi_cfg, compute_kde=compute_kde
        )
        per_level_knn.append(knn[0])
        per_level_kde.append(kde[0])
        dispersions.append(dispersion)
    weights = _fusion_weights(dispersions, gb_cfg)
    fused_knn = np.tensordot(weights, np.stack(per_level_knn), axes=(0, 0))
    fused_kde = np.tensordot(weights, np.stack(per_level_kde), axes=(0, 0))
    return fused_knn, fused_kde, weights, np.asarray(dispersions)


def _unit_local_mi(
    band_energy: np.ndarray,
    events: np.ndarray,
    mi_cfg: FrequencyConfig,
    gb_cfg: GranularBallConfig,
    compute_kde: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[GranularityResult]]:
    unit_count = band_energy.shape[1]

    def work(unit_id: int):
        return _single_unit_local_mi(
            band_energy[:, unit_id, :], events, mi_cfg, gb_cfg, compute_kde=compute_kde
        )

    results = []
    if gb_cfg.workers == 1:
        iterator = map(work, range(unit_count))
        for completed, result in enumerate(iterator, start=1):
            results.append(result)
            if completed == unit_count or completed % 512 == 0:
                print(f"[granular-ball unit-local] {completed}/{unit_count} units", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=gb_cfg.workers) as executor:
            for completed, result in enumerate(executor.map(work, range(unit_count)), start=1):
                results.append(result)
                if completed == unit_count or completed % 512 == 0:
                    print(f"[granular-ball unit-local] {completed}/{unit_count} units", flush=True)

    fused_knn = np.stack([result[0] for result in results], axis=0)
    fused_kde = np.stack([result[1] for result in results], axis=0)

    # For figures/reports, construct one layer-level diagnostic hierarchy. The
    # actual scores above still use one partition per structural unit.
    diag_features = layer_localization_features(band_energy)
    diag_hierarchy = build_multigranularity_hierarchy(
        diag_features, events, gb_cfg
    )
    mean_weights = np.mean(np.stack([result[2] for result in results]), axis=0)
    mean_dispersion = np.mean(np.stack([result[3] for result in results]), axis=0)
    details = []
    for level, (threshold, balls) in enumerate(diag_hierarchy):
        details.append(
            GranularityResult(
                threshold,
                balls,
                np.zeros_like(fused_knn),
                np.zeros_like(fused_kde),
                fusion_weight=float(mean_weights[level]),
                dispersion=float(mean_dispersion[level]),
            )
        )
    return fused_knn, fused_kde, details


def multi_granularity_local_mi(
    band_energy: np.ndarray,
    events: np.ndarray,
    mi_cfg: FrequencyConfig,
    gb_cfg: GranularBallConfig,
    compute_kde: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[GranularityResult]]:
    """Equation (8)/(23): local MI followed by adaptive multi-level fusion."""
    if gb_cfg.localization_mode == "unit_local":
        return _unit_local_mi(
            band_energy, events, mi_cfg, gb_cfg, compute_kde=compute_kde
        )
    return _shared_partition_mi(
        band_energy, events, mi_cfg, gb_cfg, compute_kde=compute_kde
    )
