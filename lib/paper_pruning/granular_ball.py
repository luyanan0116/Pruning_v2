from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import FrequencyConfig, GranularBallConfig
from .mi import encode_events, estimate_kde_mi_matrix, estimate_knn_mi_matrix


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
    """Layer-level diagnostics for one purity-controlled granularity.

    The actual proposal flow builds a separate hierarchy in every structural
    unit's own B-dimensional response space. Retaining every ball for every
    unit would be prohibitively large, so reports store exact per-level MI
    arrays together with aggregate geometry statistics. These diagnostics do
    not alter any score or selection.
    """

    purity_threshold: float
    local_band_mi: np.ndarray       # [units,bands]
    local_band_mi_kde: np.ndarray   # [units,bands]
    fusion_weight: float = 0.0
    variance: float = 0.0
    mean_ball_count: float = 0.0
    mean_ball_size: float = 0.0
    mean_purity: float = 0.0
    mean_radius: float = 0.0
    mean_max_depth: float = 0.0


@dataclass(frozen=True)
class _PartitionStats:
    ball_count: float
    mean_ball_size: float
    mean_purity: float
    mean_radius: float
    max_depth: float


def event_purity(events: np.ndarray) -> Tuple[float, int]:
    y = np.asarray(events, dtype=np.int64)
    if y.size == 0:
        return 0.0, -1
    counts = np.bincount(y)
    dominant = int(np.argmax(counts))
    return float(counts[dominant] / y.size), dominant


def make_ball(
    features: np.ndarray,
    events: np.ndarray,
    indices: np.ndarray,
    depth: int,
) -> GranularBall:
    points = np.asarray(features[indices], dtype=np.float64)
    center = points.mean(axis=0)
    delta = points - center
    distance_sq = np.einsum("ij,ij->i", delta, delta, optimize=True)
    radius = float(np.sqrt(distance_sq.max(initial=0.0)))
    purity, dominant = event_purity(events[indices])
    return GranularBall(
        indices=np.asarray(indices, dtype=np.int64),
        center=center,
        radius=radius,
        purity=purity,
        dominant_event=dominant,
        depth=int(depth),
    )


def unit_localization_features(unit_band_energy: np.ndarray) -> np.ndarray:
    """Equation (16)/(6): use the literal band-response vector v_u(x).

    No extra cross-sample z-score is applied here. The only standardization in
    the proposal is the sample-internal sequence standardization performed
    before DCT. Keeping the raw band-response coordinates makes the particle
    center and radius exactly match the displayed definitions.
    """
    values = np.asarray(unit_band_energy, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("unit_band_energy must be [samples,bands]")
    return values


def _squared_distances_to_centers(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - centers[None, :, :]
    return np.einsum("nkd,nkd->nk", delta, delta, optimize=True)


def _two_means(points: np.ndarray, iterations: int) -> Optional[np.ndarray]:
    """Deterministic binary k-means used to implement each geometric split."""
    if points.shape[0] < 2:
        return None
    center = points.mean(axis=0)
    delta = points - center
    first = int(np.argmax(np.einsum("ij,ij->i", delta, delta, optimize=True)))
    delta = points - points[first]
    second = int(np.argmax(np.einsum("ij,ij->i", delta, delta, optimize=True)))
    if first == second:
        return None
    centers = np.stack([points[first], points[second]], axis=0)
    labels = np.full(points.shape[0], -1, dtype=np.int64)
    for _ in range(max(2, int(iterations))):
        updated = np.argmin(_squared_distances_to_centers(points, centers), axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        if np.unique(labels).size < 2:
            return None
        centers = np.stack(
            [points[labels == group].mean(axis=0) for group in (0, 1)],
            axis=0,
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
    """Equations (17)-(19): nested purity/geometry-controlled partitions."""
    cfg.validate()
    x = unit_localization_features(features)
    y = encode_events(events)
    root = make_ball(x, y, np.arange(x.shape[0]), depth=0)
    root_radius = max(root.radius, 1e-12)
    balls: List[GranularBall] = [root]
    terminal: set[bytes] = set()
    snapshots: List[Tuple[float, List[GranularBall]]] = []

    def key(item: GranularBall) -> bytes:
        return np.ascontiguousarray(item.indices, dtype=np.int64).tobytes()

    for threshold in cfg.purity_thresholds:
        while len(balls) < cfg.max_balls:
            candidates = []
            for position, ball in enumerate(balls):
                if key(ball) in terminal or ball.depth >= cfg.max_depth:
                    continue
                purity_gap = max(0.0, float(threshold) - ball.purity)
                radius_gap = max(0.0, ball.radius / root_radius - cfg.compactness_ratio)
                if purity_gap > 0 or radius_gap > 0:
                    candidates.append((2.0 * purity_gap + radius_gap, ball.size, -ball.depth, position))
            if not candidates:
                break
            *_, position = max(candidates)
            parent = balls[position]
            children = split_ball(parent, x, y, cfg)
            if children is None:
                terminal.add(key(parent))
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
    # Detailed KDE text: local bandwidth is proportional to particle radius.
    return max(1e-4, ball.radius / max(1.0, np.sqrt(float(feature_dim))))


def _partition_stats(balls: Sequence[GranularBall]) -> _PartitionStats:
    if not balls:
        return _PartitionStats(0.0, 0.0, 0.0, 0.0, 0.0)
    sizes = np.asarray([ball.size for ball in balls], dtype=np.float64)
    weights = sizes / max(float(sizes.sum()), 1.0)
    return _PartitionStats(
        ball_count=float(len(balls)),
        mean_ball_size=float(sizes.mean()),
        mean_purity=float(np.dot(weights, [ball.purity for ball in balls])),
        mean_radius=float(np.dot(weights, [ball.radius for ball in balls])),
        max_depth=float(max(ball.depth for ball in balls)),
    )


def _local_mi_for_partition(
    unit_values: np.ndarray,
    events: np.ndarray,
    balls: Sequence[GranularBall],
    mi_cfg: FrequencyConfig,
    compute_kde: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Equations (20)-(22)/(7)-(8): local MI and sample-share aggregation."""
    values = np.asarray(unit_values, dtype=np.float64)
    y = encode_events(events)
    n_samples, n_bands = values.shape
    knn = np.zeros(n_bands, dtype=np.float64)
    kde = np.zeros(n_bands, dtype=np.float64)
    for ball in balls:
        indices = ball.indices
        classes, counts = np.unique(y[indices], return_counts=True)
        if classes.size < 2 or counts.min() < 2:
            # Statistically unsupported local MI contributes zero with its
            # original sample share, equivalent to the displayed weighted sum.
            continue
        weight = float(indices.size / n_samples)
        knn += weight * estimate_knn_mi_matrix(values[indices], y[indices], mi_cfg)
        if compute_kde:
            bandwidth = _ball_bandwidth(ball, n_bands)
            kde += weight * estimate_kde_mi_matrix(
                values[indices],
                y[indices],
                mi_cfg,
                bandwidth=bandwidth,
            )
    return knn, kde


def _single_unit_levels(
    unit_values: np.ndarray,
    events: np.ndarray,
    mi_cfg: FrequencyConfig,
    gb_cfg: GranularBallConfig,
    compute_kde: bool,
) -> tuple[np.ndarray, np.ndarray, list[_PartitionStats]]:
    features = unit_localization_features(unit_values)
    hierarchy = build_multigranularity_hierarchy(features, events, gb_cfg)
    level_knn = []
    level_kde = []
    level_stats = []
    for _threshold, balls in hierarchy:
        knn, kde = _local_mi_for_partition(
            unit_values,
            events,
            balls,
            mi_cfg,
            compute_kde=compute_kde,
        )
        level_knn.append(knn)
        level_kde.append(kde if compute_kde else np.full_like(knn, np.nan))
        level_stats.append(_partition_stats(balls))
    return np.stack(level_knn), np.stack(level_kde), level_stats


def multi_granularity_level_mi(
    band_energy: np.ndarray,
    events: np.ndarray,
    mi_cfg: FrequencyConfig,
    gb_cfg: GranularBallConfig,
    compute_kde: bool = True,
    kde_unit_indices: Optional[np.ndarray] = None,
    build_details: bool = True,
) -> tuple[np.ndarray, np.ndarray, List[GranularityResult]]:
    """Return every granularity before fusion.

    Output shapes are ``[granularities, units, bands]``. Each unit constructs
    its own particle hierarchy from its literal frequency-band response vector,
    exactly matching the two attached workflows.
    """
    gb_cfg.validate()
    values = np.asarray(band_energy, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("band_energy must be [samples,units,bands]")
    if values.shape[0] != np.asarray(events).size:
        raise ValueError("band_energy and events must have the same sample count")

    level_count = len(gb_cfg.purity_thresholds)
    unit_count = values.shape[1]
    band_count = values.shape[2]
    level_knn = np.empty((level_count, unit_count, band_count), dtype=np.float64)
    level_kde = np.full((level_count, unit_count, band_count), np.nan, dtype=np.float64)

    if not compute_kde or gb_cfg.kde_scope == "none":
        kde_set: set[int] = set()
    elif gb_cfg.kde_scope == "all" or kde_unit_indices is None:
        kde_set = set(range(unit_count))
    else:
        kde_set = set(np.asarray(kde_unit_indices, dtype=np.int64).tolist())

    # [level, unit, statistic]
    statistics = np.zeros((level_count, unit_count, 5), dtype=np.float64)
    chunk_size = max(1, int(gb_cfg.worker_chunk_size))

    def work_chunk(start: int, end: int):
        local_knn = np.empty((level_count, end - start, band_count), dtype=np.float64)
        local_kde = np.full((level_count, end - start, band_count), np.nan, dtype=np.float64)
        local_stats = np.zeros((level_count, end - start, 5), dtype=np.float64)
        for offset, unit_id in enumerate(range(start, end)):
            knn, kde, stats = _single_unit_levels(
                values[:, unit_id, :],
                events,
                mi_cfg,
                gb_cfg,
                compute_kde=unit_id in kde_set,
            )
            local_knn[:, offset, :] = knn
            if unit_id in kde_set:
                local_kde[:, offset, :] = kde
            for level, item in enumerate(stats):
                local_stats[level, offset] = (
                    item.ball_count,
                    item.mean_ball_size,
                    item.mean_purity,
                    item.mean_radius,
                    item.max_depth,
                )
        return start, end, local_knn, local_kde, local_stats

    chunks = [
        (start, min(start + chunk_size, unit_count))
        for start in range(0, unit_count, chunk_size)
    ]
    completed = 0

    def store(result) -> None:
        nonlocal completed
        start, end, knn, kde, stats = result
        level_knn[:, start:end, :] = knn
        level_kde[:, start:end, :] = kde
        statistics[:, start:end, :] = stats
        completed += end - start
        if completed == unit_count or completed // 512 != (completed - (end - start)) // 512:
            print(f"[granular-ball unit-local] {completed}/{unit_count} units", flush=True)

    if gb_cfg.workers == 1 or len(chunks) == 1:
        for start, end in chunks:
            store(work_chunk(start, end))
    else:
        with ThreadPoolExecutor(max_workers=gb_cfg.workers) as executor:
            futures = [executor.submit(work_chunk, start, end) for start, end in chunks]
            for future in as_completed(futures):
                store(future.result())

    if not build_details:
        return level_knn, level_kde, []

    details: List[GranularityResult] = []
    for level, threshold in enumerate(gb_cfg.purity_thresholds):
        mean_stats = statistics[level].mean(axis=0)
        details.append(
            GranularityResult(
                purity_threshold=float(threshold),
                local_band_mi=level_knn[level],
                local_band_mi_kde=level_kde[level],
                mean_ball_count=float(mean_stats[0]),
                mean_ball_size=float(mean_stats[1]),
                mean_purity=float(mean_stats[2]),
                mean_radius=float(mean_stats[3]),
                mean_max_depth=float(mean_stats[4]),
            )
        )
    return level_knn, level_kde, details



def configured_granularity_weights(gb_cfg: GranularBallConfig) -> np.ndarray:
    """Return deterministic multi-granularity fusion weights.

    ``equal`` is the strict component-ablation default: adding LCB changes only
    repeated estimation and the lower confidence bound. ``manual`` uses the
    user supplied weights. ``repeat_variance`` is resolved after repeated
    estimates and therefore cannot be produced by this helper.
    """
    gb_cfg.validate()
    count = len(gb_cfg.purity_thresholds)
    if gb_cfg.granularity_weight_mode == "repeat_variance":
        raise ValueError("repeat_variance weights require repeated estimates")
    if gb_cfg.granularity_weights is None:
        weights = np.ones(count, dtype=np.float64)
    else:
        weights = np.asarray(gb_cfg.granularity_weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0:
        raise ValueError("granularity weights must have a positive sum")
    return weights / weights.sum()

def granularity_weights_from_repeat_variance(
    repeat_level_band_mi: np.ndarray,
    gb_cfg: GranularBallConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Equation (23)/(8): adaptive fusion from repeated-estimate variance.

    ``repeat_level_band_mi`` has shape ``[repeat, granularity, unit, band]``.
    The displayed proposal permits variance or ranking stability. This strict
    implementation uses the direct variance option: average the empirical
    repeat variance over all unit-band entries, then assign inverse-variance
    weights. Explicit user-supplied weights override this estimate.
    """
    values = np.asarray(repeat_level_band_mi, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("repeat_level_band_mi must be [repeat,granularity,unit,band]")
    ddof = 1 if values.shape[0] > 1 else 0
    variances = np.nanmean(np.nanvar(values, axis=0, ddof=ddof), axis=(1, 2))
    variances = np.where(np.isfinite(variances), np.maximum(variances, 0.0), np.inf)
    if gb_cfg.granularity_weights is not None:
        weights = np.asarray(gb_cfg.granularity_weights, dtype=np.float64)
    else:
        weights = 1.0 / (variances + gb_cfg.variance_floor)
    weights = np.where(np.isfinite(weights), np.maximum(weights, 0.0), 0.0)
    if weights.sum() <= 0:
        weights = np.ones(values.shape[1], dtype=np.float64)
    weights /= weights.sum()
    return weights, variances


def fuse_granularity_levels(level_values: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    values = np.asarray(level_values, dtype=np.float64)
    fusion = np.asarray(tuple(float(value) for value in weights), dtype=np.float64)
    if values.ndim < 2 or fusion.shape != (values.shape[0],):
        raise ValueError("one fusion weight is required for every granularity")
    return np.tensordot(fusion, values, axes=(0, 0))
