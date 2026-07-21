from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
    """Event purity with a bincount fast path for already encoded labels."""
    y = np.asarray(events, dtype=np.int64)
    if y.size == 0:
        return 0.0, -1
    counts = np.bincount(y)
    position = int(np.argmax(counts))
    return float(counts[position] / y.size), position


def make_ball(
    features: np.ndarray, events: np.ndarray, indices: np.ndarray, depth: int
) -> GranularBall:
    points = features[indices]
    center = points.mean(axis=0)
    delta = points - center
    distances_sq = np.einsum("ij,ij->i", delta, delta, optimize=True)
    radius = float(np.sqrt(distances_sq.max(initial=0.0)))
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
    """Practical layer-local response space used for diagnostics/shared mode."""
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


def all_unit_localization_features(band_energy: np.ndarray) -> np.ndarray:
    """Vectorized per-unit standardization, exactly equivalent to calling
    :func:`unit_localization_features` once for every structural unit.
    """
    x = np.asarray(band_energy, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + 1e-8)


def _squared_distances_to_centers(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # Squared Euclidean distance gives the same assignments as Euclidean
    # distance but avoids a costly square root in every k-means iteration.
    delta = points[:, None, :] - centers[None, :, :]
    return np.einsum("nkd,nkd->nk", delta, delta, optimize=True)


def _two_means(points: np.ndarray, iterations: int) -> Optional[np.ndarray]:
    if points.shape[0] < 2:
        return None
    mean = points.mean(axis=0)
    delta = points - mean
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
    if purity_gain < cfg.min_purity_gain:
        return None
    if radius_reduction < cfg.min_radius_reduction:
        return None
    return children[0], children[1]


def _ball_key(ball: GranularBall) -> bytes:
    """Compact immutable key used to reuse a ball across nested granularities."""
    return np.ascontiguousarray(ball.indices, dtype=np.int64).tobytes()


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
    terminal: Set[bytes] = set()
    snapshots: List[Tuple[float, List[GranularBall]]] = []

    for threshold in cfg.purity_thresholds:
        while len(balls) < cfg.max_balls:
            candidates = []
            for position, ball in enumerate(balls):
                key = _ball_key(ball)
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
                terminal.add(_ball_key(parent))
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


@dataclass
class _BallMI:
    knn: np.ndarray
    kde: np.ndarray
    total: np.ndarray
    sample_weight: float
    valid: bool


def _compute_ball_mi(
    band_energy: np.ndarray,
    encoded_events: np.ndarray,
    ball: GranularBall,
    mi_cfg: FrequencyConfig,
    compute_kde: bool,
    kde_unit_indices: Optional[np.ndarray],
) -> _BallMI:
    """Compute all unit-band MI values in one vectorized estimator call.

    The v5 implementation invoked the estimator separately for every band.
    Flattening ``[units, bands]`` to features is mathematically identical
    because both estimators process each feature independently, while removing
    millions of Python-level calls in strict unit-local mode.
    """
    indices = ball.indices
    n_samples, n_units, n_bands = band_energy.shape
    classes, counts = np.unique(encoded_events[indices], return_counts=True)
    zeros = np.zeros((n_units, n_bands), dtype=np.float64)
    if classes.size < 2 or counts.min() < 2:
        return _BallMI(zeros, zeros.copy(), np.zeros(n_units), 0.0, False)

    values = np.asarray(band_energy[indices], dtype=np.float64)
    flat = values.reshape(indices.size, n_units * n_bands)
    knn = estimate_knn_mi_matrix(flat, encoded_events[indices], mi_cfg).reshape(
        n_units, n_bands
    )
    kde = zeros.copy()
    if compute_kde:
        if kde_unit_indices is None:
            kde = estimate_kde_mi_matrix(
                flat,
                encoded_events[indices],
                mi_cfg,
                bandwidth=_ball_bandwidth(ball, n_bands),
            ).reshape(n_units, n_bands)
        elif kde_unit_indices.size:
            selected = values[:, kde_unit_indices, :].reshape(
                indices.size, kde_unit_indices.size * n_bands
            )
            selected_kde = estimate_kde_mi_matrix(
                selected,
                encoded_events[indices],
                mi_cfg,
                bandwidth=_ball_bandwidth(ball, n_bands),
            ).reshape(kde_unit_indices.size, n_bands)
            kde[kde_unit_indices] = selected_kde
    total = knn @ normalized_band_weights(n_bands, mi_cfg)
    return _BallMI(
        knn=knn,
        kde=kde,
        total=total,
        sample_weight=float(indices.size / n_samples),
        valid=True,
    )


def _aggregate_partition_from_cache(
    balls: Sequence[GranularBall],
    cache: Dict[bytes, _BallMI],
    n_units: int,
    n_bands: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    knn_result = np.zeros((n_units, n_bands), dtype=np.float64)
    kde_result = np.zeros((n_units, n_bands), dtype=np.float64)
    totals = []
    weights = []
    for ball in balls:
        item = cache[_ball_key(ball)]
        if not item.valid:
            continue
        knn_result += item.sample_weight * item.knn
        kde_result += item.sample_weight * item.kde
        totals.append(item.total)
        weights.append(item.sample_weight)

    if len(totals) <= 1:
        dispersion = 0.0
    else:
        stacked = np.stack(totals, axis=0)
        normalized = np.asarray(weights, dtype=np.float64)
        normalized /= normalized.sum()
        center = np.average(stacked, axis=0, weights=normalized)
        dispersion = float(
            np.average(
                np.square(stacked - center[None, :]),
                axis=0,
                weights=normalized,
            ).mean()
        )
    return knn_result, kde_result, dispersion


def _evaluate_hierarchy(
    band_energy: np.ndarray,
    events: np.ndarray,
    hierarchy: Sequence[Tuple[float, Sequence[GranularBall]]],
    mi_cfg: FrequencyConfig,
    compute_kde: bool,
    kde_unit_indices: Optional[np.ndarray] = None,
) -> List[Tuple[float, List[GranularBall], np.ndarray, np.ndarray, float]]:
    """Evaluate each unique ball once and reuse it across nested snapshots."""
    y = encode_events(events)
    n_units, n_bands = band_energy.shape[1:]
    cache: Dict[bytes, _BallMI] = {}
    for _threshold, balls in hierarchy:
        for ball in balls:
            key = _ball_key(ball)
            if key not in cache:
                cache[key] = _compute_ball_mi(
                    band_energy,
                    y,
                    ball,
                    mi_cfg,
                    compute_kde=compute_kde,
                    kde_unit_indices=kde_unit_indices,
                )

    result = []
    for threshold, balls in hierarchy:
        knn, kde, dispersion = _aggregate_partition_from_cache(
            balls, cache, n_units, n_bands
        )
        result.append((float(threshold), list(balls), knn, kde, dispersion))
    return result


def local_mi_for_partition(
    band_energy: np.ndarray,
    events: np.ndarray,
    balls: Sequence[GranularBall],
    mi_cfg: FrequencyConfig,
    compute_kde: bool = True,
    kde_unit_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Equations (7)/(20)-(22): local MI and sample-share aggregation."""
    raw = _evaluate_hierarchy(
        band_energy,
        events,
        [(0.0, list(balls))],
        mi_cfg,
        compute_kde=compute_kde,
        kde_unit_indices=kde_unit_indices,
    )[0]
    return raw[2], raw[3], raw[4]


def _fusion_weights(
    dispersions: Sequence[float], gb_cfg: GranularBallConfig
) -> np.ndarray:
    if gb_cfg.granularity_weights is not None:
        weights = np.asarray(gb_cfg.granularity_weights, dtype=np.float64)
    else:
        weights = 1.0 / (
            np.asarray(dispersions, dtype=np.float64) + gb_cfg.variance_floor
        )
        if (
            not np.isfinite(weights).all()
            or weights.max() / max(weights.min(), 1e-30) > 1e12
        ):
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
    kde_unit_indices: Optional[np.ndarray] = None,
    build_details: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[GranularityResult]]:
    features = layer_localization_features(band_energy)
    hierarchy = build_multigranularity_hierarchy(features, events, gb_cfg)
    raw = _evaluate_hierarchy(
        band_energy,
        events,
        hierarchy,
        mi_cfg,
        compute_kde=compute_kde,
        kde_unit_indices=kde_unit_indices,
    )
    weights = _fusion_weights([item[4] for item in raw], gb_cfg)
    fused_knn = np.zeros_like(raw[0][2])
    fused_kde = np.zeros_like(raw[0][3])
    details: List[GranularityResult] = []
    for weight, (threshold, balls, knn, kde, dispersion) in zip(weights, raw):
        fused_knn += weight * knn
        fused_kde += weight * kde
        if build_details:
            details.append(
                GranularityResult(
                    threshold,
                    balls,
                    knn,
                    kde,
                    fusion_weight=float(weight),
                    dispersion=float(dispersion),
                )
            )
    if compute_kde and kde_unit_indices is not None:
        missing = np.ones(fused_kde.shape[0], dtype=bool)
        missing[kde_unit_indices] = False
        fused_kde[missing] = np.nan
    return fused_knn, fused_kde, details


def _single_unit_local_mi(
    unit_values: np.ndarray,
    standardized_features: np.ndarray,
    events: np.ndarray,
    mi_cfg: FrequencyConfig,
    gb_cfg: GranularBallConfig,
    compute_kde: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hierarchy = build_multigranularity_hierarchy(
        standardized_features, events, gb_cfg
    )
    raw = _evaluate_hierarchy(
        unit_values[:, None, :],
        events,
        hierarchy,
        mi_cfg,
        compute_kde=compute_kde,
        kde_unit_indices=None,
    )
    per_level_knn = [item[2][0] for item in raw]
    per_level_kde = [item[3][0] for item in raw]
    dispersions = [item[4] for item in raw]
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
    kde_unit_indices: Optional[np.ndarray] = None,
    build_details: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[GranularityResult]]:
    unit_count = band_energy.shape[1]
    band_count = band_energy.shape[2]
    standardized = all_unit_localization_features(band_energy)

    if not compute_kde or gb_cfg.kde_scope == "none":
        kde_set: Set[int] = set()
    elif gb_cfg.kde_scope == "all" or kde_unit_indices is None:
        kde_set = set(range(unit_count))
    else:
        kde_set = set(np.asarray(kde_unit_indices, dtype=np.int64).tolist())

    fused_knn = np.empty((unit_count, band_count), dtype=np.float64)
    fused_kde = np.full((unit_count, band_count), np.nan, dtype=np.float64)
    level_count = len(gb_cfg.purity_thresholds)
    all_weights = np.empty((unit_count, level_count), dtype=np.float64)
    all_dispersion = np.empty((unit_count, level_count), dtype=np.float64)

    chunk_size = max(1, int(gb_cfg.worker_chunk_size))

    def work_chunk(start: int, end: int):
        local_knn = np.empty((end - start, band_count), dtype=np.float64)
        local_kde = np.full((end - start, band_count), np.nan, dtype=np.float64)
        local_weights = np.empty((end - start, level_count), dtype=np.float64)
        local_dispersion = np.empty((end - start, level_count), dtype=np.float64)
        for offset, unit_id in enumerate(range(start, end)):
            knn, kde, weights, dispersion = _single_unit_local_mi(
                band_energy[:, unit_id, :],
                standardized[:, unit_id, :],
                events,
                mi_cfg,
                gb_cfg,
                compute_kde=unit_id in kde_set,
            )
            local_knn[offset] = knn
            if unit_id in kde_set:
                local_kde[offset] = kde
            local_weights[offset] = weights
            local_dispersion[offset] = dispersion
        return start, end, local_knn, local_kde, local_weights, local_dispersion

    chunks = [
        (start, min(start + chunk_size, unit_count))
        for start in range(0, unit_count, chunk_size)
    ]
    completed_units = 0

    def store(result) -> None:
        nonlocal completed_units
        start, end, knn, kde, weights, dispersion = result
        fused_knn[start:end] = knn
        fused_kde[start:end] = kde
        all_weights[start:end] = weights
        all_dispersion[start:end] = dispersion
        completed_units += end - start
        if completed_units == unit_count or completed_units // 512 != (completed_units - (end - start)) // 512:
            print(
                f"[granular-ball unit-local] {completed_units}/{unit_count} units",
                flush=True,
            )

    if gb_cfg.workers == 1 or len(chunks) == 1:
        for start, end in chunks:
            store(work_chunk(start, end))
    else:
        with ThreadPoolExecutor(max_workers=gb_cfg.workers) as executor:
            futures = [executor.submit(work_chunk, start, end) for start, end in chunks]
            for future in as_completed(futures):
                store(future.result())

    if not build_details:
        return fused_knn, fused_kde, []

    # Diagnostics are constructed once per layer. LCB repeats set
    # ``build_details=False`` and therefore avoid this extra hierarchy.
    diag_features = layer_localization_features(band_energy)
    diag_hierarchy = build_multigranularity_hierarchy(diag_features, events, gb_cfg)
    mean_weights = np.mean(all_weights, axis=0)
    mean_dispersion = np.mean(all_dispersion, axis=0)
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
    kde_unit_indices: Optional[np.ndarray] = None,
    build_details: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[GranularityResult]]:
    """Equation (8)/(23): local MI followed by adaptive multi-level fusion.

    Performance optimizations are algebraically exact for the primary kNN
    scores: MI is batched over bands, nested granularities reuse identical
    balls, unit features are standardized once, and LCB repeats can omit
    diagnostic-only hierarchies. ``kde_scope=probe`` only reduces the auxiliary
    KDE diagnostic and cannot change the pruning score or PPL.
    """
    if gb_cfg.localization_mode == "unit_local":
        return _unit_local_mi(
            band_energy,
            events,
            mi_cfg,
            gb_cfg,
            compute_kde=compute_kde,
            kde_unit_indices=kde_unit_indices,
            build_details=build_details,
        )
    return _shared_partition_mi(
        band_energy,
        events,
        mi_cfg,
        gb_cfg,
        compute_kde=compute_kde,
        kde_unit_indices=kde_unit_indices,
        build_details=build_details,
    )
