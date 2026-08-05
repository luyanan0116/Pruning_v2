from __future__ import annotations

from dataclasses import dataclass
from math import gamma, pi
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.fft import dct
from scipy.special import digamma

from .config import FrequencyConfig


@dataclass
class FrequencySpectrum:
    fine_energy: np.ndarray          # [N,U,Q]
    band_energy: np.ndarray          # [N,U,B]
    band_ranges: List[Tuple[int, int]]
    band_mi: np.ndarray              # primary KL/kNN estimator [U,B]
    band_mi_kde: np.ndarray          # auxiliary KDE estimator [U,B]
    total_mi: np.ndarray             # weighted primary score [U]
    probe_indices: np.ndarray
    fine_relevance: np.ndarray


def encode_events(events: np.ndarray) -> np.ndarray:
    y = np.asarray(events)
    if y.ndim != 1:
        raise ValueError(f"events must be one-dimensional, got {y.shape}")
    _, encoded = np.unique(y, return_inverse=True)
    if np.unique(encoded).size < 2:
        raise ValueError("events must contain at least two classes")
    return encoded.astype(np.int64, copy=False)


def standardize_responses(responses: np.ndarray, eps: float) -> np.ndarray:
    """Per-sample, per-unit standardization in the sequence dimension."""
    x = np.asarray(responses, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"responses must be [samples,units,sequence], got {x.shape}")
    mean = x.mean(axis=-1, keepdims=True, dtype=np.float32)
    std = x.std(axis=-1, keepdims=True, dtype=np.float32)
    return (x - mean) / (std + eps)


def proposal_dct(responses: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Unnormalised proposal DCT-II.

    SciPy's unnormalised DCT-II is twice the cosine sum written in the proposal,
    therefore division by two reproduces its displayed equation.
    """
    standardized = standardize_responses(responses, eps)
    return np.asarray(dct(standardized, type=2, axis=-1, norm=None, workers=1) / 2.0, dtype=np.float32)


def dct_frequency_energy(responses: np.ndarray, cfg: FrequencyConfig) -> np.ndarray:
    """DCT coefficients, squared energy, and contiguous fine-bin sums."""
    coefficients = proposal_dct(responses, cfg.eps)
    groups = [
        group for group in np.array_split(
            np.arange(coefficients.shape[-1]),
            min(cfg.fine_bins, coefficients.shape[-1]),
        ) if group.size
    ]
    return np.stack(
        [
            np.square(coefficients[..., group], dtype=np.float32).sum(axis=-1, dtype=np.float32)
            for group in groups
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def _kth_neighbor_radius_1d(values: np.ndarray, k_neighbors: int) -> np.ndarray:
    """Exact kth-neighbour radius for each scalar feature using sorted samples."""
    x = np.asarray(values, dtype=np.float64)
    n_samples, n_features = x.shape
    if n_samples <= k_neighbors:
        return np.zeros_like(x)
    scale = np.maximum(np.std(x, axis=0), 1.0)
    jitter = (
        np.arange(n_samples, dtype=np.float64)[:, None] - n_samples / 2.0
    ) * (1e-12 * scale[None, :])
    order = np.argsort(x + jitter, axis=0, kind="stable")
    sorted_x = np.take_along_axis(x + jitter, order, axis=0)
    candidates = []
    for offset in range(1, k_neighbors + 1):
        left = np.full((n_samples, n_features), np.inf, dtype=np.float64)
        right = np.full((n_samples, n_features), np.inf, dtype=np.float64)
        left[offset:] = sorted_x[offset:] - sorted_x[:-offset]
        right[:-offset] = sorted_x[offset:] - sorted_x[:-offset]
        candidates.extend([left, right])
    stacked = np.stack(candidates, axis=0)
    kth_sorted = np.partition(stacked, k_neighbors - 1, axis=0)[k_neighbors - 1]
    radii = np.empty_like(kth_sorted)
    np.put_along_axis(radii, order, kth_sorted, axis=0)
    return radii


def _kl_entropy_1d_matrix(values: np.ndarray, k_neighbors: int, eps: float = 1e-12) -> np.ndarray:
    """Kozachenko-Leonenko entropy for independent scalar feature columns."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n_samples, n_features = x.shape
    result = np.zeros(n_features, dtype=np.float64)
    if n_samples < 2:
        return result
    k = min(int(k_neighbors), n_samples - 1)
    active = np.var(x, axis=0) > 1e-14
    if not np.any(active):
        return result
    radii = _kth_neighbor_radius_1d(x[:, active], k)
    # Detailed equation (29), with d=1 and V_1=2.
    entropy = (
        digamma(n_samples)
        - digamma(k)
        + np.log(2.0)
        + np.mean(np.log(np.maximum(radii, eps)), axis=0)
    )
    entropy = np.where(np.isfinite(entropy), entropy, 0.0)
    result[active] = entropy
    return result


def estimate_knn_mi_matrix(
    values: np.ndarray,
    events: np.ndarray,
    cfg: FrequencyConfig,
    feature_chunk_size: int = 256,
) -> np.ndarray:
    """Entropy-decomposition kNN MI from detailed equations (28)-(29).

    Each column is one continuous response variable Z. The estimator computes
    H(Z)-sum_c p(c)H(Z|Y=c) with a Kozachenko-Leonenko entropy estimate.
    """
    x = np.asarray(values, dtype=np.float64)
    y = encode_events(events)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("values must have shape [samples,features]")
    result = np.zeros(x.shape[1], dtype=np.float64)
    classes, counts = np.unique(y, return_counts=True)
    if x.shape[0] < 4 or classes.size < 2 or counts.min() < 2:
        return result

    for start in range(0, x.shape[1], feature_chunk_size):
        end = min(start + feature_chunk_size, x.shape[1])
        chunk = x[:, start:end]
        total_entropy = _kl_entropy_1d_matrix(chunk, cfg.mi_neighbors)
        conditional = np.zeros(end - start, dtype=np.float64)
        for class_id, count in zip(classes, counts):
            class_values = chunk[y == class_id]
            class_k = min(cfg.mi_neighbors, int(count) - 1)
            conditional += (count / y.size) * _kl_entropy_1d_matrix(class_values, class_k)
        estimate = total_entropy - conditional
        result[start:end] = np.where(np.isfinite(estimate), np.maximum(estimate, 0.0), 0.0)
    return result


def _silverman_bandwidth(values: np.ndarray, scale: float) -> np.ndarray:
    n = values.shape[0]
    std = np.std(values, axis=0)
    iqr = np.subtract(*np.percentile(values, [75, 25], axis=0))
    robust = np.where(iqr > 0, np.minimum(std, iqr / 1.349), std)
    robust = np.where(robust > 1e-10, robust, np.maximum(std, 1e-3))
    return np.maximum(scale * 0.9 * robust * n ** (-1.0 / 5.0), 1e-6)


def estimate_kde_mi_matrix(
    values: np.ndarray,
    events: np.ndarray,
    cfg: FrequencyConfig,
    bandwidth: Optional[np.ndarray | float] = None,
    feature_chunk_size: int = 128,
) -> np.ndarray:
    """Gaussian-KDE auxiliary MI estimator, including ball-radius bandwidth."""
    x = np.asarray(values, dtype=np.float64)
    y = encode_events(events)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("values must have shape [samples,features]")
    n_samples, n_features = x.shape
    result = np.zeros(n_features, dtype=np.float64)
    classes, encoded, counts = np.unique(y, return_inverse=True, return_counts=True)
    if n_samples < 4 or classes.size < 2 or counts.min() < 2:
        return result
    class_counts = counts[encoded]
    same_class = encoded[:, None] == encoded[None, :]

    if bandwidth is None:
        bw_all = None
    else:
        bw_all = np.asarray(bandwidth, dtype=np.float64)
        if bw_all.ndim == 0:
            bw_all = np.full(n_features, float(bw_all), dtype=np.float64)
        if bw_all.size != n_features:
            raise ValueError("bandwidth must be scalar or one value per feature")

    for start in range(0, n_features, feature_chunk_size):
        end = min(start + feature_chunk_size, n_features)
        chunk = x[:, start:end]
        active = np.var(chunk, axis=0) > 1e-14
        if not np.any(active):
            continue
        bw = _silverman_bandwidth(chunk, cfg.kde_bandwidth_scale) if bw_all is None else np.maximum(bw_all[start:end], 1e-6)
        diff = (chunk[:, None, :] - chunk[None, :, :]) / bw[None, None, :]
        kernel = np.exp(-0.5 * np.square(diff)) / (np.sqrt(2.0 * np.pi) * bw[None, None, :])
        marginal = kernel.mean(axis=1)
        conditional = (kernel * same_class[:, :, None]).sum(axis=1) / class_counts[:, None]
        estimate = np.mean(np.log(conditional + 1e-12) - np.log(marginal + 1e-12), axis=0)
        result[start:end] = np.where(active & np.isfinite(estimate), np.maximum(estimate, 0.0), 0.0)
    return result


def choose_probe_units(fine_energy: np.ndarray, probe_count: int) -> np.ndarray:
    if fine_energy.ndim != 3:
        raise ValueError("fine_energy must be [samples,units,bins]")
    unit_count = fine_energy.shape[1]
    count = min(max(1, int(probe_count)), unit_count)
    activity = np.var(fine_energy, axis=(0, 2)) + np.mean(fine_energy, axis=(0, 2))
    order = np.argsort(activity, kind="stable")
    positions = np.linspace(0, unit_count - 1, count).round().astype(np.int64)
    return np.unique(order[positions]).astype(np.int64)


def aggregate_probe_energy(
    fine_energy: np.ndarray,
    probe_indices: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    return fine_energy[:, probe_indices, start:end].sum(axis=-1).mean(axis=1)


def _range_relevance(
    fine_energy: np.ndarray,
    events: np.ndarray,
    cfg: FrequencyConfig,
    probes: np.ndarray,
    start: int,
    end: int,
) -> float:
    values = aggregate_probe_energy(fine_energy, probes, start, end)
    return float(estimate_knn_mi_matrix(values, events, cfg)[0])


def adaptive_merge_adjacent_bins(
    fine_energy: np.ndarray,
    events: np.ndarray,
    cfg: FrequencyConfig,
    probe_indices: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """Greedily merge the adjacent pair with minimum information loss."""
    y = encode_events(events)
    probes = (
        choose_probe_units(fine_energy, cfg.probe_units)
        if probe_indices is None
        else np.asarray(probe_indices, dtype=np.int64)
    )
    ranges: List[Tuple[int, int]] = [(index, index + 1) for index in range(fine_energy.shape[-1])]
    cache: dict[Tuple[int, int], float] = {}

    def relevance(interval: Tuple[int, int]) -> float:
        if interval not in cache:
            cache[interval] = _range_relevance(fine_energy, y, cfg, probes, interval[0], interval[1])
        return cache[interval]

    values = [relevance(interval) for interval in ranges]
    fine_relevance = np.asarray(values, dtype=np.float64).copy()
    target = min(max(1, int(cfg.target_bands)), len(ranges))
    while len(ranges) > target:
        losses, unions = [], []
        for position in range(len(ranges) - 1):
            union = (ranges[position][0], ranges[position + 1][1])
            union_relevance = relevance(union)
            losses.append(values[position] + values[position + 1] - union_relevance)
            unions.append(union_relevance)
        position = int(np.argmin(np.asarray(losses)))
        if cfg.max_merge_loss is not None and losses[position] > cfg.max_merge_loss:
            break
        ranges[position : position + 2] = [(ranges[position][0], ranges[position + 1][1])]
        values[position : position + 2] = [unions[position]]
    return ranges, fine_relevance, probes


def aggregate_bands(fine_energy: np.ndarray, ranges: Sequence[Tuple[int, int]]) -> np.ndarray:
    return np.stack(
        [fine_energy[..., start:end].sum(axis=-1, dtype=np.float32) for start, end in ranges],
        axis=-1,
    ).astype(np.float32, copy=False)


def calculate_band_mi(
    band_energy: np.ndarray,
    events: np.ndarray,
    cfg: FrequencyConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, n_units, n_bands = band_energy.shape
    flat = np.asarray(band_energy, dtype=np.float64).reshape(n_samples, n_units * n_bands)
    primary = estimate_knn_mi_matrix(flat, events, cfg).reshape(n_units, n_bands)
    auxiliary = estimate_kde_mi_matrix(flat, events, cfg).reshape(n_units, n_bands)
    return primary, auxiliary


def normalized_band_weights(band_count: int, cfg: FrequencyConfig) -> np.ndarray:
    weights = (
        np.ones(band_count, dtype=np.float64)
        if cfg.band_weights is None
        else np.asarray(cfg.band_weights, dtype=np.float64)[:band_count]
    )
    if weights.sum() <= 0:
        raise ValueError("band weights must have positive sum")
    return weights / weights.sum()


def weighted_total_mi(band_mi: np.ndarray, cfg: FrequencyConfig) -> np.ndarray:
    if band_mi.ndim != 2:
        raise ValueError("band_mi must be [units,bands]")
    return np.asarray(band_mi @ normalized_band_weights(band_mi.shape[1], cfg), dtype=np.float64)


def build_spectrum_from_fine_energy(
    fine_energy: np.ndarray,
    events: np.ndarray,
    cfg: FrequencyConfig,
    band_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    probe_indices: Optional[np.ndarray] = None,
) -> FrequencySpectrum:
    y = encode_events(events)
    if band_ranges is None:
        ranges, fine_relevance, probes = adaptive_merge_adjacent_bins(
            fine_energy, y, cfg, probe_indices=probe_indices
        )
    else:
        ranges = [(int(start), int(end)) for start, end in band_ranges]
        probes = choose_probe_units(fine_energy, cfg.probe_units) if probe_indices is None else np.asarray(probe_indices, dtype=np.int64)
        fine_relevance = np.asarray([
            _range_relevance(fine_energy, y, cfg, probes, index, index + 1)
            for index in range(fine_energy.shape[-1])
        ])
    band_energy = aggregate_bands(fine_energy, ranges)
    band_mi, band_mi_kde = calculate_band_mi(band_energy, y, cfg)
    return FrequencySpectrum(
        fine_energy=np.asarray(fine_energy, dtype=np.float32),
        band_energy=band_energy,
        band_ranges=ranges,
        band_mi=band_mi,
        band_mi_kde=band_mi_kde,
        total_mi=weighted_total_mi(band_mi, cfg),
        probe_indices=probes,
        fine_relevance=fine_relevance,
    )


def build_frequency_spectrum(
    responses: np.ndarray,
    events: np.ndarray,
    cfg: FrequencyConfig,
) -> FrequencySpectrum:
    cfg.validate()
    return build_spectrum_from_fine_energy(dct_frequency_energy(responses, cfg), events, cfg)
