from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.fft import dct
from scipy.special import digamma

from .config import FrequencyConfig


@dataclass
class FrequencySpectrum:
    """Frequency-domain task-contribution spectrum for one transformer layer."""

    fine_energy: np.ndarray          # [N, U, Q]
    band_energy: np.ndarray          # [N, U, B]
    band_ranges: List[Tuple[int, int]]
    band_mi: np.ndarray              # primary kNN MI [U, B]
    band_mi_kde: np.ndarray          # auxiliary KDE MI [U, B]
    total_mi: np.ndarray             # primary total [U]
    probe_indices: np.ndarray        # representative units used for band merging
    fine_relevance: np.ndarray       # probe MI per fine bin


def encode_events(events: np.ndarray) -> np.ndarray:
    y = np.asarray(events)
    if y.ndim != 1:
        raise ValueError(f"events must be one-dimensional, got {y.shape}")
    _, encoded = np.unique(y, return_inverse=True)
    if np.unique(encoded).size < 2:
        raise ValueError("events must contain at least two classes")
    return encoded.astype(np.int64, copy=False)


def standardize_responses(responses: np.ndarray, eps: float) -> np.ndarray:
    """Equation (1)/(4): sample-wise, unit-wise sequence standardization."""
    x = np.asarray(responses, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"responses must be [samples, units, sequence], got {x.shape}")
    mean = x.mean(axis=-1, keepdims=True, dtype=np.float32)
    std = x.std(axis=-1, keepdims=True, dtype=np.float32)
    return (x - mean) / (std + eps)


def dct_frequency_energy(responses: np.ndarray, cfg: FrequencyConfig) -> np.ndarray:
    """Equations (2)-(3)/(5)-(7): DCT-II, squared energy, fine-bin sum."""
    standardized = standardize_responses(responses, cfg.eps)
    coefficients = dct(standardized, type=2, axis=-1, norm="ortho", workers=1)
    frequency_indices = [
        idx for idx in np.array_split(
            np.arange(coefficients.shape[-1]),
            min(cfg.fine_bins, coefficients.shape[-1]),
        ) if idx.size
    ]
    energy = np.stack(
        [
            np.square(coefficients[..., idx], dtype=np.float32).sum(
                axis=-1, dtype=np.float32
            )
            for idx in frequency_indices
        ],
        axis=-1,
    )
    return np.asarray(energy, dtype=np.float32)


def estimate_knn_mi_matrix(
    values: np.ndarray,
    events: np.ndarray,
    cfg: FrequencyConfig,
    feature_chunk_size: int = 256,
) -> np.ndarray:
    """Ross/Kraskov-style kNN MI for continuous variables and discrete events.

    This is the primary estimator used for contribution spectra and ranking.
    It is vectorized over features and chunked to bound the N x N x features
    temporary tensor.
    """
    x = np.asarray(values, dtype=np.float64)
    y = encode_events(events)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("values must have shape [samples, units]")
    n_samples, n_units = x.shape
    result = np.zeros(n_units, dtype=np.float64)
    classes, encoded, counts = np.unique(y, return_inverse=True, return_counts=True)
    if n_samples < 4 or classes.size < 2 or counts.min() < 2:
        return result

    class_counts = counts[encoded]
    k_per_sample = np.minimum(int(cfg.mi_neighbors), class_counts - 1).astype(np.int64)
    same_class = encoded[:, None] == encoded[None, :]
    np.fill_diagonal(same_class, False)
    constant_term = (
        digamma(n_samples)
        + np.mean(digamma(k_per_sample))
        - np.mean(digamma(class_counts))
    )

    for start in range(0, n_units, feature_chunk_size):
        end = min(start + feature_chunk_size, n_units)
        chunk = x[:, start:end].copy()
        variance = np.var(chunk, axis=0)
        active = variance > 1e-14
        if not np.any(active):
            continue

        scale = np.maximum(np.std(chunk, axis=0), 1.0)
        jitter = (
            np.arange(n_samples, dtype=np.float64)[:, None] - n_samples / 2.0
        ) * (1e-10 * scale[None, :])
        chunk += jitter
        distances = np.abs(chunk[:, None, :] - chunk[None, :, :])
        same_distances = np.where(same_class[:, :, None], distances, np.inf)
        radii = np.empty((n_samples, end - start), dtype=np.float64)
        for sample in range(n_samples):
            kth = int(k_per_sample[sample] - 1)
            radii[sample] = np.partition(same_distances[sample], kth, axis=0)[kth]
        radii = np.nextafter(radii, 0.0)
        neighbor_counts = np.sum(distances <= radii[:, None, :], axis=1) - 1
        neighbor_counts = np.maximum(neighbor_counts, 1)
        estimates = constant_term - np.mean(digamma(neighbor_counts + 1), axis=0)
        estimates = np.where(
            active & np.isfinite(estimates), np.maximum(estimates, 0.0), 0.0
        )
        result[start:end] = estimates
    return result


def estimate_kde_mi_matrix(
    values: np.ndarray,
    events: np.ndarray,
    cfg: FrequencyConfig,
    bandwidth: Optional[np.ndarray | float] = None,
    feature_chunk_size: int = 256,
) -> np.ndarray:
    """Auxiliary Gaussian-KDE MI estimator.

    Inside a granular ball, ``bandwidth`` can be tied to the ball radius, as
    required by the detailed proposal. The auxiliary estimator is reported for
    consistency checks but does not get averaged into the primary ranking.
    """
    x = np.asarray(values, dtype=np.float64)
    y = encode_events(events)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("values must have shape [samples, units]")
    n_samples, n_units = x.shape
    result = np.zeros(n_units, dtype=np.float64)
    classes, encoded, counts = np.unique(y, return_inverse=True, return_counts=True)
    if n_samples < 4 or classes.size < 2 or counts.min() < 2:
        return result
    class_counts = counts[encoded]
    same_class_with_self = encoded[:, None] == encoded[None, :]

    if bandwidth is not None:
        bw_all = np.asarray(bandwidth, dtype=np.float64)
        if bw_all.ndim == 0:
            bw_all = np.full(n_units, float(bw_all), dtype=np.float64)
        if bw_all.size != n_units:
            raise ValueError("bandwidth must be scalar or have one value per feature")
    else:
        bw_all = None

    for start in range(0, n_units, feature_chunk_size):
        end = min(start + feature_chunk_size, n_units)
        chunk = x[:, start:end]
        active = np.var(chunk, axis=0) > 1e-14
        if not np.any(active):
            continue
        distances = np.abs(chunk[:, None, :] - chunk[None, :, :])
        if bw_all is None:
            std = np.std(chunk, axis=0)
            iqr = np.subtract(*np.percentile(chunk, [75, 25], axis=0))
            robust_scale = np.where(iqr > 0, np.minimum(std, iqr / 1.349), std)
            robust_scale = np.where(
                robust_scale > 1e-10,
                robust_scale,
                np.maximum(np.abs(chunk.mean(axis=0)) * 1e-3, 1e-3),
            )
            bw = 0.9 * robust_scale * n_samples ** (-0.2)
        else:
            bw = bw_all[start:end]
        bw = np.maximum(1e-4, bw * cfg.kde_bandwidth_scale)
        normalized = distances / bw[None, None, :]
        kernel = np.exp(-0.5 * normalized * normalized) / (
            np.sqrt(2.0 * np.pi) * bw[None, None, :]
        )
        marginal = kernel.mean(axis=1)
        conditional = (
            kernel * same_class_with_self[:, :, None]
        ).sum(axis=1) / class_counts[:, None]
        estimates = np.mean(
            np.log(conditional + 1e-12) - np.log(marginal + 1e-12), axis=0
        )
        estimates = np.where(
            active & np.isfinite(estimates), np.maximum(estimates, 0.0), 0.0
        )
        result[start:end] = estimates
    return result


# Backward-compatible primary-estimator alias.
def estimate_mi_matrix(
    values: np.ndarray,
    events: np.ndarray,
    cfg: FrequencyConfig,
    seed_offset: int = 0,
    feature_chunk_size: int = 256,
) -> np.ndarray:
    del seed_offset
    return estimate_knn_mi_matrix(values, events, cfg, feature_chunk_size)


def choose_probe_units(fine_energy: np.ndarray, probe_count: int) -> np.ndarray:
    """Select representative units spanning the layer response-energy range."""
    if fine_energy.ndim != 3:
        raise ValueError("fine_energy must be [samples, units, bins]")
    unit_count = fine_energy.shape[1]
    count = min(max(1, int(probe_count)), unit_count)
    activity = np.var(fine_energy, axis=(0, 2)) + np.mean(fine_energy, axis=(0, 2))
    order = np.argsort(activity, kind="stable")
    positions = np.linspace(0, unit_count - 1, count).round().astype(np.int64)
    return np.unique(order[positions]).astype(np.int64)


def aggregate_probe_energy(
    fine_energy: np.ndarray, probe_indices: np.ndarray, start: int, end: int
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
    """Task-relevance-driven adjacent merging, detailed equations (10)-(12).

    Each round chooses the adjacent pair with the smallest information-loss
    quantity r(left)+r(right)-r(union), and recomputes the union MI rather than
    merely comparing neighboring scalar relevance values.
    """
    y = encode_events(events)
    probes = choose_probe_units(fine_energy, cfg.probe_units) if probe_indices is None else np.asarray(probe_indices, dtype=np.int64)
    ranges: List[Tuple[int, int]] = [(q, q + 1) for q in range(fine_energy.shape[-1])]
    relevance = [
        _range_relevance(fine_energy, y, cfg, probes, start, end)
        for start, end in ranges
    ]
    fine_relevance = np.asarray(relevance, dtype=np.float64).copy()

    target = min(max(1, int(cfg.target_bands)), len(ranges))
    while len(ranges) > target:
        losses = []
        unions = []
        for pos in range(len(ranges) - 1):
            union = (ranges[pos][0], ranges[pos + 1][1])
            union_rel = _range_relevance(
                fine_energy, y, cfg, probes, union[0], union[1]
            )
            loss = relevance[pos] + relevance[pos + 1] - union_rel
            losses.append(loss)
            unions.append(union_rel)
        pos = int(np.argmin(np.asarray(losses)))
        ranges[pos : pos + 2] = [(ranges[pos][0], ranges[pos + 1][1])]
        relevance[pos : pos + 2] = [unions[pos]]
    return ranges, fine_relevance, probes


def merge_adjacent_bins(relevance: np.ndarray, target_bands: int) -> List[Tuple[int, int]]:
    """Legacy scalar-only merging retained for external callers/tests."""
    relevance = np.asarray(relevance, dtype=np.float64)
    if relevance.ndim != 1 or relevance.size == 0:
        raise ValueError("relevance must be a non-empty vector")
    target_bands = min(max(1, int(target_bands)), relevance.size)
    ranges: List[Tuple[int, int]] = [(i, i + 1) for i in range(relevance.size)]
    values = relevance.tolist()
    while len(ranges) > target_bands:
        losses = [abs(values[i] - values[i + 1]) for i in range(len(ranges) - 1)]
        pos = int(np.argmin(losses))
        left, right = ranges[pos], ranges[pos + 1]
        ranges[pos : pos + 2] = [(left[0], right[1])]
        values[pos : pos + 2] = [(values[pos] + values[pos + 1]) / 2.0]
    return ranges


def aggregate_bands(
    fine_energy: np.ndarray, ranges: Sequence[Tuple[int, int]]
) -> np.ndarray:
    return np.stack(
        [
            fine_energy[..., start:end].sum(axis=-1, dtype=np.float32)
            for start, end in ranges
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def calculate_band_mi(
    band_energy: np.ndarray, events: np.ndarray, cfg: FrequencyConfig
) -> Tuple[np.ndarray, np.ndarray]:
    n_units, n_bands = band_energy.shape[1], band_energy.shape[2]
    knn = np.zeros((n_units, n_bands), dtype=np.float64)
    kde = np.zeros((n_units, n_bands), dtype=np.float64)
    for band in range(n_bands):
        values = band_energy[:, :, band]
        knn[:, band] = estimate_knn_mi_matrix(values, events, cfg)
        kde[:, band] = estimate_kde_mi_matrix(values, events, cfg)
    return knn, kde


def normalized_band_weights(band_count: int, cfg: FrequencyConfig) -> np.ndarray:
    if cfg.band_weights is None:
        weights = np.ones(band_count, dtype=np.float64)
    else:
        weights = np.asarray(cfg.band_weights, dtype=np.float64)[:band_count]
    total = weights.sum()
    if total <= 0:
        raise ValueError("band weights must have positive sum")
    return weights / total


def weighted_total_mi(band_mi: np.ndarray, cfg: FrequencyConfig) -> np.ndarray:
    if band_mi.ndim != 2:
        raise ValueError("band_mi must be [units, bands]")
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
        ranges = [(int(a), int(b)) for a, b in band_ranges]
        probes = choose_probe_units(fine_energy, cfg.probe_units) if probe_indices is None else np.asarray(probe_indices, dtype=np.int64)
        fine_relevance = np.asarray([
            _range_relevance(fine_energy, y, cfg, probes, q, q + 1)
            for q in range(fine_energy.shape[-1])
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
    responses: np.ndarray, events: np.ndarray, cfg: FrequencyConfig
) -> FrequencySpectrum:
    cfg.validate()
    fine_energy = dct_frequency_energy(responses, cfg)
    return build_spectrum_from_fine_energy(fine_energy, events, cfg)
