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
    band_mi: np.ndarray              # [U, B]
    total_mi: np.ndarray             # [U]


def encode_events(events: np.ndarray) -> np.ndarray:
    y = np.asarray(events)
    if y.ndim != 1:
        raise ValueError(f"events must be one-dimensional, got {y.shape}")
    _, encoded = np.unique(y, return_inverse=True)
    if np.unique(encoded).size < 2:
        raise ValueError("events must contain at least two classes")
    return encoded.astype(np.int64, copy=False)


def standardize_responses(responses: np.ndarray, eps: float) -> np.ndarray:
    """Equation (1), independently normalizing every sample and unit sequence."""
    x = np.asarray(responses, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"responses must be [samples, units, sequence], got {x.shape}")
    mean = x.mean(axis=-1, keepdims=True, dtype=np.float32)
    std = x.std(axis=-1, keepdims=True, dtype=np.float32)
    return (x - mean) / (std + eps)


def dct_frequency_energy(responses: np.ndarray, cfg: FrequencyConfig) -> np.ndarray:
    """Equations (2)-(3): orthonormal DCT-II and squared energy in fine bins."""
    standardized = standardize_responses(responses, cfg.eps)
    coefficients = dct(standardized, type=2, axis=-1, norm="ortho", workers=1)
    frequency_indices = [idx for idx in np.array_split(np.arange(coefficients.shape[-1]), min(cfg.fine_bins, coefficients.shape[-1])) if idx.size]
    energy = np.stack(
        [np.square(coefficients[..., idx], dtype=np.float32).sum(axis=-1, dtype=np.float32) for idx in frequency_indices],
        axis=-1,
    )
    return np.asarray(energy, dtype=np.float32)


def estimate_mi_matrix(
    values: np.ndarray,
    events: np.ndarray,
    cfg: FrequencyConfig,
    seed_offset: int = 0,
    feature_chunk_size: int = 256,
) -> np.ndarray:
    """Vectorized Ross-style kNN MI for continuous units and a discrete event.

    The computation is chunked over units to keep the temporary pairwise-distance
    tensor bounded. It is substantially faster than invoking a Python/scikit-learn
    estimator once per channel and is the primary non-parametric estimator used by
    both global and local contribution calculations.
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
    same_class_with_self = encoded[:, None] == encoded[None, :]
    same_class = same_class_with_self.copy()
    diagonal = np.eye(n_samples, dtype=bool)
    same_class[diagonal] = False
    constant_term = digamma(n_samples) + np.mean(digamma(k_per_sample)) - np.mean(digamma(class_counts))

    for start in range(0, n_units, feature_chunk_size):
        end = min(start + feature_chunk_size, n_units)
        chunk = x[:, start:end].copy()
        variance = np.var(chunk, axis=0)
        active = variance > 1e-14
        if not np.any(active):
            continue
        # Deterministic tiny jitter avoids zero-radius neighborhoods for duplicates.
        scale = np.maximum(np.std(chunk, axis=0), 1.0)
        jitter = (np.arange(n_samples, dtype=np.float64)[:, None] - n_samples / 2.0) * (1e-10 * scale[None, :])
        chunk += jitter
        distances = np.abs(chunk[:, None, :] - chunk[None, :, :])  # [N,N,C]
        same_distances = np.where(same_class[:, :, None], distances, np.inf)
        radii = np.empty((n_samples, end - start), dtype=np.float64)
        for sample in range(n_samples):
            kth = int(k_per_sample[sample] - 1)
            radii[sample] = np.partition(same_distances[sample], kth, axis=0)[kth]
        radii = np.nextafter(radii, 0.0)
        neighbor_counts = np.sum(distances <= radii[:, None, :], axis=1) - 1
        neighbor_counts = np.maximum(neighbor_counts, 1)
        knn_estimates = constant_term - np.mean(digamma(neighbor_counts + 1), axis=0)

        # Auxiliary Gaussian KDE estimator: E[log p(x|Y) - log p(x)].
        std = np.std(chunk, axis=0)
        iqr = np.subtract(*np.percentile(chunk, [75, 25], axis=0))
        robust_scale = np.where(iqr > 0, np.minimum(std, iqr / 1.349), std)
        robust_scale = np.where(robust_scale > 1e-10, robust_scale, np.maximum(np.abs(chunk.mean(axis=0)) * 1e-3, 1e-3))
        bandwidth = np.maximum(1e-4, 0.9 * robust_scale * n_samples ** (-0.2))
        normalized = distances / bandwidth[None, None, :]
        kernel = np.exp(-0.5 * normalized * normalized) / (np.sqrt(2.0 * np.pi) * bandwidth[None, None, :])
        marginal = kernel.mean(axis=1)
        conditional = (kernel * same_class_with_self[:, :, None]).sum(axis=1) / class_counts[:, None]
        kde_estimates = np.mean(np.log(conditional + 1e-12) - np.log(marginal + 1e-12), axis=0)

        total_weight = cfg.knn_weight + cfg.kde_weight
        estimates = (cfg.knn_weight * knn_estimates + cfg.kde_weight * kde_estimates) / total_weight
        estimates = np.where(active & np.isfinite(estimates), np.maximum(estimates, 0.0), 0.0)
        result[start:end] = estimates
    return result

def fine_bin_relevance(fine_energy: np.ndarray, events: np.ndarray, cfg: FrequencyConfig) -> np.ndarray:
    """Layer-level task relevance used to merge adjacent fine bins."""
    if fine_energy.ndim != 3:
        raise ValueError("fine_energy must be [samples, units, bins]")
    relevance = np.zeros(fine_energy.shape[-1], dtype=np.float64)
    for q in range(fine_energy.shape[-1]):
        relevance[q] = float(estimate_mi_matrix(fine_energy[:, :, q], events, cfg, seed_offset=q).mean())
    return relevance


def merge_adjacent_bins(relevance: np.ndarray, target_bands: int) -> List[Tuple[int, int]]:
    """Task-relevance-driven contiguous merging from fine bins to frequency bands."""
    relevance = np.asarray(relevance, dtype=np.float64)
    if relevance.ndim != 1 or relevance.size == 0:
        raise ValueError("relevance must be a non-empty vector")
    target_bands = min(max(1, int(target_bands)), relevance.size)
    ranges: List[Tuple[int, int]] = [(i, i + 1) for i in range(relevance.size)]
    band_relevance = relevance.tolist()
    widths = [1] * relevance.size
    while len(ranges) > target_bands:
        losses = [abs(band_relevance[i] - band_relevance[i + 1]) for i in range(len(ranges) - 1)]
        pos = int(np.argmin(losses))
        left, right = ranges[pos], ranges[pos + 1]
        width = widths[pos] + widths[pos + 1]
        merged_relevance = (
            band_relevance[pos] * widths[pos] + band_relevance[pos + 1] * widths[pos + 1]
        ) / width
        ranges[pos : pos + 2] = [(left[0], right[1])]
        band_relevance[pos : pos + 2] = [merged_relevance]
        widths[pos : pos + 2] = [width]
    return ranges


def aggregate_bands(fine_energy: np.ndarray, ranges: Sequence[Tuple[int, int]]) -> np.ndarray:
    return np.stack(
        [fine_energy[..., start:end].sum(axis=-1, dtype=np.float32) for start, end in ranges],
        axis=-1,
    ).astype(np.float32, copy=False)


def calculate_band_mi(band_energy: np.ndarray, events: np.ndarray, cfg: FrequencyConfig) -> np.ndarray:
    n_units, n_bands = band_energy.shape[1], band_energy.shape[2]
    scores = np.zeros((n_units, n_bands), dtype=np.float64)
    for band in range(n_bands):
        scores[:, band] = estimate_mi_matrix(
            band_energy[:, :, band], events, cfg, seed_offset=10_000 + band
        )
    return scores


def weighted_total_mi(band_mi: np.ndarray, cfg: FrequencyConfig) -> np.ndarray:
    if band_mi.ndim != 2:
        raise ValueError("band_mi must be [units, bands]")
    if cfg.band_weights is None:
        weights = np.ones(band_mi.shape[1], dtype=np.float64)
    else:
        weights = np.asarray(cfg.band_weights, dtype=np.float64)
        if weights.size != band_mi.shape[1]:
            # This can occur only when sequence length is shorter than fine_bins.
            weights = weights[: band_mi.shape[1]]
    weights /= weights.sum()
    return np.asarray(band_mi @ weights, dtype=np.float64)


def build_spectrum_from_fine_energy(
    fine_energy: np.ndarray,
    events: np.ndarray,
    cfg: FrequencyConfig,
    band_ranges: Optional[Sequence[Tuple[int, int]]] = None,
) -> FrequencySpectrum:
    y = encode_events(events)
    if band_ranges is None:
        relevance = fine_bin_relevance(fine_energy, y, cfg)
        band_ranges = merge_adjacent_bins(relevance, cfg.target_bands)
    ranges = [(int(a), int(b)) for a, b in band_ranges]
    band_energy = aggregate_bands(fine_energy, ranges)
    band_mi = calculate_band_mi(band_energy, y, cfg)
    return FrequencySpectrum(
        fine_energy=fine_energy,
        band_energy=band_energy,
        band_ranges=ranges,
        band_mi=band_mi,
        total_mi=weighted_total_mi(band_mi, cfg),
    )


def build_frequency_spectrum(responses: np.ndarray, events: np.ndarray, cfg: FrequencyConfig) -> FrequencySpectrum:
    cfg.validate()
    fine_energy = dct_frequency_energy(responses, cfg)
    return build_spectrum_from_fine_energy(fine_energy, events, cfg)
