from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import BudgetConfig


def _rank01(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.nan_to_num(x, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    order = np.argsort(finite, kind="stable")
    result = np.empty_like(finite, dtype=np.float64)
    if finite.size == 1:
        result[order] = 1.0
    else:
        result[order] = np.linspace(0.0, 1.0, finite.size)
    return result


def _normalized_positive_bands(band_contribution: np.ndarray) -> np.ndarray:
    bands = np.asarray(band_contribution, dtype=np.float64)
    positive = np.maximum(np.nan_to_num(bands, nan=0.0), 0.0)
    band_scale = positive.sum(axis=0, keepdims=True)
    return positive / np.maximum(band_scale, 1e-12)


def _coverage_targets(cfg: BudgetConfig, band_count: int) -> np.ndarray:
    if cfg.coverage_ratios is None:
        return np.full(band_count, float(cfg.coverage_ratio), dtype=np.float64)
    values = np.asarray(cfg.coverage_ratios, dtype=np.float64)
    if values.size != band_count:
        raise ValueError(
            f"coverage_ratios has {values.size} values but there are {band_count} bands"
        )
    return values


def _band_weights(cfg: BudgetConfig, band_count: int) -> np.ndarray:
    if cfg.band_selection_weights is None:
        return np.ones(band_count, dtype=np.float64)
    values = np.asarray(cfg.band_selection_weights, dtype=np.float64)
    if values.size != band_count:
        raise ValueError(
            f"band_selection_weights has {values.size} values but there are {band_count} bands"
        )
    return values


def _achieved_coverage(normalized_bands: np.ndarray, keep: np.ndarray) -> np.ndarray:
    if keep.size == 0:
        return np.zeros(normalized_bands.shape[1], dtype=np.float64)
    return normalized_bands[keep].sum(axis=0)


def score_only_keep_indices(
    scores: np.ndarray,
    band_contribution: np.ndarray,
    keep_count: int,
    cfg: BudgetConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the highest scalar scores only (LCB-only for the full method)."""
    cfg.validate()
    raw = np.asarray(scores, dtype=np.float64).reshape(-1)
    bands = np.asarray(band_contribution, dtype=np.float64)
    if bands.ndim != 2 or bands.shape[0] != raw.size:
        raise ValueError("band_contribution must have shape [units,bands]")
    if not 1 <= keep_count <= raw.size:
        raise ValueError("keep_count must be in [1, unit_count]")
    finite = np.nan_to_num(raw, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    order = np.lexsort((np.arange(raw.size), -finite))
    keep = np.sort(order[:keep_count].astype(np.int64))
    priority = _rank01(raw)
    achieved = _achieved_coverage(_normalized_positive_bands(bands), keep)
    return keep, priority, achieved


def coverage_aware_keep_indices(
    scores: np.ndarray,
    band_contribution: np.ndarray,
    keep_count: int,
    cfg: BudgetConfig,
    costs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate proposal Eqs. (11)-(13) with per-band gamma_b targets.

    ``coverage_ratios`` implements the proposal's band-specific gamma_b. With
    three contiguous bands, values such as (0.85, 0.70, 0.55) express the
    requested low > mid > high retention profile without assigning a unit to
    only one band; a unit may contribute to several bands.
    """
    cfg.validate()
    raw_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    bands = np.asarray(band_contribution, dtype=np.float64)
    if bands.ndim != 2 or bands.shape[0] != raw_score.size:
        raise ValueError("band_contribution must have shape [units,bands]")
    unit_count = raw_score.size
    if not 1 <= keep_count <= unit_count:
        raise ValueError("keep_count must be in [1, unit_count]")
    if costs is None:
        cost = np.ones(unit_count, dtype=np.float64)
    else:
        cost = np.asarray(costs, dtype=np.float64).reshape(-1)
        if cost.size != unit_count or np.any(cost <= 0):
            raise ValueError("costs must be positive with one value per unit")

    score_term = _rank01(raw_score) / cost
    normalized_bands = _normalized_positive_bands(bands)
    target_coverage = _coverage_targets(cfg, bands.shape[1])
    current_coverage = np.zeros(bands.shape[1], dtype=np.float64)
    selected = np.zeros(unit_count, dtype=bool)
    priority_accumulator = np.zeros(unit_count, dtype=np.float64)
    remaining = keep_count
    batch_size = max(1, int(np.ceil(keep_count / cfg.greedy_batches)))

    # Coverage-first allocation approximates the hard constraint in Eq. (11).
    # At each step we serve the band with the largest absolute coverage deficit.
    # Therefore asymmetric gamma_b values such as low/mid/high=(.85,.70,.55)
    # naturally allocate more of a tight fixed budget to low frequency while
    # still allowing one unit to satisfy several bands simultaneously.
    blocked = np.zeros(bands.shape[1], dtype=bool)
    while remaining > 0:
        under = np.maximum(0.0, target_coverage - current_coverage)
        under[blocked] = 0.0
        if under.max(initial=0.0) <= 1e-12:
            break
        band = int(np.argmax(under))
        candidates = np.flatnonzero((~selected) & (normalized_bands[:, band] > 0))
        if candidates.size == 0:
            blocked[band] = True
            continue
        utility = normalized_bands[candidates, band] / cost[candidates]
        best_value = utility.max(initial=0.0)
        best_candidates = candidates[np.flatnonzero(utility == best_value)]
        if best_candidates.size > 1:
            best = int(best_candidates[np.argmax(score_term[best_candidates])])
        else:
            best = int(best_candidates[0])
        selected[best] = True
        priority_accumulator[best] = score_term[best] + cfg.coverage_alpha * under[band]
        current_coverage += normalized_bands[best]
        remaining -= 1

    while remaining > 0:
        under = np.maximum(0.0, target_coverage - current_coverage)
        marginal = score_term + cfg.coverage_alpha * (normalized_bands @ under) / cost
        marginal[selected] = -np.inf
        take = min(batch_size, remaining)
        candidate = np.argpartition(marginal, -take)[-take:]
        candidate = candidate[np.argsort(marginal[candidate], kind="stable")]
        selected[candidate] = True
        priority_accumulator[candidate] = marginal[candidate]
        current_coverage += normalized_bands[candidate].sum(axis=0)
        remaining -= take

    keep = np.flatnonzero(selected).astype(np.int64)
    final_priority = score_term.copy()
    selected_values = priority_accumulator[selected]
    if selected_values.size:
        final_priority[selected] += 1.0 + _rank01(selected_values)
    return np.sort(keep), final_priority, current_coverage


def band_only_keep_indices(
    band_contribution: np.ndarray,
    keep_count: int,
    cfg: BudgetConfig,
    costs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select units from frequency-band contribution only, without scalar LCB."""
    bands = np.asarray(band_contribution, dtype=np.float64)
    if bands.ndim != 2:
        raise ValueError("band_contribution must have shape [units,bands]")
    normalized = _normalized_positive_bands(bands)
    weights = _band_weights(cfg, bands.shape[1])
    band_score = normalized @ weights
    return coverage_aware_keep_indices(
        band_score, bands, keep_count, cfg, costs=costs
    )


def select_keep_indices(
    strategy: str,
    scores: np.ndarray,
    band_contribution: np.ndarray,
    keep_count: int,
    cfg: BudgetConfig,
    costs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Final-unit selection ablation.

    paper_hybrid: scalar contribution/LCB + band under-coverage (paper Eqs. 11-13)
    lcb_only: scalar score only; for paper_mi_gb_lcb this is LCB_i
    band_only: frequency-band contribution only, with optional low/mid/high weights
    """
    if strategy == "paper_hybrid":
        return coverage_aware_keep_indices(
            scores, band_contribution, keep_count, cfg, costs=costs
        )
    if strategy == "lcb_only":
        return score_only_keep_indices(scores, band_contribution, keep_count, cfg)
    if strategy == "band_only":
        return band_only_keep_indices(band_contribution, keep_count, cfg, costs=costs)
    raise ValueError(
        f"unknown final keep strategy {strategy!r}; choose paper_hybrid, lcb_only, band_only"
    )
