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


def coverage_aware_keep_indices(
    scores: np.ndarray,
    band_contribution: np.ndarray,
    keep_count: int,
    cfg: BudgetConfig,
    costs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate equations (11)-(13) with batched greedy selection.

    Returns selected keep indices, a coverage-aware priority for all units, and
    the achieved per-band coverage ratio. Batched updates preserve the greedy
    under-coverage feedback while avoiding O(U*K) work on 11k FFN channels.
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

    # Scores and band contributions have different scales across layers/types.
    # Rank scaling realizes the proposal's cross-unit scale alignment before
    # applying the cost-normalized marginal-gain rule.
    score_term = _rank01(raw_score) / cost
    positive_bands = np.maximum(np.nan_to_num(bands, nan=0.0), 0.0)
    band_scale = positive_bands.sum(axis=0, keepdims=True)
    normalized_bands = positive_bands / np.maximum(band_scale, 1e-12)

    target_coverage = cfg.coverage_ratio * normalized_bands.sum(axis=0)
    current_coverage = np.zeros(bands.shape[1], dtype=np.float64)
    selected = np.zeros(unit_count, dtype=bool)
    priority_accumulator = np.zeros(unit_count, dtype=np.float64)
    remaining = keep_count
    batch_size = max(1, int(np.ceil(keep_count / cfg.greedy_batches)))

    # Seed every non-empty frequency band before the score-driven fill. This
    # realizes the hard coverage intent in Eq. (11) and prevents a high-score
    # cluster from starving an entire low/mid/high-frequency region.
    for band in range(normalized_bands.shape[1]):
        if remaining <= 0 or normalized_bands[:, band].sum() <= 0:
            break
        candidates = np.flatnonzero(~selected)
        if candidates.size == 0:
            break
        utility = normalized_bands[candidates, band] / cost[candidates]
        best_value = utility.max(initial=0.0)
        best_candidates = candidates[np.flatnonzero(utility == best_value)]
        if best_candidates.size > 1:
            best = best_candidates[np.argmax(score_term[best_candidates])]
        else:
            best = int(best_candidates[0])
        selected[best] = True
        priority_accumulator[best] = score_term[best] + cfg.coverage_alpha
        current_coverage += normalized_bands[best]
        remaining -= 1

    while remaining > 0:
        under = np.maximum(0.0, target_coverage - current_coverage)
        marginal = score_term + cfg.coverage_alpha * (
            normalized_bands @ under
        ) / cost
        marginal[selected] = -np.inf
        take = min(batch_size, remaining)
        candidate = np.argpartition(marginal, -take)[-take:]
        candidate = candidate[np.argsort(marginal[candidate], kind="stable")]
        selected[candidate] = True
        priority_accumulator[candidate] = marginal[candidate]
        current_coverage += normalized_bands[candidate].sum(axis=0)
        remaining -= take

    keep = np.flatnonzero(selected).astype(np.int64)
    # Give every unit a smooth final priority for Wanda guidance. Selected units
    # receive their final marginal-gain rank; unselected units retain the base
    # contribution rank, avoiding a brittle binary step.
    final_priority = score_term.copy()
    selected_values = priority_accumulator[selected]
    if selected_values.size:
        final_priority[selected] += 1.0 + _rank01(selected_values)
    achieved = current_coverage / np.maximum(
        normalized_bands.sum(axis=0), 1e-12
    )
    return np.sort(keep), final_priority, achieved
