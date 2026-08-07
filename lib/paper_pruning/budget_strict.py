from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import BudgetConfig


@dataclass(frozen=True)
class BudgetSelection:
    keep_indices: np.ndarray
    achieved_coverage: np.ndarray
    spent_cost: float
    budget: float
    total_cost: float
    target_coverage: np.ndarray
    final_coverage: np.ndarray


def select_keep_set_eq11_13(
    lcb_scores: np.ndarray,
    robust_band_contribution: np.ndarray,
    costs: np.ndarray,
    budget: float,
    cfg: BudgetConfig,
    mandatory_indices: Optional[np.ndarray] = None,
) -> BudgetSelection:
    """Greedy approximation of sample.pdf Eq.(11)-(13).

    Eq.(11): hard per-band coverage and total resource budget.
    Eq.(12): w_b(K)=max(0, gamma_b*C_b(U)-C_b(K)).
    Eq.(13): Delta(i|K)=LCB_i/cost_i +
             alpha*sum_b w_b(K)*I_i(b)/cost_i.

    The implementation first runs Eq.(13) while any hard coverage constraint is
    unsatisfied, then fills the remaining deployment budget by LCB/cost (the
    Eq.(13) form after all w_b become zero). It raises instead of silently
    returning an infeasible K.
    """
    cfg.validate()
    score = np.asarray(lcb_scores, dtype=np.float64).reshape(-1)
    bands = np.asarray(robust_band_contribution, dtype=np.float64)
    cost = np.asarray(costs, dtype=np.float64).reshape(-1)
    if bands.ndim != 2 or bands.shape[0] != score.size:
        raise ValueError("robust_band_contribution must be [units,bands]")
    if cost.size != score.size or np.any(cost <= 0):
        raise ValueError("costs must be positive with one value per unit")
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be positive")

    n = score.size
    positive_bands = np.maximum(np.nan_to_num(bands, nan=0.0), 0.0)
    total_band = positive_bands.sum(axis=0)
    target = cfg.coverage_ratio * total_band
    total_cost = float(cost.sum())
    budget = min(float(budget), total_cost)

    selected = np.zeros(n, dtype=bool)
    if mandatory_indices is not None:
        mandatory = np.unique(np.asarray(mandatory_indices, dtype=np.int64))
        if mandatory.size and (mandatory.min() < 0 or mandatory.max() >= n):
            raise IndexError("mandatory index out of range")
        selected[mandatory] = True

    spent = float(cost[selected].sum())
    if spent > budget + 1e-9:
        raise ValueError(
            f"mandatory structural-validity set costs {spent:.6g}, above budget {budget:.6g}"
        )
    current = positive_bands[selected].sum(axis=0) if selected.any() else np.zeros_like(total_band)

    # Eq.(13) hard-coverage phase. Batch size controls runtime on hundreds of
    # thousands of FFN candidates, while every batch recomputes Eq.(12).
    batch_size = max(1, int(np.ceil(n / max(1, cfg.greedy_batches))))
    tol = 1e-12

    while np.any(current + tol < target):
        deficit = np.maximum(0.0, target - current)  # Eq.(12)
        remaining_budget = budget - spent
        feasible = (~selected) & (cost <= remaining_budget + tol)
        candidates = np.flatnonzero(feasible)
        if candidates.size == 0:
            achieved = np.divide(
                current, np.maximum(total_band, tol),
                out=np.ones_like(current), where=total_band > tol
            )
            raise ValueError(
                "Eq.(11) hard frequency coverage is infeasible under the requested budget; "
                f"achieved={achieved.tolist()}, target_ratio={cfg.coverage_ratio:.4f}"
            )

        marginal = (
            np.nan_to_num(score, nan=-np.inf) / cost
            + cfg.coverage_alpha * (positive_bands @ deficit) / cost
        )
        local = marginal[candidates]
        take = min(batch_size, candidates.size)
        top_pos = np.argpartition(local, -take)[-take:]
        top = candidates[top_pos]
        top = top[np.argsort(marginal[top], kind="stable")[::-1]]

        added = 0
        for idx in top:
            if selected[idx] or spent + cost[idx] > budget + tol:
                continue
            selected[idx] = True
            spent += float(cost[idx])
            current += positive_bands[idx]
            added += 1
            if np.all(current + tol >= target):
                break
        if added == 0:
            raise ValueError("Eq.(11) coverage could not progress without violating budget")

    # With all Eq.(12) deficits equal to zero, Eq.(13) reduces to LCB/cost.
    remaining = np.flatnonzero(~selected)
    efficiency = np.nan_to_num(score[remaining], nan=-np.inf) / cost[remaining]
    order = remaining[np.argsort(efficiency, kind="stable")[::-1]]
    for idx in order:
        if spent + cost[idx] <= budget + tol:
            selected[idx] = True
            spent += float(cost[idx])

    final = positive_bands[selected].sum(axis=0)
    achieved = np.divide(
        final, np.maximum(total_band, tol),
        out=np.ones_like(final), where=total_band > tol
    )
    if np.any(final + tol < target):
        raise RuntimeError("internal error: final K violates Eq.(11) hard coverage")

    return BudgetSelection(
        keep_indices=np.flatnonzero(selected).astype(np.int64),
        achieved_coverage=achieved,
        spent_cost=spent,
        budget=budget,
        total_cost=total_cost,
        target_coverage=target,
        final_coverage=final,
    )
