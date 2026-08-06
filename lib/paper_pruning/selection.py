from __future__ import annotations

from typing import List

import numpy as np


def resolve_prune_count(unit_count: int, sparsity_ratio: float, prune_per_layer: int) -> int:
    """Resolve final per-layer budget.

    A positive ``prune_per_layer`` means an exact count. Set it to zero to use
    ``sparsity_ratio``. The iterative selection step is controlled separately.
    """
    if unit_count < 2:
        raise ValueError("unit_count must be >= 2")
    if prune_per_layer > 0:
        count = int(prune_per_layer)
    else:
        if not 0 < sparsity_ratio < 1:
            raise ValueError("sparsity_ratio must be in (0,1) when prune_per_layer is zero")
        count = int(round(unit_count * float(sparsity_ratio)))
    return min(max(1, count), unit_count - 1)


def select_bottom_k(scores: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 1 <= k < values.size:
        raise ValueError(f"cannot prune {k} of {values.size} units")
    # NaN/inf are treated as least trustworthy and therefore pruned first.
    finite_values = np.nan_to_num(values, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    indices = np.lexsort((np.arange(values.size), finite_values))[:k]
    return np.sort(indices.astype(np.int64))


def split_into_steps(indices: np.ndarray, step_size: int = 10) -> List[np.ndarray]:
    values = np.asarray(indices, dtype=np.int64)
    if step_size < 1:
        raise ValueError("step_size must be >= 1")
    return [values[start : start + step_size] for start in range(0, values.size, step_size)]
