from __future__ import annotations

from typing import Optional

import numpy as np


def stratified_bootstrap_indices(
    events: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
    scenario_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Bootstrap within event or joint (event, scenario) strata."""
    y = np.asarray(events)
    if scenario_ids is None:
        strata = y.astype(str)
    else:
        scenarios = np.asarray(scenario_ids)
        if scenarios.shape != y.shape:
            raise ValueError("scenario_ids must have the same shape as events")
        strata = np.char.add(np.char.add(y.astype(str), "::"), scenarios.astype(str))

    selected = []
    for stratum in np.unique(strata):
        pool = np.flatnonzero(strata == stratum)
        count = max(2, int(np.ceil(pool.size * fraction)))
        selected.append(rng.choice(pool, size=count, replace=True))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result.astype(np.int64, copy=False)
