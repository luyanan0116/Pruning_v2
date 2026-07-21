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


def dual_source_bootstrap_indices(
    events: np.ndarray,
    base_sample_ids: np.ndarray,
    scenario_ids: np.ndarray,
    sample_fraction: float,
    scenario_fraction: float,
    rng: np.random.Generator,
    max_retries: int = 20,
) -> np.ndarray:
    """Two-stage bootstrap for sample-side and scenario-side uncertainty.

    Stage 1 resamples base calibration samples with replacement. Stage 2 samples
    scenario realizations for every selected base sample. This preserves the
    dependence among multiple scenarios derived from the same calibration
    sample, unlike treating every sample-scenario observation as independent.
    """
    y = np.asarray(events)
    base = np.asarray(base_sample_ids)
    scenario = np.asarray(scenario_ids)
    if not (y.shape == base.shape == scenario.shape):
        raise ValueError("events, base_sample_ids and scenario_ids must match")
    unique_base = np.unique(base)
    unique_scenarios = np.unique(scenario)
    sample_count = max(2, int(np.ceil(unique_base.size * sample_fraction)))
    scenario_count = max(1, int(np.ceil(unique_scenarios.size * scenario_fraction)))

    lookup = {
        (int(b), int(s)): np.flatnonzero((base == b) & (scenario == s))
        for b in unique_base
        for s in unique_scenarios
    }
    any_lookup = {int(b): np.flatnonzero(base == b) for b in unique_base}

    for _ in range(max_retries):
        chosen_base = rng.choice(unique_base, size=sample_count, replace=True)
        selected = []
        for b in chosen_base:
            chosen_scenarios = rng.choice(
                unique_scenarios, size=scenario_count, replace=True
            )
            for s in chosen_scenarios:
                pool = lookup[(int(b), int(s))]
                if pool.size == 0:
                    pool = any_lookup[int(b)]
                selected.append(int(rng.choice(pool)))
        result = np.asarray(selected, dtype=np.int64)
        rng.shuffle(result)
        classes, counts = np.unique(y[result], return_counts=True)
        if classes.size >= 2 and counts.min() >= 2:
            return result

    # Conservative fallback retains event/scenario support if a small data set
    # makes the two-stage draw degenerate.
    return stratified_bootstrap_indices(y, sample_fraction, rng, scenario)
