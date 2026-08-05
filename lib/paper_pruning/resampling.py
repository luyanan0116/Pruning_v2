from __future__ import annotations

from typing import Optional

import numpy as np


def stratified_bootstrap_indices(
    events: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
    scenario_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Bootstrap within event or joint event/scenario strata."""
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
        candidates = np.flatnonzero(strata == stratum)
        count = max(2, int(np.ceil(candidates.size * fraction)))
        selected.append(rng.choice(candidates, size=count, replace=True))
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
    """Paired two-stage repeated estimation for sample and scenario sources.

    Stage 1 samples base calibration examples with replacement. Stage 2 samples
    only the scenario realizations actually available for each selected base
    example. This preserves the pairing structure and never substitutes an
    observation from a different scenario when a manifest is unbalanced.
    """
    y = np.asarray(events)
    base = np.asarray(base_sample_ids)
    scenario = np.asarray(scenario_ids)
    if not (y.shape == base.shape == scenario.shape):
        raise ValueError("events, base_sample_ids and scenario_ids must match")
    unique_base = np.unique(base)
    sample_count = max(2, int(np.ceil(unique_base.size * sample_fraction)))

    by_base: dict[object, dict[object, np.ndarray]] = {}
    for base_id in unique_base:
        base_mask = base == base_id
        available = np.unique(scenario[base_mask])
        by_base[base_id.item() if hasattr(base_id, "item") else base_id] = {
            scenario_id.item() if hasattr(scenario_id, "item") else scenario_id: np.flatnonzero(
                base_mask & (scenario == scenario_id)
            )
            for scenario_id in available
        }

    for _ in range(max_retries):
        chosen_base = rng.choice(unique_base, size=sample_count, replace=True)
        selected: list[int] = []
        for base_id in chosen_base:
            key = base_id.item() if hasattr(base_id, "item") else base_id
            groups = by_base[key]
            available = np.asarray(list(groups.keys()), dtype=object)
            scenario_count = max(1, int(np.ceil(available.size * scenario_fraction)))
            chosen_scenarios = rng.choice(available, size=scenario_count, replace=True)
            for scenario_id in chosen_scenarios:
                scenario_key = scenario_id.item() if hasattr(scenario_id, "item") else scenario_id
                selected.append(int(rng.choice(groups[scenario_key])))
        result = np.asarray(selected, dtype=np.int64)
        rng.shuffle(result)
        classes, counts = np.unique(y[result], return_counts=True)
        if classes.size >= 2 and counts.min() >= 2:
            return result

    # Conservative fallback retains event/scenario support for very small data.
    return stratified_bootstrap_indices(y, sample_fraction, rng, scenario)
