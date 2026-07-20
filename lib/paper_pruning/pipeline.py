from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import PipelineConfig
from .granular_ball import GranularityResult, multi_granularity_local_mi
from .mi import (
    FrequencySpectrum,
    build_frequency_spectrum,
    build_spectrum_from_fine_energy,
    weighted_total_mi,
)
from .resampling import stratified_bootstrap_indices


@dataclass
class LayerAblationScores:
    global_spectrum: FrequencySpectrum
    granular_band_mi: np.ndarray
    granularities: List[GranularityResult]
    mi_score: np.ndarray
    granular_score: np.ndarray
    bootstrap_scores: np.ndarray
    lcb_mean: np.ndarray
    lcb_std: np.ndarray
    lcb_score: np.ndarray


def score_layer(
    responses: np.ndarray,
    events: np.ndarray,
    scenario_ids: Optional[np.ndarray],
    config: PipelineConfig,
    layer_id: int = 0,
) -> LayerAblationScores:
    """Compute the three ablation paths for one layer.

    Path A: global frequency-domain MI.
    Path B: MI estimated inside purity-driven, multi-granularity balls.
    Path C: repeated Path B estimates followed by mean - lambda * std.
    """
    config.validate()
    x = np.asarray(responses)
    y = np.asarray(events)
    scenarios = None if scenario_ids is None else np.asarray(scenario_ids)
    if x.ndim != 3:
        raise ValueError("responses must be [samples, units, sequence]")
    if x.shape[0] != y.size:
        raise ValueError("response/event sample counts differ")
    if scenarios is not None and scenarios.size != y.size:
        raise ValueError("scenario_ids must match events")

    spectrum = build_frequency_spectrum(x, y, config.frequency)
    granular_band_mi, granularities = multi_granularity_local_mi(
        spectrum.band_energy,
        y,
        config.frequency,
        config.granular_ball,
    )
    granular_score = weighted_total_mi(granular_band_mi, config.frequency)

    rng = np.random.default_rng(config.lcb.random_state + int(layer_id) * 1009)
    repeated_scores = []
    for repeat in range(config.lcb.repeats):
        stratify_scenarios = scenarios if config.lcb.stratify_by_scenario else None
        indices = stratified_bootstrap_indices(
            y,
            config.lcb.sample_fraction,
            rng,
            stratify_scenarios,
        )
        # Eq. (1)-(3) are sample-wise, so cached fine energy can be safely indexed.
        # Adjacent band merging and all local balls are rebuilt in every repeat.
        repeat_spectrum = build_spectrum_from_fine_energy(
            spectrum.fine_energy[indices],
            y[indices],
            config.frequency,
        )
        repeat_local_mi, _ = multi_granularity_local_mi(
            repeat_spectrum.band_energy,
            y[indices],
            config.frequency,
            config.granular_ball,
        )
        repeated_scores.append(weighted_total_mi(repeat_local_mi, config.frequency))

    bootstrap = np.stack(repeated_scores, axis=0)
    mean = bootstrap.mean(axis=0)
    std = bootstrap.std(axis=0, ddof=1)
    lcb = mean - config.lcb.lcb_lambda * std

    return LayerAblationScores(
        global_spectrum=spectrum,
        granular_band_mi=granular_band_mi,
        granularities=granularities,
        mi_score=spectrum.total_mi,
        granular_score=granular_score,
        bootstrap_scores=bootstrap,
        lcb_mean=mean,
        lcb_std=std,
        lcb_score=lcb,
    )
