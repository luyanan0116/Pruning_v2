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
from .resampling import dual_source_bootstrap_indices, stratified_bootstrap_indices


@dataclass
class LayerAblationScores:
    global_spectrum: FrequencySpectrum
    granular_band_mi: np.ndarray
    granular_band_mi_kde: np.ndarray
    granularities: List[GranularityResult]
    mi_score: np.ndarray
    granular_score: np.ndarray
    bootstrap_scores: np.ndarray
    bootstrap_band_scores: np.ndarray
    lcb_mean: np.ndarray
    lcb_std: np.ndarray
    lcb_score: np.ndarray
    lcb_band_mean: np.ndarray
    lcb_band_std: np.ndarray


def score_layer(
    responses: np.ndarray,
    events: np.ndarray,
    scenario_ids: Optional[np.ndarray],
    config: PipelineConfig,
    layer_id: int = 0,
    base_sample_ids: Optional[np.ndarray] = None,
) -> LayerAblationScores:
    """Compute MI, MI+granular-ball and MI+granular-ball+LCB.

    Every LCB repeat rebuilds task-driven frequency bands, granular balls and
    local MI estimates. When base-sample IDs are available, repeated estimates
    use a two-stage sample/scenario bootstrap matching the proposal's dual-source
    uncertainty model.
    """
    config.validate()
    x = np.asarray(responses)
    y = np.asarray(events)
    scenarios = None if scenario_ids is None else np.asarray(scenario_ids)
    base_ids = None if base_sample_ids is None else np.asarray(base_sample_ids)
    if x.ndim != 3:
        raise ValueError("responses must be [samples, units, sequence]")
    if x.shape[0] != y.size:
        raise ValueError("response/event sample counts differ")
    if scenarios is not None and scenarios.size != y.size:
        raise ValueError("scenario_ids must match events")
    if base_ids is not None and base_ids.size != y.size:
        raise ValueError("base_sample_ids must match events")

    spectrum = build_frequency_spectrum(x, y, config.frequency)
    granular_band_mi, granular_band_mi_kde, granularities = (
        multi_granularity_local_mi(
            spectrum.band_energy,
            y,
            config.frequency,
            config.granular_ball,
        )
    )
    granular_score = weighted_total_mi(granular_band_mi, config.frequency)

    rng = np.random.default_rng(config.lcb.random_state + int(layer_id) * 1009)
    repeated_scores = []
    repeated_band_scores = []
    for _repeat in range(config.lcb.repeats):
        if (
            config.lcb.cluster_by_base_sample
            and base_ids is not None
            and scenarios is not None
        ):
            indices = dual_source_bootstrap_indices(
                y,
                base_ids,
                scenarios,
                config.lcb.sample_fraction,
                config.lcb.scenario_fraction,
                rng,
            )
        else:
            stratify_scenarios = (
                scenarios if config.lcb.stratify_by_scenario else None
            )
            indices = stratified_bootstrap_indices(
                y,
                config.lcb.sample_fraction,
                rng,
                stratify_scenarios,
            )

        # Sample-wise normalization and DCT were already completed. The task-
        # related band merge and all granular balls are rebuilt each repeat.
        repeat_spectrum = build_spectrum_from_fine_energy(
            spectrum.fine_energy[indices],
            y[indices],
            config.frequency,
        )
        repeat_local_mi, _repeat_kde, _ = multi_granularity_local_mi(
            repeat_spectrum.band_energy,
            y[indices],
            config.frequency,
            config.granular_ball,
            compute_kde=False,
        )
        repeated_band_scores.append(repeat_local_mi)
        repeated_scores.append(weighted_total_mi(repeat_local_mi, config.frequency))

    bootstrap = np.stack(repeated_scores, axis=0)
    bootstrap_bands = np.stack(repeated_band_scores, axis=0)
    mean = bootstrap.mean(axis=0)
    std = bootstrap.std(axis=0, ddof=1)
    lcb = mean - config.lcb.lcb_lambda * std
    band_mean = bootstrap_bands.mean(axis=0)
    band_std = bootstrap_bands.std(axis=0, ddof=1)

    return LayerAblationScores(
        global_spectrum=spectrum,
        granular_band_mi=granular_band_mi,
        granular_band_mi_kde=granular_band_mi_kde,
        granularities=granularities,
        mi_score=spectrum.total_mi,
        granular_score=granular_score,
        bootstrap_scores=bootstrap,
        bootstrap_band_scores=bootstrap_bands,
        lcb_mean=mean,
        lcb_std=std,
        lcb_score=lcb,
        lcb_band_mean=band_mean,
        lcb_band_std=band_std,
    )
