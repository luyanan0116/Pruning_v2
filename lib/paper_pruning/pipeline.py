from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import PipelineConfig
from .granular_ball import GranularityResult, multi_granularity_local_mi
from .mi import FrequencySpectrum, build_frequency_spectrum, weighted_total_mi


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
    """Compute MI and MI+granular-ball once.

    This build intentionally disables repeated LCB estimation. The single
    granular-ball contribution is reused as the LCB mean, standard deviation
    is fixed to zero, and therefore LCB equals the one-pass contribution score.
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
    if config.granular_ball.kde_scope == "probe":
        kde_units: Optional[np.ndarray] = spectrum.probe_indices
    elif config.granular_ball.kde_scope == "none":
        kde_units = np.empty(0, dtype=np.int64)
    else:
        kde_units = None

    granular_band_mi, granular_band_mi_kde, granularities = (
        multi_granularity_local_mi(
            spectrum.band_energy,
            y,
            config.frequency,
            config.granular_ball,
            compute_kde=config.granular_ball.kde_scope != "none",
            kde_unit_indices=kde_units,
            build_details=True,
        )
    )
    granular_score = weighted_total_mi(granular_band_mi, config.frequency)

    # Single-pass mode requested by the experiment:
    # no bootstrap/resampling loop and no repeated granular-ball/MI computation.
    # The one granular-ball estimate computed above is used directly.
    bootstrap = granular_score[None, :].copy()
    bootstrap_bands = granular_band_mi[None, :, :].copy()

    mean = granular_score.copy()
    std = np.zeros_like(mean)
    lcb = mean - config.lcb.lcb_lambda * std

    band_mean = granular_band_mi.copy()
    band_std = np.zeros_like(band_mean)

    print(
        "[paper LCB] single-pass mode: repeated estimation disabled; "
        "std=0, so LCB equals the single granular-ball contribution score",
        flush=True,
    )

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
