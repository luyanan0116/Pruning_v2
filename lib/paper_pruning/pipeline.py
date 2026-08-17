from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import PipelineConfig
from .granular_ball import GranularityResult, multi_granularity_local_mi
from .mi import FrequencySpectrum, build_frequency_spectrum, weighted_total_mi
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


def _bootstrap_indices(
    events: np.ndarray,
    scenario_ids: Optional[np.ndarray],
    base_sample_ids: Optional[np.ndarray],
    config: PipelineConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one resample matching the proposal's sample/scenario uncertainty.

    When both base-sample and scenario identifiers are available, the default
    is a two-stage bootstrap: resample base calibration examples, then resample
    scenario realizations within every selected base example.  Otherwise fall
    back to event-stratified (optionally event+scenario stratified) bootstrap.
    """
    lcb_cfg = config.lcb
    if (
        lcb_cfg.cluster_by_base_sample
        and base_sample_ids is not None
        and scenario_ids is not None
    ):
        return dual_source_bootstrap_indices(
            events,
            base_sample_ids,
            scenario_ids,
            sample_fraction=lcb_cfg.sample_fraction,
            scenario_fraction=lcb_cfg.scenario_fraction,
            rng=rng,
        )
    stratify_scenarios = (
        scenario_ids if lcb_cfg.stratify_by_scenario and scenario_ids is not None else None
    )
    return stratified_bootstrap_indices(
        events,
        lcb_cfg.sample_fraction,
        rng,
        scenario_ids=stratify_scenarios,
    )


def score_layer(
    responses: np.ndarray,
    events: np.ndarray,
    scenario_ids: Optional[np.ndarray],
    config: PipelineConfig,
    layer_id: int = 0,
    base_sample_ids: Optional[np.ndarray] = None,
) -> LayerAblationScores:
    """Compute MI, granular-ball score, and repeated-estimation LCB.

    The full-data frequency decomposition fixes the task-driven band boundaries
    once.  LCB repeats then resample observations and recompute the granular
    local-MI estimator on those same bands.  This gives comparable per-band
    bootstrap samples while still reflecting both calibration-sample and
    scenario uncertainty.
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

    repeats = int(config.lcb.repeats)
    if repeats == 1:
        # Compatibility/debug mode: one full-data estimate has no empirical
        # variance, therefore LCB equals the granular score by definition.
        bootstrap = granular_score[None, :].copy()
        bootstrap_bands = granular_band_mi[None, :, :].copy()
    else:
        bootstrap = np.empty((repeats, granular_score.size), dtype=np.float64)
        bootstrap_bands = np.empty(
            (repeats, granular_band_mi.shape[0], granular_band_mi.shape[1]),
            dtype=np.float64,
        )
        rng = np.random.default_rng(config.lcb.random_state + 1009 * int(layer_id))
        for repeat in range(repeats):
            indices = _bootstrap_indices(y, scenarios, base_ids, config, rng)
            sampled_bands, _sampled_kde, _ = multi_granularity_local_mi(
                spectrum.band_energy[indices],
                y[indices],
                config.frequency,
                config.granular_ball,
                compute_kde=False,
                kde_unit_indices=np.empty(0, dtype=np.int64),
                build_details=False,
            )
            bootstrap_bands[repeat] = sampled_bands
            bootstrap[repeat] = weighted_total_mi(sampled_bands, config.frequency)
            if repeat == 0 or (repeat + 1) == repeats or (repeat + 1) % 5 == 0:
                print(
                    f"[paper LCB] layer-key={layer_id} repeat {repeat + 1}/{repeats} "
                    f"n={indices.size}",
                    flush=True,
                )

    # Population std (ddof=0) is deliberately used because these repeated
    # estimates are an empirical uncertainty distribution rather than an
    # unbiased estimator of a separate sample variance parameter.
    mean = bootstrap.mean(axis=0)
    std = bootstrap.std(axis=0, ddof=0)
    lcb = mean - config.lcb.lcb_lambda * std
    band_mean = bootstrap_bands.mean(axis=0)
    band_std = bootstrap_bands.std(axis=0, ddof=0)

    print(
        f"[paper LCB] repeats={repeats}, sample_fraction={config.lcb.sample_fraction:.3f}, "
        f"scenario_fraction={config.lcb.scenario_fraction:.3f}, "
        f"lambda={config.lcb.lcb_lambda:.3f}, mean_std={std.mean():.6g}",
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
