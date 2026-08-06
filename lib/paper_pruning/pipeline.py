from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

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


def _bootstrap_index_sets(
    y: np.ndarray,
    scenarios: Optional[np.ndarray],
    base_ids: Optional[np.ndarray],
    config: PipelineConfig,
    layer_id: int,
) -> List[np.ndarray]:
    """Generate all repeated samples serially so parallel execution preserves
    the exact random sequence and therefore the same LCB experiment.
    """
    rng = np.random.default_rng(config.lcb.random_state + int(layer_id) * 1009)
    index_sets: List[np.ndarray] = []
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
        index_sets.append(np.asarray(indices, dtype=np.int64))
    return index_sets


def score_layer(
    responses: np.ndarray,
    events: np.ndarray,
    scenario_ids: Optional[np.ndarray],
    config: PipelineConfig,
    layer_id: int = 0,
    base_sample_ids: Optional[np.ndarray] = None,
) -> LayerAblationScores:
    """Compute MI, MI+granular-ball and MI+granular-ball+LCB.

    The paper's execution order is unchanged. Every LCB repeat still rebuilds
    task-driven frequency bands, per-unit granular balls and local MI. The fast
    implementation only batches algebraically independent calculations, reuses
    identical balls across nested granularities and optionally evaluates LCB
    repeats concurrently.
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

    index_sets = _bootstrap_index_sets(y, scenarios, base_ids, config, layer_id)
    lcb_workers = min(max(1, config.lcb.workers), config.lcb.repeats)
    # Keep total nested concurrency close to --paper_gb_workers. For example,
    # lcb_workers=4 and gb_workers=16 gives four repeats with four unit workers.
    repeat_gb_workers = max(1, config.granular_ball.workers // lcb_workers)
    repeat_gb_cfg = replace(config.granular_ball, workers=repeat_gb_workers)

    def score_repeat(item: Tuple[int, np.ndarray]):
        repeat_id, indices = item
        repeat_spectrum = build_spectrum_from_fine_energy(
            spectrum.fine_energy[indices],
            y[indices],
            config.frequency,
        )
        repeat_local_mi, _repeat_kde, _details = multi_granularity_local_mi(
            repeat_spectrum.band_energy,
            y[indices],
            config.frequency,
            repeat_gb_cfg,
            compute_kde=False,
            kde_unit_indices=np.empty(0, dtype=np.int64),
            build_details=False,
        )
        repeat_score = weighted_total_mi(repeat_local_mi, config.frequency)
        return repeat_id, repeat_score, repeat_local_mi

    items = list(enumerate(index_sets))
    if lcb_workers == 1:
        repeat_results = []
        for completed, item in enumerate(items, start=1):
            repeat_results.append(score_repeat(item))
            print(
                f"[paper LCB] repeat {completed}/{config.lcb.repeats}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=lcb_workers) as executor:
            repeat_results = []
            for completed, result in enumerate(executor.map(score_repeat, items), start=1):
                repeat_results.append(result)
                print(
                    f"[paper LCB] repeat {completed}/{config.lcb.repeats} "
                    f"(parallel={lcb_workers}, unit-workers={repeat_gb_workers})",
                    flush=True,
                )

    repeat_results.sort(key=lambda item: item[0])
    bootstrap = np.stack([item[1] for item in repeat_results], axis=0)
    bootstrap_bands = np.stack([item[2] for item in repeat_results], axis=0)
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
