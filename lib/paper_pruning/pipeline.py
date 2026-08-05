from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .granular_ball import (
    GranularityResult,
    configured_granularity_weights,
    fuse_granularity_levels,
    granularity_weights_from_repeat_variance,
    multi_granularity_level_mi,
)
from .mi import (
    FrequencySpectrum,
    build_spectrum_from_fine_energy,
    dct_frequency_energy,
    normalized_band_weights,
    weighted_total_mi,
)
from .resampling import dual_source_bootstrap_indices, stratified_bootstrap_indices


SCORING_METHODS = ("paper_mi", "paper_mi_gb", "paper_mi_gb_lcb")


@dataclass
class LayerAblationScores:
    """Scores for exactly one requested ablation stage.

    Unexecuted later-stage outputs are represented by empty repeat arrays and
    NaN score arrays. ``selected_score`` and ``selected_band_score`` are always
    the values used to generate the structural pruning configuration.
    """

    method: str
    executed_stages: Tuple[str, ...]
    global_spectrum: FrequencySpectrum
    granular_band_mi: np.ndarray
    granular_band_mi_kde: np.ndarray
    granularities: List[GranularityResult]
    granularity_weights: np.ndarray
    granularity_variances: np.ndarray
    mi_score: np.ndarray
    granular_score: np.ndarray
    bootstrap_scores: np.ndarray
    bootstrap_band_scores: np.ndarray
    bootstrap_level_band_scores: np.ndarray
    lcb_mean: np.ndarray
    lcb_std: np.ndarray
    lcb_score: np.ndarray
    lcb_band_mean: np.ndarray
    lcb_band_std: np.ndarray
    selected_score: np.ndarray
    selected_band_score: np.ndarray


def _bootstrap_index_sets(
    y: np.ndarray,
    scenarios: Optional[np.ndarray],
    base_ids: Optional[np.ndarray],
    config: PipelineConfig,
    layer_id: int,
) -> List[np.ndarray]:
    """Generate paired sample/scenario repeated-estimate index sets."""
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
            stratify_scenarios = scenarios if config.lcb.stratify_by_scenario else None
            indices = stratified_bootstrap_indices(
                y,
                config.lcb.sample_fraction,
                rng,
                stratify_scenarios,
            )
        index_sets.append(np.asarray(indices, dtype=np.int64))
    return index_sets


def _weighted_scores(band_scores: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Vectorized weighted band aggregation for one or more leading dimensions."""
    weights = normalized_band_weights(band_scores.shape[-1], config.frequency)
    return np.tensordot(np.asarray(band_scores, dtype=np.float64), weights, axes=([-1], [0]))


def _repeat_level_estimates(
    spectrum: FrequencySpectrum,
    y: np.ndarray,
    scenarios: Optional[np.ndarray],
    base_ids: Optional[np.ndarray],
    config: PipelineConfig,
    layer_id: int,
) -> np.ndarray:
    """Repeat the full band-building and granular-ball estimation flow."""
    index_sets = _bootstrap_index_sets(y, scenarios, base_ids, config, layer_id)
    lcb_workers = min(max(1, config.lcb.workers), config.lcb.repeats)
    repeat_gb_workers = max(1, config.granular_ball.workers // lcb_workers)
    repeat_gb_cfg = config.granular_ball.__class__(
        **{
            **config.granular_ball.__dict__,
            "workers": repeat_gb_workers,
            "kde_scope": "none",
        }
    )

    def score_repeat(item: Tuple[int, np.ndarray]):
        repeat_id, indices = item
        repeat_spectrum = build_spectrum_from_fine_energy(
            spectrum.fine_energy[indices],
            y[indices],
            config.frequency,
        )
        repeat_levels, _repeat_kde, _details = multi_granularity_level_mi(
            repeat_spectrum.band_energy,
            y[indices],
            config.frequency,
            repeat_gb_cfg,
            compute_kde=False,
            kde_unit_indices=np.empty(0, dtype=np.int64),
            build_details=False,
        )
        return repeat_id, repeat_levels

    items = list(enumerate(index_sets))
    if lcb_workers == 1:
        repeat_results = []
        for completed, item in enumerate(items, start=1):
            repeat_results.append(score_repeat(item))
            print(f"[paper repeated estimate] {completed}/{config.lcb.repeats}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=lcb_workers) as executor:
            repeat_results = []
            for completed, result in enumerate(executor.map(score_repeat, items), start=1):
                repeat_results.append(result)
                print(
                    f"[paper repeated estimate] {completed}/{config.lcb.repeats} "
                    f"(parallel={lcb_workers}, unit-workers={repeat_gb_workers})",
                    flush=True,
                )
    repeat_results.sort(key=lambda item: item[0])
    return np.stack([item[1] for item in repeat_results], axis=0)


def _blank_outputs(unit_count: int, band_count: int, level_count: int):
    nan_units = np.full(unit_count, np.nan, dtype=np.float64)
    nan_bands = np.full((unit_count, band_count), np.nan, dtype=np.float64)
    return {
        "granular_band_mi": nan_bands.copy(),
        "granular_band_mi_kde": nan_bands.copy(),
        "granularity_weights": np.full(level_count, np.nan, dtype=np.float64),
        "granularity_variances": np.full(level_count, np.nan, dtype=np.float64),
        "granular_score": nan_units.copy(),
        "bootstrap_scores": np.empty((0, unit_count), dtype=np.float64),
        "bootstrap_band_scores": np.empty((0, unit_count, band_count), dtype=np.float64),
        "bootstrap_level_band_scores": np.empty(
            (0, level_count, unit_count, band_count), dtype=np.float64
        ),
        "lcb_mean": nan_units.copy(),
        "lcb_std": nan_units.copy(),
        "lcb_score": nan_units.copy(),
        "lcb_band_mean": nan_bands.copy(),
        "lcb_band_std": nan_bands.copy(),
    }


def score_layer_from_fine_energy(
    fine_energy: np.ndarray,
    events: np.ndarray,
    scenario_ids: Optional[np.ndarray],
    config: PipelineConfig,
    layer_id: int = 0,
    base_sample_ids: Optional[np.ndarray] = None,
    method: str = "paper_mi_gb_lcb",
) -> LayerAblationScores:
    """Run exactly the requested incremental ablation stage.

    ``paper_mi`` executes only the frequency-domain mutual-information flow.
    ``paper_mi_gb`` additionally executes granular-ball localization and
    multi-granularity fusion. With deterministic fusion weights it does not run
    repeated estimation. ``paper_mi_gb_lcb`` additionally performs paired
    sample/scenario repeated estimation and LCB ranking.
    """
    if method not in SCORING_METHODS:
        raise ValueError(f"unsupported scoring method: {method}")
    config.validate()
    x = np.asarray(fine_energy)
    y = np.asarray(events)
    scenarios = None if scenario_ids is None else np.asarray(scenario_ids)
    base_ids = None if base_sample_ids is None else np.asarray(base_sample_ids)
    if x.ndim != 3:
        raise ValueError("fine_energy must be [samples,units,fine_bins]")
    if x.shape[0] != y.size:
        raise ValueError("fine-energy/event sample counts differ")
    if scenarios is not None and scenarios.size != y.size:
        raise ValueError("scenario_ids must match events")
    if base_ids is not None and base_ids.size != y.size:
        raise ValueError("base_sample_ids must match events")

    spectrum = build_spectrum_from_fine_energy(x, y, config.frequency)
    unit_count = x.shape[1]
    band_count = config.frequency.target_bands
    level_count = len(config.granular_ball.purity_thresholds)
    blank = _blank_outputs(unit_count, band_count, level_count)

    if method == "paper_mi":
        return LayerAblationScores(
            method=method,
            executed_stages=("mutual_information",),
            global_spectrum=spectrum,
            granularities=[],
            mi_score=spectrum.total_mi,
            selected_score=spectrum.total_mi,
            selected_band_score=spectrum.band_mi,
            **blank,
        )

    if config.granular_ball.kde_scope == "probe":
        kde_units: Optional[np.ndarray] = spectrum.probe_indices
    elif config.granular_ball.kde_scope == "none":
        kde_units = np.empty(0, dtype=np.int64)
    else:
        kde_units = None

    base_levels, base_kde_levels, granularities = multi_granularity_level_mi(
        spectrum.band_energy,
        y,
        config.frequency,
        config.granular_ball,
        compute_kde=config.granular_ball.kde_scope != "none",
        kde_unit_indices=kde_units,
        build_details=True,
    )

    need_repeats = (
        method == "paper_mi_gb_lcb"
        or config.granular_ball.granularity_weight_mode == "repeat_variance"
    )
    repeat_level_bands = (
        _repeat_level_estimates(spectrum, y, scenarios, base_ids, config, layer_id)
        if need_repeats
        else blank["bootstrap_level_band_scores"]
    )

    if config.granular_ball.granularity_weight_mode == "repeat_variance":
        fusion_weights, fusion_variances = granularity_weights_from_repeat_variance(
            repeat_level_bands,
            config.granular_ball,
        )
        fusion_stage = "repeat_variance_granularity_fusion"
    else:
        fusion_weights = configured_granularity_weights(config.granular_ball)
        if repeat_level_bands.shape[0] >= 2:
            _unused, fusion_variances = granularity_weights_from_repeat_variance(
                repeat_level_bands,
                config.granular_ball.__class__(
                    **{
                        **config.granular_ball.__dict__,
                        "granularity_weights": None,
                        "granularity_weight_mode": "repeat_variance",
                    }
                ),
            )
        else:
            fusion_variances = np.full(level_count, np.nan, dtype=np.float64)
        fusion_stage = f"{config.granular_ball.granularity_weight_mode}_granularity_fusion"

    granular_band_mi = fuse_granularity_levels(base_levels, fusion_weights)
    if np.isnan(base_kde_levels).all():
        granular_band_mi_kde = np.full_like(granular_band_mi, np.nan)
    else:
        granular_band_mi_kde = np.tensordot(
            fusion_weights,
            np.nan_to_num(base_kde_levels, nan=0.0),
            axes=(0, 0),
        )
        available = np.any(np.isfinite(base_kde_levels), axis=(0, 2))
        granular_band_mi_kde[~available] = np.nan

    for level, details in enumerate(granularities):
        details.fusion_weight = float(fusion_weights[level])
        details.variance = float(fusion_variances[level])

    granular_score = weighted_total_mi(granular_band_mi, config.frequency)

    if repeat_level_bands.shape[0] > 0:
        bootstrap_bands = np.einsum(
            "m,rmub->rub",
            fusion_weights,
            repeat_level_bands,
            optimize=True,
        )
        bootstrap = _weighted_scores(bootstrap_bands, config)
    else:
        bootstrap_bands = blank["bootstrap_band_scores"]
        bootstrap = blank["bootstrap_scores"]

    if method == "paper_mi_gb":
        stages = ["mutual_information", "granular_ball", "multi_granularity_fusion"]
        if need_repeats:
            stages.append(fusion_stage)
        return LayerAblationScores(
            method=method,
            executed_stages=tuple(stages),
            global_spectrum=spectrum,
            granular_band_mi=granular_band_mi,
            granular_band_mi_kde=granular_band_mi_kde,
            granularities=granularities,
            granularity_weights=fusion_weights,
            granularity_variances=fusion_variances,
            mi_score=spectrum.total_mi,
            granular_score=granular_score,
            bootstrap_scores=bootstrap,
            bootstrap_band_scores=bootstrap_bands,
            bootstrap_level_band_scores=repeat_level_bands,
            lcb_mean=blank["lcb_mean"],
            lcb_std=blank["lcb_std"],
            lcb_score=blank["lcb_score"],
            lcb_band_mean=blank["lcb_band_mean"],
            lcb_band_std=blank["lcb_band_std"],
            selected_score=granular_score,
            selected_band_score=granular_band_mi,
        )

    mean = bootstrap.mean(axis=0)
    std = bootstrap.std(axis=0, ddof=1)
    lcb = mean - config.lcb.lcb_lambda * std
    band_mean = bootstrap_bands.mean(axis=0)
    band_std = bootstrap_bands.std(axis=0, ddof=1)
    robust_bands = band_mean - config.lcb.lcb_lambda * band_std

    return LayerAblationScores(
        method=method,
        executed_stages=(
            "mutual_information",
            "granular_ball",
            "multi_granularity_fusion",
            fusion_stage,
            "dual_source_repeated_estimation",
            "lcb",
        ),
        global_spectrum=spectrum,
        granular_band_mi=granular_band_mi,
        granular_band_mi_kde=granular_band_mi_kde,
        granularities=granularities,
        granularity_weights=fusion_weights,
        granularity_variances=fusion_variances,
        mi_score=spectrum.total_mi,
        granular_score=granular_score,
        bootstrap_scores=bootstrap,
        bootstrap_band_scores=bootstrap_bands,
        bootstrap_level_band_scores=repeat_level_bands,
        lcb_mean=mean,
        lcb_std=std,
        lcb_score=lcb,
        lcb_band_mean=band_mean,
        lcb_band_std=band_std,
        selected_score=lcb,
        selected_band_score=robust_bands,
    )


def score_layer(
    responses: np.ndarray,
    events: np.ndarray,
    scenario_ids: Optional[np.ndarray],
    config: PipelineConfig,
    layer_id: int = 0,
    base_sample_ids: Optional[np.ndarray] = None,
    method: str = "paper_mi_gb_lcb",
) -> LayerAblationScores:
    """Convenience path for fixed-length in-memory response sequences."""
    fine_energy = dct_frequency_energy(np.asarray(responses), config.frequency)
    return score_layer_from_fine_energy(
        fine_energy,
        events,
        scenario_ids,
        config,
        layer_id=layer_id,
        base_sample_ids=base_sample_ids,
        method=method,
    )
