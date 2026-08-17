from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class FrequencyConfig:
    """Paper equations (1)-(5) / detailed equations (4)-(15).

    The kNN estimator is the primary estimator used for ranking. KDE is kept as
    an auxiliary diagnostic estimator, matching the proposal's "one primary,
    one auxiliary" design rather than averaging two estimators into one score.
    """

    fine_bins: int = 16
    target_bands: int = 4
    mi_neighbors: int = 3
    probe_units: int = 64
    band_weights: Optional[Tuple[float, ...]] = None
    kde_bandwidth_scale: float = 1.0
    eps: float = 1e-8
    random_state: int = 0
    fast_small_mi: bool = True
    fast_small_mi_max_features: int = 16
    fast_small_mi_max_samples: int = 512

    def validate(self) -> None:
        if self.fine_bins < 2:
            raise ValueError("fine_bins must be >= 2")
        if not 1 <= self.target_bands <= self.fine_bins:
            raise ValueError("target_bands must be in [1, fine_bins]")
        if self.mi_neighbors < 1:
            raise ValueError("mi_neighbors must be >= 1")
        if self.probe_units < 1:
            raise ValueError("probe_units must be >= 1")
        if self.kde_bandwidth_scale <= 0:
            raise ValueError("kde_bandwidth_scale must be positive")
        if self.fast_small_mi_max_features < 1:
            raise ValueError("fast_small_mi_max_features must be >= 1")
        if self.fast_small_mi_max_samples < 4:
            raise ValueError("fast_small_mi_max_samples must be >= 4")
        if self.band_weights is not None and len(self.band_weights) != self.target_bands:
            raise ValueError("band_weights length must equal target_bands")


@dataclass(frozen=True)
class GranularBallConfig:
    """Paper equations (6)-(8) / detailed equations (16)-(24)."""

    purity_thresholds: Tuple[float, ...] = (0.65, 0.75, 0.85)
    granularity_weights: Optional[Tuple[float, ...]] = None
    min_ball_size: int = 8
    max_balls: int = 64
    max_depth: int = 5
    min_purity_gain: float = 0.0
    min_radius_reduction: float = 0.0
    compactness_ratio: float = 0.55
    min_event_classes: int = 2
    min_event_count_per_class: int = 2
    kmeans_iterations: int = 12
    localization_mode: str = "layer_shared"
    workers: int = 1
    worker_chunk_size: int = 64
    kde_scope: str = "probe"
    variance_floor: float = 1e-8
    fusion_mode: str = "inverse_sqrt_dispersion"
    fusion_max_ratio: float = 5.0
    random_state: int = 0

    def validate(self) -> None:
        if not self.purity_thresholds:
            raise ValueError("purity_thresholds cannot be empty")
        if tuple(sorted(self.purity_thresholds)) != self.purity_thresholds:
            raise ValueError("purity_thresholds must be sorted from coarse to fine")
        if any(not 0.5 <= p < 1.0 for p in self.purity_thresholds):
            raise ValueError("each purity threshold must be in [0.5, 1.0)")
        if self.granularity_weights is not None and len(self.granularity_weights) != len(self.purity_thresholds):
            raise ValueError("granularity_weights length must match purity_thresholds")
        if self.min_ball_size < 3:
            raise ValueError("min_ball_size must be >= 3")
        if self.max_balls < 1:
            raise ValueError("max_balls must be >= 1")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if not 0 < self.compactness_ratio <= 1:
            raise ValueError("compactness_ratio must be in (0, 1]")
        if self.min_event_classes < 1:
            raise ValueError("min_event_classes must be >= 1")
        if self.min_event_count_per_class < 1:
            raise ValueError("min_event_count_per_class must be >= 1")
        if self.localization_mode not in {"layer_shared", "unit_local"}:
            raise ValueError("localization_mode must be layer_shared or unit_local")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.worker_chunk_size < 1:
            raise ValueError("worker_chunk_size must be >= 1")
        if self.kde_scope not in {"all", "probe", "none"}:
            raise ValueError("kde_scope must be all, probe or none")
        if self.variance_floor <= 0:
            raise ValueError("variance_floor must be positive")
        if self.fusion_mode not in {"equal", "inverse_sqrt_dispersion", "inverse_dispersion"}:
            raise ValueError("fusion_mode must be equal, inverse_sqrt_dispersion or inverse_dispersion")
        if self.fusion_max_ratio < 1:
            raise ValueError("fusion_max_ratio must be >= 1")


@dataclass(frozen=True)
class LCBConfig:
    """Paper equations (9)-(10) / detailed equations (25)-(27)."""

    repeats: int = 10
    sample_fraction: float = 0.8
    scenario_fraction: float = 0.67
    lcb_lambda: float = 1.0
    stratify_by_scenario: bool = True
    cluster_by_base_sample: bool = True
    random_state: int = 0
    workers: int = 1

    def validate(self) -> None:
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be in (0, 1]")
        if not 0 < self.scenario_fraction <= 1:
            raise ValueError("scenario_fraction must be in (0, 1]")
        if self.lcb_lambda < 0:
            raise ValueError("lcb_lambda must be non-negative")
        if self.workers < 1:
            raise ValueError("LCB workers must be >= 1")


@dataclass(frozen=True)
class BudgetConfig:
    """Paper equations (11)-(13): coverage-aware budget configuration."""

    coverage_ratio: float = 0.90
    coverage_ratios: Optional[Tuple[float, ...]] = None
    band_selection_weights: Optional[Tuple[float, ...]] = None
    coverage_alpha: float = 0.25
    greedy_batches: int = 64

    def validate(self) -> None:
        if not 0 < self.coverage_ratio <= 1:
            raise ValueError("coverage_ratio must be in (0,1]")
        if self.coverage_ratios is not None:
            if not self.coverage_ratios:
                raise ValueError("coverage_ratios cannot be empty")
            if any(not 0 < value <= 1 for value in self.coverage_ratios):
                raise ValueError("every coverage ratio must be in (0,1]")
        if self.band_selection_weights is not None:
            if not self.band_selection_weights:
                raise ValueError("band_selection_weights cannot be empty")
            if any(value < 0 for value in self.band_selection_weights):
                raise ValueError("band_selection_weights must be non-negative")
            if not any(value > 0 for value in self.band_selection_weights):
                raise ValueError("at least one band_selection_weight must be positive")
        if self.coverage_alpha < 0:
            raise ValueError("coverage_alpha must be non-negative")
        if self.greedy_batches < 1:
            raise ValueError("greedy_batches must be >= 1")


@dataclass(frozen=True)
class PipelineConfig:
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    granular_ball: GranularBallConfig = field(default_factory=GranularBallConfig)
    lcb: LCBConfig = field(default_factory=LCBConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    def validate(self) -> None:
        self.frequency.validate()
        self.granular_ball.validate()
        self.lcb.validate()
        self.budget.validate()
