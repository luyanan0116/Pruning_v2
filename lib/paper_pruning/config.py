from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class FrequencyConfig:
    """Frequency-domain contribution spectrum configuration.

    This maps to the concise proposal equations (1)-(5) and the detailed
    proposal equations (4)-(15): sample-wise normalization, proposal-form
    DCT-II, squared frequency energy, fine bins, task-driven adjacent merging,
    band mutual information and weighted total contribution.
    """

    fine_bins: int = 16
    target_bands: int = 4
    mi_neighbors: int = 3
    probe_units: int = 64
    band_weights: Optional[Tuple[float, ...]] = None
    kde_bandwidth_scale: float = 1.0
    max_merge_loss: Optional[float] = None
    eps: float = 1e-8
    random_state: int = 0

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
        if self.max_merge_loss is not None and self.max_merge_loss < 0:
            raise ValueError("max_merge_loss must be non-negative or None")
        if self.band_weights is not None:
            if len(self.band_weights) != self.target_bands:
                raise ValueError("band_weights length must equal target_bands")
            if any(weight < 0 for weight in self.band_weights) or sum(self.band_weights) <= 0:
                raise ValueError("band_weights must be non-negative with positive sum")


@dataclass(frozen=True)
class GranularBallConfig:
    """Granular-ball localization and multi-granularity fusion.

    Maps to concise equations (6)-(8) and detailed equations (16)-(24).
    Every structural unit builds balls in its own B-dimensional band-response
    space, matching the displayed particle-vector definition.
    """

    purity_thresholds: Tuple[float, ...] = (0.65, 0.75, 0.85)
    granularity_weights: Optional[Tuple[float, ...]] = None
    granularity_weight_mode: str = "repeat_variance"
    min_ball_size: int = 8
    max_balls: int = 64
    max_depth: int = 5
    min_purity_gain: float = 0.0
    min_radius_reduction: float = 0.0
    compactness_ratio: float = 0.55
    min_event_classes: int = 2
    kmeans_iterations: int = 12
    workers: int = 1
    worker_chunk_size: int = 64
    kde_scope: str = "probe"
    variance_floor: float = 1e-8
    random_state: int = 0

    def validate(self) -> None:
        if not self.purity_thresholds:
            raise ValueError("purity_thresholds cannot be empty")
        if tuple(sorted(self.purity_thresholds)) != self.purity_thresholds:
            raise ValueError("purity_thresholds must be sorted from coarse to fine")
        if any(not 0.5 <= value < 1.0 for value in self.purity_thresholds):
            raise ValueError("each purity threshold must be in [0.5, 1.0)")
        if self.granularity_weight_mode not in {"equal", "manual", "repeat_variance"}:
            raise ValueError("granularity_weight_mode must be equal, manual or repeat_variance")
        if self.granularity_weight_mode == "manual" and self.granularity_weights is None:
            raise ValueError("manual granularity weighting requires granularity_weights")
        if self.granularity_weights is not None:
            if len(self.granularity_weights) != len(self.purity_thresholds):
                raise ValueError("granularity_weights length must match purity_thresholds")
            if any(value < 0 for value in self.granularity_weights) or sum(self.granularity_weights) <= 0:
                raise ValueError("granularity_weights must be non-negative with positive sum")
        if self.min_ball_size < 3:
            raise ValueError("min_ball_size must be >= 3")
        if self.max_balls < 1 or self.max_depth < 1:
            raise ValueError("max_balls and max_depth must be positive")
        if not 0 < self.compactness_ratio <= 1:
            raise ValueError("compactness_ratio must be in (0,1]")
        if self.min_event_classes < 1:
            raise ValueError("min_event_classes must be >= 1")
        if self.workers < 1 or self.worker_chunk_size < 1:
            raise ValueError("workers and worker_chunk_size must be positive")
        if self.kde_scope not in {"all", "probe", "none"}:
            raise ValueError("kde_scope must be all, probe or none")
        if self.variance_floor <= 0:
            raise ValueError("variance_floor must be positive")


@dataclass(frozen=True)
class LCBConfig:
    """Dual-source repeated estimation and lower-confidence-bound ranking."""

    repeats: int = 20
    sample_fraction: float = 0.8
    scenario_fraction: float = 1.0
    lcb_lambda: float = 1.0
    stratify_by_scenario: bool = True
    cluster_by_base_sample: bool = True
    random_state: int = 0
    workers: int = 1

    def validate(self) -> None:
        if self.repeats < 2:
            raise ValueError("repeats must be >= 2")
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be in (0,1]")
        if not 0 < self.scenario_fraction <= 1:
            raise ValueError("scenario_fraction must be in (0,1]")
        if self.lcb_lambda < 0:
            raise ValueError("lcb_lambda must be non-negative")
        if self.workers < 1:
            raise ValueError("workers must be positive")


@dataclass(frozen=True)
class BudgetConfig:
    """Frequency coverage and greedy deployment-budget configuration."""

    coverage_ratio: float = 0.90
    coverage_alpha: float = 0.25
    fill_budget: bool = True

    def validate(self) -> None:
        if not 0 < self.coverage_ratio <= 1:
            raise ValueError("coverage_ratio must be in (0,1]")
        if self.coverage_alpha < 0:
            raise ValueError("coverage_alpha must be non-negative")


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
