from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class FrequencyConfig:
    """Paper equations (1)-(5): normalization, DCT, frequency energy and MI."""

    fine_bins: int = 16
    target_bands: int = 4
    mi_neighbors: int = 3
    knn_weight: float = 0.85
    kde_weight: float = 0.15
    band_weights: Optional[Tuple[float, ...]] = None
    eps: float = 1e-8
    random_state: int = 0

    def validate(self) -> None:
        if self.fine_bins < 2:
            raise ValueError("fine_bins must be >= 2")
        if not 1 <= self.target_bands <= self.fine_bins:
            raise ValueError("target_bands must be in [1, fine_bins]")
        if self.mi_neighbors < 1:
            raise ValueError("mi_neighbors must be >= 1")
        if self.knn_weight < 0 or self.kde_weight < 0 or self.knn_weight + self.kde_weight <= 0:
            raise ValueError("MI estimator weights must be non-negative with a positive sum")
        if self.band_weights is not None and len(self.band_weights) != self.target_bands:
            raise ValueError("band_weights length must equal target_bands")


@dataclass(frozen=True)
class GranularBallConfig:
    """Paper equations (6)-(8): purity/compactness splitting and fusion."""

    purity_thresholds: Tuple[float, ...] = (0.65, 0.75, 0.85)
    granularity_weights: Optional[Tuple[float, ...]] = None
    min_ball_size: int = 8
    max_balls: int = 64
    min_purity_gain: float = 0.01
    min_radius_reduction: float = 0.05
    compactness_ratio: float = 0.55
    min_event_classes: int = 2
    kmeans_iterations: int = 12
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
        if not 0 < self.compactness_ratio <= 1:
            raise ValueError("compactness_ratio must be in (0, 1]")
        if self.min_event_classes < 1:
            raise ValueError("min_event_classes must be >= 1")


@dataclass(frozen=True)
class LCBConfig:
    """Paper equations (9)-(10): repeated estimation and lower confidence bound."""

    repeats: int = 20
    sample_fraction: float = 0.8
    lcb_lambda: float = 1.0
    stratify_by_scenario: bool = True
    random_state: int = 0

    def validate(self) -> None:
        if self.repeats < 2:
            raise ValueError("repeats must be >= 2")
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be in (0, 1]")
        if self.lcb_lambda < 0:
            raise ValueError("lcb_lambda must be non-negative")


@dataclass(frozen=True)
class PipelineConfig:
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    granular_ball: GranularBallConfig = field(default_factory=GranularBallConfig)
    lcb: LCBConfig = field(default_factory=LCBConfig)

    def validate(self) -> None:
        self.frequency.validate()
        self.granular_ball.validate()
        self.lcb.validate()
