from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .config import BudgetConfig


COST_METRICS = ("params", "flops", "memory", "kv_cache", "latency")


@dataclass(frozen=True)
class CostComponents:
    params: float
    flops: float
    memory: float
    kv_cache: float
    latency: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "params": float(self.params),
            "flops": float(self.flops),
            "memory": float(self.memory),
            "kv_cache": float(self.kv_cache),
            "latency": float(self.latency),
        }


def _cfg_value(config, name: str, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def unit_cost_components(
    model_config,
    unit_type: str,
    sequence_length: int,
    dtype_bytes: int = 2,
    latency: float = 0.0,
) -> CostComponents:
    """Return the deployment cost of one complete structural unit.

    For an FFN channel, the removable tensors are one row from each of the
    gate/up projections and the matching column of the down projection. For an
    attention head, the removable tensors are the matching q/k/v rows and the
    matching output-projection columns. FLOPs are analytic forward-pass counts;
    memory includes parameter bytes and, for attention, KV-cache bytes. Latency
    is accepted only from a measured profile and is never fabricated.
    """
    hidden = int(_cfg_value(model_config, "hidden_size", 0) or 0)
    heads = int(_cfg_value(model_config, "num_attention_heads", 0) or 0)
    if hidden <= 0 or heads <= 0 or hidden % heads:
        raise ValueError("model config must expose compatible hidden_size and num_attention_heads")
    head_dim = hidden // heads
    sequence = max(1, int(sequence_length))
    value_bytes = max(1, int(dtype_bytes))

    if unit_type == "mlp":
        params = float(3 * hidden + 2)
        flops = float(2 * 3 * hidden * sequence)
        kv_cache = 0.0
    elif unit_type == "attention":
        params = float(4 * hidden * head_dim + 3 * head_dim)
        projection_flops = 2.0 * 4 * hidden * head_dim * sequence
        attention_flops = 4.0 * sequence * sequence * head_dim
        flops = float(projection_flops + attention_flops)
        kv_cache = float(2 * sequence * head_dim * value_bytes)
    else:
        raise ValueError(f"unknown structural unit type: {unit_type}")

    return CostComponents(
        params=params,
        flops=flops,
        memory=float(params * value_bytes + kv_cache),
        kv_cache=kv_cache,
        latency=float(latency),
    )


def build_cost_matrix(
    components: Sequence[CostComponents],
    metrics: Sequence[str],
) -> np.ndarray:
    metrics = tuple(str(metric).strip().lower() for metric in metrics)
    if not metrics:
        raise ValueError("at least one budget metric is required")
    if any(metric not in COST_METRICS for metric in metrics):
        raise ValueError(f"budget metrics must be chosen from {COST_METRICS}")
    matrix = np.asarray(
        [[item.as_dict()[metric] for metric in metrics] for item in components],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("costs must be finite and non-negative")
    for column, metric in enumerate(metrics):
        if np.all(matrix[:, column] <= 0):
            raise ValueError(f"metric {metric} has no positive costs; provide a measured profile or remove it")
    return matrix


def budget_limits_from_keep_ratios(
    cost_matrix: np.ndarray,
    keep_ratios: Sequence[float],
) -> np.ndarray:
    matrix = np.asarray(cost_matrix, dtype=np.float64)
    ratios = np.asarray(tuple(float(value) for value in keep_ratios), dtype=np.float64)
    if matrix.ndim != 2 or ratios.shape != (matrix.shape[1],):
        raise ValueError("one keep ratio is required for each budget metric")
    if np.any(ratios <= 0) or np.any(ratios > 1):
        raise ValueError("keep ratios must be in (0,1]")
    return matrix.sum(axis=0) * ratios


def normalized_effective_cost(
    cost_matrix: np.ndarray,
    budget_limits: np.ndarray,
    metric_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Scalar cost used in the marginal-gain formula under several budgets.

    Each resource constraint remains a separate hard constraint. The scalar
    denominator is the weighted sum of normalized resource consumption,
    ``sum_j a_j c_ij / C_j``. With one active metric this is exactly the scalar
    cost in the proposal; with several metrics it is its dimensionless
    multi-budget extension.
    """
    costs = np.asarray(cost_matrix, dtype=np.float64)
    limits = np.asarray(budget_limits, dtype=np.float64).reshape(-1)
    if metric_weights is None:
        weights = np.ones(costs.shape[1], dtype=np.float64)
    else:
        weights = np.asarray(tuple(float(value) for value in metric_weights), dtype=np.float64)
        if weights.shape != (costs.shape[1],):
            raise ValueError("metric_weights must contain one value per active metric")
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("metric_weights must be non-negative with positive sum")
    weights = weights / weights.sum()
    normalized = costs / np.maximum(limits[None, :], 1e-12)
    return np.maximum(normalized @ weights, 1e-12)


def coverage_aware_multi_budget_indices(
    scores: np.ndarray,
    band_contribution: np.ndarray,
    cost_matrix: np.ndarray,
    budget_limits: np.ndarray,
    cfg: BudgetConfig,
    metric_weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Exact one-unit-at-a-time greedy selection with lazy recomputation.

    At a current keep set K, the implementation uses

      gain(i|K) = LCB_i/cost_i
                  + alpha * sum_b under_b(K) * contribution_i,b / cost_i.

    The under-coverage weights only decrease as units are selected, so stale
    heap values are valid upper bounds. Lazy recomputation therefore returns
    the same next unit as a full scan while avoiding a quadratic full-rescore.
    All deployment budgets are checked independently before a unit can enter K.
    """
    cfg.validate()
    raw_score = np.nan_to_num(
        np.asarray(scores, dtype=np.float64).reshape(-1),
        nan=-np.inf,
        neginf=-np.inf,
        posinf=np.finfo(np.float64).max,
    )
    bands = np.maximum(
        np.nan_to_num(np.asarray(band_contribution, dtype=np.float64), nan=0.0),
        0.0,
    )
    costs = np.asarray(cost_matrix, dtype=np.float64)
    limits = np.asarray(budget_limits, dtype=np.float64).reshape(-1)
    if bands.ndim != 2 or bands.shape[0] != raw_score.size:
        raise ValueError("band_contribution must be [units,bands]")
    if costs.ndim != 2 or costs.shape[0] != raw_score.size:
        raise ValueError("cost_matrix must be [units,metrics]")
    if limits.shape != (costs.shape[1],) or np.any(limits <= 0):
        raise ValueError("budget_limits must contain one positive value per metric")
    if np.any(costs < 0) or np.any(~np.isfinite(costs)):
        raise ValueError("cost_matrix must be finite and non-negative")

    effective_cost = normalized_effective_cost(costs, limits, metric_weights)
    band_totals = bands.sum(axis=0)
    band_targets = cfg.coverage_ratio * band_totals
    selected = np.zeros(raw_score.size, dtype=bool)
    spent = np.zeros(costs.shape[1], dtype=np.float64)
    coverage = np.zeros(bands.shape[1], dtype=np.float64)
    version = 0

    def under_coverage() -> np.ndarray:
        return np.maximum(0.0, band_targets - coverage)

    def marginal(index: int, under: np.ndarray) -> float:
        return float(
            (raw_score[index] + cfg.coverage_alpha * float(bands[index] @ under))
            / effective_cost[index]
        )

    initial_under = under_coverage()
    heap: list[tuple[float, int, int]] = [
        (-marginal(index, initial_under), index, version)
        for index in range(raw_score.size)
        if np.isfinite(raw_score[index])
    ]
    heapq.heapify(heap)

    while heap:
        negative_bound, index, evaluated_version = heapq.heappop(heap)
        if selected[index]:
            continue
        if np.any(spent + costs[index] > limits + 1e-12):
            continue

        current_under = under_coverage()
        current_gain = marginal(index, current_under)
        if evaluated_version != version:
            next_bound = -heap[0][0] if heap else -np.inf
            if current_gain + 1e-14 < next_bound:
                heapq.heappush(heap, (-current_gain, index, version))
                continue

        coverage_satisfied = bool(np.all(coverage + 1e-12 >= band_targets))
        if not cfg.fill_budget and coverage_satisfied and current_gain <= 0:
            break

        selected[index] = True
        spent += costs[index]
        coverage += bands[index]
        version += 1

    achieved = np.divide(
        coverage,
        np.maximum(band_totals, 1e-12),
        out=np.ones_like(coverage),
        where=band_totals > 0,
    )
    coverage_feasible = bool(np.all(coverage + 1e-12 >= band_targets))
    return np.flatnonzero(selected).astype(np.int64), spent, achieved, coverage_feasible
