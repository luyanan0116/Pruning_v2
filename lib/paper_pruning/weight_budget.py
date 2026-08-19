from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple
import csv
import hashlib
import json
import math

import numpy as np
import torch


@dataclass(frozen=True)
class UnitKey:
    unit_type: str
    layer_id: int
    unit_id: int


@dataclass(frozen=True)
class WeightSegment:
    module_name: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    @property
    def size(self) -> int:
        return (self.row_end - self.row_start) * (self.col_end - self.col_start)


@dataclass
class WeightBudget:
    keys: list[UnitKey]
    costs: np.ndarray
    prune_counts: np.ndarray
    scores: np.ndarray
    band_contribution: np.ndarray
    target_pruned_weights: int
    total_weights: int
    method: str
    allocation: str

    @property
    def actual_sparsity(self) -> float:
        return float(self.prune_counts.sum()) / float(self.total_weights)


def _layers(model: torch.nn.Module):
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise AttributeError("expected transformer layers at model.model.layers or model.layers")
    return layers


def _module(layer: torch.nn.Module, name: str) -> torch.nn.Module:
    current = layer
    for part in name.split("."):
        current = getattr(current, part)
    return current


def _attention_layout(layer: torch.nn.Module, model: torch.nn.Module) -> tuple[int, int, int]:
    attn = layer.self_attn
    config = getattr(model, "config", None)
    num_heads = int(getattr(attn, "num_heads", 0) or getattr(config, "num_attention_heads", 0) or 0)
    num_kv_heads = int(
        getattr(attn, "num_key_value_heads", 0)
        or getattr(config, "num_key_value_heads", 0)
        or num_heads
    )
    head_dim = int(getattr(attn, "head_dim", 0) or 0)
    if num_heads <= 0:
        raise ValueError("unable to infer attention head count")
    if head_dim <= 0:
        if attn.q_proj.out_features % num_heads:
            raise ValueError("q_proj output dimension is not divisible by attention heads")
        head_dim = attn.q_proj.out_features // num_heads
    if num_heads != num_kv_heads:
        raise ValueError(
            "paper-only weight mapping currently requires num_attention_heads == "
            "num_key_value_heads so every scored head owns a disjoint q/k/v/o weight group. "
            "This is satisfied by Llama-2-7B."
        )
    return num_heads, num_kv_heads, head_dim


def unit_segments(model: torch.nn.Module, key: UnitKey) -> list[WeightSegment]:
    layer = _layers(model)[key.layer_id]
    if key.unit_type == "mlp":
        gate = layer.mlp.gate_proj
        up = layer.mlp.up_proj
        down = layer.mlp.down_proj
        if not 0 <= key.unit_id < down.in_features:
            raise IndexError(key)
        j = key.unit_id
        return [
            WeightSegment("mlp.gate_proj", j, j + 1, 0, gate.in_features),
            WeightSegment("mlp.up_proj", j, j + 1, 0, up.in_features),
            WeightSegment("mlp.down_proj", 0, down.out_features, j, j + 1),
        ]
    if key.unit_type == "attention":
        attn = layer.self_attn
        num_heads, _, head_dim = _attention_layout(layer, model)
        if not 0 <= key.unit_id < num_heads:
            raise IndexError(key)
        a = key.unit_id * head_dim
        b = a + head_dim
        return [
            WeightSegment("self_attn.q_proj", a, b, 0, attn.q_proj.in_features),
            WeightSegment("self_attn.k_proj", a, b, 0, attn.k_proj.in_features),
            WeightSegment("self_attn.v_proj", a, b, 0, attn.v_proj.in_features),
            WeightSegment("self_attn.o_proj", 0, attn.o_proj.out_features, a, b),
        ]
    raise ValueError(f"unknown unit type: {key.unit_type}")


def unit_cost(model: torch.nn.Module, key: UnitKey) -> int:
    return int(sum(segment.size for segment in unit_segments(model, key)))


def flatten_evidence(
    model: torch.nn.Module,
    method_evidence: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    targets: Sequence[str],
) -> tuple[list[UnitKey], np.ndarray, np.ndarray, np.ndarray]:
    keys: list[UnitKey] = []
    costs: list[int] = []
    scores: list[float] = []
    bands: list[np.ndarray] = []
    for unit_type in targets:
        layer_map = method_evidence.get(unit_type, {})
        for layer_id in sorted(layer_map):
            payload = layer_map[layer_id]
            score = np.asarray(payload["score"], dtype=np.float64).reshape(-1)
            band = np.asarray(payload["bands"], dtype=np.float64)
            if band.ndim != 2 or band.shape[0] != score.size:
                raise ValueError(f"invalid evidence shape for {unit_type} layer {layer_id}")
            if unit_type == "mlp":
                layer = _layers(model)[layer_id]
                per_unit_cost = int(
                    layer.mlp.gate_proj.in_features
                    + layer.mlp.up_proj.in_features
                    + layer.mlp.down_proj.out_features
                )
            else:
                layer = _layers(model)[layer_id]
                heads, _, head_dim = _attention_layout(layer, model)
                if heads != score.size:
                    raise ValueError(f"attention score count mismatch at layer {layer_id}")
                attn = layer.self_attn
                per_unit_cost = int(head_dim * (
                    attn.q_proj.in_features + attn.k_proj.in_features
                    + attn.v_proj.in_features + attn.o_proj.out_features
                ))
            for unit_id in range(score.size):
                keys.append(UnitKey(unit_type, int(layer_id), int(unit_id)))
                costs.append(per_unit_cost)
                scores.append(float(score[unit_id]))
                bands.append(band[unit_id].copy())
    if not keys:
        raise ValueError("no paper-scored units found")
    return (
        keys,
        np.asarray(costs, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        np.asarray(bands, dtype=np.float64),
    )


def _bounds(costs: np.ndarray, min_sparsity: float, max_sparsity: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= min_sparsity <= max_sparsity < 1:
        raise ValueError("weight sparsity bounds must satisfy 0 <= min <= max < 1")
    min_prune = np.ceil(costs.astype(np.float64) * min_sparsity).astype(np.int64)
    max_prune = np.floor(costs.astype(np.float64) * max_sparsity).astype(np.int64)
    max_prune = np.minimum(max_prune, costs - 1)
    return min_prune, max_prune


def _exact_uniform_counts(costs: np.ndarray, target: int) -> np.ndarray:
    total = int(costs.sum())
    raw = costs.astype(np.float64) * (float(target) / float(total))
    counts = np.floor(raw).astype(np.int64)
    remainder = int(target - counts.sum())
    if remainder > 0:
        frac = raw - counts
        order = np.lexsort((np.arange(costs.size), -frac))
        counts[order[:remainder]] += 1
    return counts


def _finite_scores(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x)
    floor = float(np.min(x[finite]))
    ceiling = float(np.max(x[finite]))
    x[np.isnan(x)] = floor
    x[np.isneginf(x)] = floor
    x[np.isposinf(x)] = ceiling
    return x


def _normalized_bands(bands: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.nan_to_num(bands, nan=0.0), 0.0)
    denom = positive.sum(axis=0, keepdims=True)
    return positive / np.maximum(denom, 1e-12)


def _coverage_targets(coverage_ratio: float, coverage_ratios: Sequence[float] | None, band_count: int) -> np.ndarray:
    if coverage_ratios is None:
        return np.full(band_count, float(coverage_ratio), dtype=np.float64)
    result = np.asarray(tuple(coverage_ratios), dtype=np.float64)
    if result.size != band_count:
        raise ValueError("coverage-ratio count must equal band count")
    return result


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    """Robustly normalize contribution scores without dividing by unit cost.

    V8 used score/cost, which strongly penalized attention heads because a head
    owns far more weights than one FFN channel. V8.2 maps score -> sparsity
    directly and lets the global weighted projection enforce the exact budget.
    """
    x = _finite_scores(values)
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.std(x))
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - median) / scale, -6.0, 6.0)


def _integerize_exact(
    raw_prune: np.ndarray,
    costs: np.ndarray,
    min_prune: np.ndarray,
    max_prune: np.ndarray,
    target_pruned: int,
) -> np.ndarray:
    raw_counts = raw_prune * costs.astype(np.float64)
    counts = np.floor(raw_counts).astype(np.int64)
    counts = np.maximum(counts, min_prune)
    counts = np.minimum(counts, max_prune)
    residual = int(target_pruned - counts.sum())
    frac = raw_counts - np.floor(raw_counts)
    if residual > 0:
        eligible = np.flatnonzero(counts < max_prune)
        order = eligible[np.lexsort((eligible, -frac[eligible]))]
        if residual > order.size:
            # This should only happen if clipping created a larger integer gap.
            for idx in order:
                if residual <= 0:
                    break
                take = min(int(max_prune[idx] - counts[idx]), residual)
                counts[idx] += take
                residual -= take
        else:
            counts[order[:residual]] += 1
            residual = 0
    elif residual < 0:
        need = -residual
        eligible = np.flatnonzero(counts > min_prune)
        order = eligible[np.lexsort((eligible, frac[eligible]))]
        if need > order.size:
            for idx in order:
                if need <= 0:
                    break
                take = min(int(counts[idx] - min_prune[idx]), need)
                counts[idx] -= take
                need -= take
        else:
            counts[order[:need]] -= 1
            need = 0
        residual = -need
    if residual != 0:
        raise RuntimeError("unable to integerize projected sparsity to exact global target")
    return counts


def _project_from_effective_score(
    effective_score: np.ndarray,
    costs: np.ndarray,
    target_sparsity: float,
    min_sparsity: float,
    max_sparsity: float,
    temperature: float,
) -> np.ndarray:
    """Constrained score-to-sparsity projection with exact weighted 50% target.

    Higher contribution -> lower sparsity. A scalar shift is solved by bisection
    so sum_i cost_i * sparsity_i equals the requested global weight budget while
    every unit remains inside [min_sparsity, max_sparsity].
    """
    if temperature <= 0:
        raise ValueError("projection temperature must be positive")
    costs_f = costs.astype(np.float64)
    target_pruned = int(round(float(costs.sum()) * float(target_sparsity)))
    min_prune, max_prune = _bounds(costs, min_sparsity, max_sparsity)
    if target_pruned < int(min_prune.sum()) or target_pruned > int(max_prune.sum()):
        raise ValueError("global sparsity target is incompatible with per-unit bounds")

    z = np.asarray(effective_score, dtype=np.float64)
    half_span = min(float(target_sparsity - min_sparsity), float(max_sparsity - target_sparsity))
    base = float(target_sparsity) - half_span * np.tanh(z / float(temperature))

    def weighted(shift: float) -> tuple[float, np.ndarray]:
        frac = np.clip(base + shift, min_sparsity, max_sparsity)
        return float(np.dot(frac, costs_f)), frac

    lo, hi = -1.0, 1.0
    target_f = float(target_pruned)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value, _ = weighted(mid)
        if value < target_f:
            lo = mid
        else:
            hi = mid
    _, frac = weighted(0.5 * (lo + hi))
    return _integerize_exact(frac, costs, min_prune, max_prune, target_pruned)


def _allocate_prune_projected(
    scores: np.ndarray,
    bands: np.ndarray,
    costs: np.ndarray,
    target_sparsity: float,
    min_sparsity: float,
    max_sparsity: float,
    temperature: float,
    full: bool,
    coverage_ratio: float,
    coverage_ratios: Sequence[float] | None,
    coverage_alpha: float,
    projection_iterations: int,
) -> np.ndarray:
    base_z = _robust_zscore(scores)
    effective = base_z.copy()
    if not full or coverage_alpha <= 0:
        return _project_from_effective_score(
            effective, costs, target_sparsity, min_sparsity, max_sparsity, temperature
        )

    positive = np.maximum(np.nan_to_num(bands, nan=0.0), 0.0)
    per_band_share = positive / np.maximum(positive.sum(axis=0, keepdims=True), 1e-12)
    relative = positive / np.maximum(positive.max(axis=0, keepdims=True), 1e-12)
    ratios = _coverage_targets(coverage_ratio, coverage_ratios, bands.shape[1])
    # In the original structured objective, 0.90 means retaining 90% of band
    # coverage. Under partial weight quotas (45%-55%), absolute 0.90 is
    # impossible. V8.2 therefore maps the ratio onto the feasible keep-fraction
    # headroom: 0.50 + 0.90*(0.55-0.50) = 0.545 by default.
    target_keep = 1.0 - float(target_sparsity)
    max_keep = 1.0 - float(min_sparsity)
    desired = target_keep + ratios * max(0.0, max_keep - target_keep)

    counts = None
    for _ in range(max(1, int(projection_iterations))):
        counts = _project_from_effective_score(
            effective, costs, target_sparsity, min_sparsity, max_sparsity, temperature
        )
        keep_fraction = 1.0 - counts.astype(np.float64) / costs.astype(np.float64)
        coverage = (per_band_share * keep_fraction[:, None]).sum(axis=0)
        under = np.maximum(0.0, desired - coverage)
        if float(np.max(under)) < 1e-7:
            break
        bonus = relative @ under
        effective = base_z + float(coverage_alpha) * _robust_zscore(bonus)
    assert counts is not None
    return counts

def allocate_weight_budget(
    model: torch.nn.Module,
    method: str,
    method_evidence: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    targets: Sequence[str],
    target_sparsity: float = 0.50,
    allocation: str = "paper_nonuniform",
    min_unit_sparsity: float = 0.45,
    max_unit_sparsity: float = 0.55,
    projection_temperature: float = 1.0,
    coverage_ratio: float = 0.90,
    coverage_ratios: Sequence[float] | None = None,
    coverage_alpha: float = 0.10,
    greedy_batches: int = 8,
) -> WeightBudget:
    if not 0 < target_sparsity < 1:
        raise ValueError("target_sparsity must be in (0,1)")
    keys, costs, scores, bands = flatten_evidence(model, method_evidence, targets)
    total_weights = int(costs.sum())
    target_pruned = int(round(total_weights * float(target_sparsity)))

    if allocation == "uniform":
        prune = _exact_uniform_counts(costs, target_pruned)
    elif allocation == "paper_nonuniform":
        prune = _allocate_prune_projected(
            scores,
            bands,
            costs,
            target_sparsity=float(target_sparsity),
            min_sparsity=float(min_unit_sparsity),
            max_sparsity=float(max_unit_sparsity),
            temperature=float(projection_temperature),
            full=(method == "paper_full"),
            coverage_ratio=float(coverage_ratio),
            coverage_ratios=coverage_ratios,
            coverage_alpha=float(coverage_alpha),
            projection_iterations=int(greedy_batches),
        )
    else:
        raise ValueError("allocation must be 'uniform' or 'paper_nonuniform'")

    if int(prune.sum()) != target_pruned:
        raise RuntimeError("weight-budget allocator failed exact global sparsity")
    if np.any(prune >= costs):
        raise RuntimeError("a complete paper unit would be zeroed; this v8 forbids that")
    return WeightBudget(
        keys=keys,
        costs=costs,
        prune_counts=prune.astype(np.int64, copy=False),
        scores=scores,
        band_contribution=bands,
        target_pruned_weights=target_pruned,
        total_weights=total_weights,
        method=method,
        allocation=allocation,
    )


def _stable_u64(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(text, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _coprime_step(size: int, seed_value: int) -> int:
    if size <= 1:
        return 1
    step = int(seed_value % size) or 1
    while math.gcd(step, size) != 1:
        step += 1
        if step >= size:
            step = 1
    return step


def _group_rank(size: int, seed: int, device: torch.device) -> torch.Tensor:
    token = _stable_u64(seed, "rank", size)
    step = _coprime_step(size, token)
    shift = int((token >> 11) % size)
    positions = torch.arange(size, device=device, dtype=torch.int64)
    return (step * positions + shift) % size


def _group_offsets(count: int, size: int, seed: int, start: int, device: torch.device) -> torch.Tensor:
    values = [int(_stable_u64(seed, "offset", start + i) % size) for i in range(count)]
    return torch.as_tensor(values, device=device, dtype=torch.int64)


def _zero_row_groups_(weight: torch.Tensor, counts: np.ndarray, seed: int, chunk_groups: int = 256) -> None:
    """Each matrix row is one paper unit; zero an exact variable count per row."""
    rows, cols = weight.shape
    if len(counts) != rows:
        raise ValueError("row-group count mismatch")
    rank = _group_rank(cols, seed, weight.device)
    count_tensor = torch.as_tensor(counts, device=weight.device, dtype=torch.int64)
    with torch.no_grad():
        for start in range(0, rows, chunk_groups):
            end = min(rows, start + chunk_groups)
            offsets = _group_offsets(end - start, cols, seed, start, weight.device)
            order = (rank.unsqueeze(0) + offsets.unsqueeze(1)) % cols
            mask = order < count_tensor[start:end].unsqueeze(1)
            weight[start:end][mask] = 0


def _zero_col_groups_(weight: torch.Tensor, counts: np.ndarray, seed: int, chunk_groups: int = 256) -> None:
    """Each matrix column is one paper unit; zero an exact variable count per column."""
    rows, cols = weight.shape
    if len(counts) != cols:
        raise ValueError("column-group count mismatch")
    rank = _group_rank(rows, seed, weight.device)
    count_tensor = torch.as_tensor(counts, device=weight.device, dtype=torch.int64)
    with torch.no_grad():
        for start in range(0, cols, chunk_groups):
            end = min(cols, start + chunk_groups)
            offsets = _group_offsets(end - start, rows, seed, start, weight.device)
            order = (rank.unsqueeze(1) + offsets.unsqueeze(0)) % rows
            mask = order < count_tensor[start:end].unsqueeze(0)
            weight[:, start:end][mask] = 0


def _zero_head_row_blocks_(weight: torch.Tensor, counts: np.ndarray, head_dim: int, seed: int, chunk_heads: int = 2) -> None:
    """q/k/v: every head owns head_dim consecutive output rows."""
    heads = len(counts)
    if weight.shape[0] != heads * head_dim:
        raise ValueError("head-row layout mismatch")
    block_size = head_dim * weight.shape[1]
    rank = _group_rank(block_size, seed, weight.device)
    count_tensor = torch.as_tensor(counts, device=weight.device, dtype=torch.int64)
    with torch.no_grad():
        for h0 in range(0, heads, chunk_heads):
            h1 = min(heads, h0 + chunk_heads)
            offsets = _group_offsets(h1 - h0, block_size, seed, h0, weight.device)
            order = (rank.unsqueeze(0) + offsets.unsqueeze(1)) % block_size
            mask = order < count_tensor[h0:h1].unsqueeze(1)
            block = weight[h0 * head_dim:h1 * head_dim].view(h1 - h0, block_size)
            block[mask] = 0


def _zero_head_col_blocks_(weight: torch.Tensor, counts: np.ndarray, head_dim: int, seed: int, chunk_heads: int = 2) -> None:
    """o projection: every head owns head_dim consecutive input columns."""
    heads = len(counts)
    if weight.shape[1] != heads * head_dim:
        raise ValueError("head-column layout mismatch")
    rows = weight.shape[0]
    block_size = rows * head_dim
    rank = _group_rank(block_size, seed, weight.device)
    count_tensor = torch.as_tensor(counts, device=weight.device, dtype=torch.int64)
    with torch.no_grad():
        for h0 in range(0, heads, chunk_heads):
            h1 = min(heads, h0 + chunk_heads)
            for h in range(h0, h1):
                offset = int(_stable_u64(seed, "offset", h) % block_size)
                order = (rank + offset) % block_size
                mask = (order < count_tensor[h]).view(rows, head_dim)
                a, b = h * head_dim, (h + 1) * head_dim
                weight[:, a:b][mask] = 0


def _split_equal_segments(counts: np.ndarray, segment_count: int) -> list[np.ndarray]:
    base = counts // segment_count
    remainder = counts % segment_count
    return [base + (remainder > index).astype(np.int64) for index in range(segment_count)]


def apply_weight_budget_(
    model: torch.nn.Module,
    budget: WeightBudget,
    seed: int = 0,
    chunk_size: int = 262144,
) -> list[dict]:
    """LEGACY V8 compatibility helper; not used by the V8.2 pruning path.

    V8.2 applies budgets through `wanda_mask.sequential_wanda_prune_`, which
    uses activation-aware element metrics. This deterministic-uniform helper is
    retained only so older scripts/tests importing it do not break.
    """
    del chunk_size  # retained as a stable CLI/API knob; vectorized kernels bound their own chunks.
    layers = _layers(model)
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for key, count in zip(budget.keys, budget.prune_counts):
        grouped.setdefault((key.unit_type, key.layer_id), []).append((key.unit_id, int(count)))
    module_summaries: list[dict] = []

    for layer_id, layer in enumerate(layers):
        mlp_items = sorted(grouped.get(("mlp", layer_id), []))
        if mlp_items:
            counts = np.asarray([count for _unit, count in mlp_items], dtype=np.int64)
            if len(mlp_items) != layer.mlp.down_proj.in_features:
                raise ValueError("MLP budget does not cover every intermediate channel")
            gate_counts, up_counts, down_counts = _split_equal_segments(counts, 3)
            _zero_row_groups_(layer.mlp.gate_proj.weight.data, gate_counts, int(_stable_u64(seed, layer_id, "gate")))
            _zero_row_groups_(layer.mlp.up_proj.weight.data, up_counts, int(_stable_u64(seed, layer_id, "up")))
            _zero_col_groups_(layer.mlp.down_proj.weight.data, down_counts, int(_stable_u64(seed, layer_id, "down")))
            for name, segment_counts, module in (
                ("mlp.gate_proj", gate_counts, layer.mlp.gate_proj),
                ("mlp.up_proj", up_counts, layer.mlp.up_proj),
                ("mlp.down_proj", down_counts, layer.mlp.down_proj),
            ):
                module_summaries.append({
                    "layer": layer_id, "module": name,
                    "weights": int(module.weight.numel()), "pruned": int(segment_counts.sum()),
                })

        attn_items = sorted(grouped.get(("attention", layer_id), []))
        if attn_items:
            counts = np.asarray([count for _unit, count in attn_items], dtype=np.int64)
            heads, _, head_dim = _attention_layout(layer, model)
            if len(attn_items) != heads:
                raise ValueError("attention budget does not cover every head")
            q_counts, k_counts, v_counts, o_counts = _split_equal_segments(counts, 4)
            attn = layer.self_attn
            _zero_head_row_blocks_(attn.q_proj.weight.data, q_counts, head_dim, int(_stable_u64(seed, layer_id, "q")))
            _zero_head_row_blocks_(attn.k_proj.weight.data, k_counts, head_dim, int(_stable_u64(seed, layer_id, "k")))
            _zero_head_row_blocks_(attn.v_proj.weight.data, v_counts, head_dim, int(_stable_u64(seed, layer_id, "v")))
            _zero_head_col_blocks_(attn.o_proj.weight.data, o_counts, head_dim, int(_stable_u64(seed, layer_id, "o")))
            for name, segment_counts, module in (
                ("self_attn.q_proj", q_counts, attn.q_proj),
                ("self_attn.k_proj", k_counts, attn.k_proj),
                ("self_attn.v_proj", v_counts, attn.v_proj),
                ("self_attn.o_proj", o_counts, attn.o_proj),
            ):
                module_summaries.append({
                    "layer": layer_id, "module": name,
                    "weights": int(module.weight.numel()), "pruned": int(segment_counts.sum()),
                })

    requested = int(budget.prune_counts.sum())
    applied = int(sum(row["pruned"] for row in module_summaries))
    if requested != applied:
        raise RuntimeError(f"applied {applied} zero assignments but budget requested {requested}")
    return module_summaries


def write_weight_budget(path: str | Path, budget: WeightBudget, module_summaries: Sequence[Mapping] | None = None) -> None:
    root = Path(path)
    root.parent.mkdir(parents=True, exist_ok=True)
    with root.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["method", "unit_type", "layer", "unit", "cost", "pruned", "sparsity", "score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (key, cost, pruned) in enumerate(zip(budget.keys, budget.costs, budget.prune_counts)):
            writer.writerow({
                "method": budget.method, "unit_type": key.unit_type, "layer": key.layer_id,
                "unit": key.unit_id, "cost": int(cost), "pruned": int(pruned),
                "sparsity": float(pruned) / float(cost), "score": float(budget.scores[index]),
            })

    normalized_bands = _normalized_bands(budget.band_contribution)
    keep_fraction = 1.0 - (budget.prune_counts.astype(np.float64) / budget.costs.astype(np.float64))
    retained_coverage = (normalized_bands * keep_fraction[:, None]).sum(axis=0)
    meta = {
        "method": budget.method,
        "allocation": budget.allocation,
        "target_pruned_weights": budget.target_pruned_weights,
        "total_weights": budget.total_weights,
        "actual_sparsity": budget.actual_sparsity,
        "min_unit_sparsity": float(np.min(budget.prune_counts / budget.costs)),
        "max_unit_sparsity": float(np.max(budget.prune_counts / budget.costs)),
        "retained_band_coverage": [float(v) for v in retained_coverage],
    }
    root.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez_compressed(
        root.with_name(root.stem + "_counts.npz"),
        prune_counts=budget.prune_counts.astype(np.int64, copy=False),
        costs=budget.costs.astype(np.int64, copy=False),
    )

    aggregates: dict[tuple[int, str], dict[str, int]] = {}
    for key, cost, pruned in zip(budget.keys, budget.costs, budget.prune_counts):
        item = aggregates.setdefault((key.layer_id, key.unit_type), {"cost": 0, "pruned": 0})
        item["cost"] += int(cost); item["pruned"] += int(pruned)
    layer_path = root.with_name(root.stem + "_layer_summary.csv")
    with layer_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "unit_type", "layer", "weights", "pruned", "sparsity"])
        writer.writeheader()
        for (layer_id, unit_type), item in sorted(aggregates.items()):
            writer.writerow({
                "method": budget.method, "unit_type": unit_type, "layer": layer_id,
                "weights": item["cost"], "pruned": item["pruned"],
                "sparsity": item["pruned"] / float(item["cost"]),
            })

    if module_summaries is not None:
        module_path = root.with_name(root.stem + "_module_summary.csv")
        with module_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["layer", "module", "weights", "pruned", "sparsity"])
            writer.writeheader()
            for row in module_summaries:
                writer.writerow({**row, "sparsity": int(row["pruned"]) / float(int(row["weights"]))})


def budget_overlap(a: WeightBudget, b: WeightBudget) -> tuple[int, int, float]:
    if a.keys != b.keys or not np.array_equal(a.costs, b.costs):
        raise ValueError("budgets do not share the same paper-unit layout")
    intersection = int(np.minimum(a.prune_counts, b.prune_counts).sum())
    union = int(np.maximum(a.prune_counts, b.prune_counts).sum())
    return intersection, union, float(intersection) / float(max(1, union))
