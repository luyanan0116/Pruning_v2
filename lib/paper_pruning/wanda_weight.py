from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


ATTENTION_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
MLP_MODULES = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
ALL_LINEAR_MODULES = ATTENTION_MODULES + MLP_MODULES


def transformer_layers(model: torch.nn.Module):
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise AttributeError("expected transformer layers at model.model.layers or model.layers")
    return layers


def get_submodule(root: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
    module = root
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


def _rank01(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("scores must be non-empty")
    finite = np.nan_to_num(values, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    order = np.argsort(finite, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    if values.size == 1:
        ranks[order] = 0.5
    else:
        ranks[order] = np.linspace(0.0, 1.0, values.size)
    return ranks


def rank_normalize(scores: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """Legacy [floor,1] rank map retained for compatibility."""
    if not 0 < floor <= 1:
        raise ValueError("score floor must be in (0,1]")
    ranks = _rank01(scores)
    return (floor + (1.0 - floor) * ranks).astype(np.float32)


def centered_rank_factor(scores: np.ndarray, strength: float = 0.01) -> np.ndarray:
    """Near-one multiplicative guidance that preserves Wanda's base metric.

    At the recommended strength 0.01, factors lie in approximately
    [0.990, 1.010]. This is deliberately much gentler than the v4 [0.05, 1]
    multiplier, which could overwhelm Wanda and caused high PPL.
    """
    if strength < 0:
        raise ValueError("guidance strength must be non-negative")
    centered = 2.0 * _rank01(scores) - 1.0
    return np.exp(strength * centered).astype(np.float32)


def allocate_unit_sparsities(
    scores: np.ndarray,
    target_ratio: float,
    min_ratio: float = 0.30,
    max_ratio: float = 0.70,
) -> np.ndarray:
    """Map unit priority to a real pruning budget.

    Higher contribution/priority -> lower sparsity.
    Lower contribution/priority -> higher sparsity.

    The returned unit ratios are bounded by ``[min_ratio, max_ratio]`` and
    their mean is forced to ``target_ratio``.  With target=0.5, min=0.3,
    max=0.7 the lowest-priority unit approaches 70% sparsity while the
    highest-priority unit approaches 30% sparsity.
    """
    if not 0.0 <= min_ratio < max_ratio < 1.0:
        raise ValueError("unit sparsity bounds must satisfy 0 <= min < max < 1")
    if not min_ratio <= target_ratio <= max_ratio:
        raise ValueError("target_ratio must lie inside the unit sparsity bounds")

    rank = _rank01(scores)  # 0 = least important, 1 = most important
    base = max_ratio - (max_ratio - min_ratio) * rank

    # Shift-and-clip so the mean is exactly the requested target even when
    # bounds are asymmetric around the target.
    lo = min_ratio - float(base.max()) - 1.0
    hi = max_ratio - float(base.min()) + 1.0
    for _ in range(100):
        delta = (lo + hi) / 2.0
        current = np.clip(base + delta, min_ratio, max_ratio).mean()
        if current < target_ratio:
            lo = delta
        else:
            hi = delta
    result = np.clip(base + (lo + hi) / 2.0, min_ratio, max_ratio)
    return result.astype(np.float64)


def allocate_axis_prune_counts(
    scores: np.ndarray,
    items_per_unit: int,
    axis_length: int,
    target_ratio: float,
    min_ratio: float = 0.30,
    max_ratio: float = 0.70,
) -> np.ndarray:
    """Allocate exact integer zero counts for rows or columns.

    ``items_per_unit`` is rows-per-unit for row-oriented matrices and
    columns-per-unit for column-oriented matrices.  The sum of the returned
    counts is exactly round(target_ratio * number_of_weights).
    """
    if items_per_unit <= 0 or axis_length <= 0:
        raise ValueError("items_per_unit and axis_length must be positive")

    unit_sparsity = allocate_unit_sparsities(
        scores,
        target_ratio=target_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )
    item_sparsity = np.repeat(unit_sparsity, int(items_per_unit))
    raw = item_sparsity * int(axis_length)
    counts = np.floor(raw).astype(np.int64)
    counts = np.clip(counts, 0, int(axis_length))

    target = int(round(target_ratio * counts.size * int(axis_length)))
    residual = target - int(counts.sum())
    fractional = raw - np.floor(raw)

    if residual > 0:
        order = np.lexsort((np.arange(counts.size), -fractional))
        for index in order:
            if residual <= 0:
                break
            if counts[index] < axis_length:
                counts[index] += 1
                residual -= 1
    elif residual < 0:
        order = np.lexsort((np.arange(counts.size), fractional))
        for index in order:
            if residual >= 0:
                break
            if counts[index] > 0:
                counts[index] -= 1
                residual += 1

    cursor = 0
    while residual != 0:
        index = cursor % counts.size
        if residual > 0 and counts[index] < axis_length:
            counts[index] += 1
            residual -= 1
        elif residual < 0 and counts[index] > 0:
            counts[index] -= 1
            residual += 1
        cursor += 1
        if cursor > counts.size * axis_length * 2:
            raise RuntimeError("unable to satisfy exact weight budget")

    if int(counts.sum()) != target:
        raise RuntimeError("integer allocation failed to preserve exact target sparsity")
    return counts


def _mask_from_metric_variable_k_columns_(
    weight: torch.Tensor,
    metric: torch.Tensor,
    column_counts: np.ndarray,
    chunk_columns: int,
) -> None:
    """Column-oriented counterpart of ``_mask_from_metric_variable_k_``."""
    rows, columns = weight.shape
    if int(column_counts.size) != columns:
        raise ValueError("column_counts length must equal weight columns")
    for start in range(0, columns, chunk_columns):
        end = min(columns, start + chunk_columns)
        local_counts = torch.as_tensor(
            column_counts[start:end], device=weight.device, dtype=torch.long
        )
        max_k = int(local_counts.max().item()) if local_counts.numel() else 0
        if max_k <= 0:
            continue
        # Work on [local_columns, rows] so every column has its own k.
        local_metric = metric[:, start:end].transpose(0, 1)
        indices = torch.topk(
            local_metric, k=max_k, dim=1, largest=False, sorted=True
        ).indices
        valid = (
            torch.arange(max_k, device=weight.device).unsqueeze(0)
            < local_counts.unsqueeze(1)
        )
        local_cols = torch.arange(
            end - start, device=weight.device
        ).unsqueeze(1).expand_as(indices)
        weight[
            indices[valid],
            (local_cols[valid] + start),
        ] = 0


def _realized_unit_sparsities(
    module_name: str,
    weight: torch.Tensor,
    unit_count: int,
    head_dim: int,
) -> np.ndarray:
    zeros = (weight == 0)
    orientation = _module_orientation(module_name)
    if orientation == "rows":
        rows_per_unit = 1 if module_name.startswith("mlp.") else int(head_dim)
        if rows_per_unit * unit_count != weight.shape[0]:
            raise ValueError(f"{module_name}: invalid rows-per-unit layout")
        shaped = zeros.reshape(unit_count, rows_per_unit, weight.shape[1])
    else:
        cols_per_unit = 1 if module_name == "mlp.down_proj" else int(head_dim)
        if cols_per_unit * unit_count != weight.shape[1]:
            raise ValueError(f"{module_name}: invalid columns-per-unit layout")
        shaped = zeros.transpose(0, 1).reshape(
            unit_count, cols_per_unit, weight.shape[0]
        )
    return shaped.float().mean(dim=(1, 2)).cpu().numpy()


def apply_unit_budget_weight_module_(
    module_name: str,
    module: torch.nn.Module,
    input_scale: torch.Tensor,
    unit_scores: np.ndarray,
    ratio: float,
    head_dim: int,
    min_unit_sparsity: float = 0.30,
    max_unit_sparsity: float = 0.70,
    chunk_rows: int = 256,
) -> dict:
    """Weight pruning where unit score controls *how much* each unit is pruned.

    The unit-level score determines the local sparsity budget.  The usual
    |W|*sqrt(activation) metric is used only to decide which individual
    weights to remove *inside* that already-assigned unit budget.
    """
    weight = module.weight.data
    before_zero = int((weight == 0).sum().item())
    scores = np.asarray(unit_scores, dtype=np.float64).reshape(-1)

    scale = torch.sqrt(torch.clamp(input_scale.float(), min=1e-12))
    if scale.numel() != weight.shape[1]:
        raise ValueError(
            f"{module_name}: activation scale {scale.numel()} != input width {weight.shape[1]}"
        )
    metric = weight.detach().float().abs() * scale.unsqueeze(0)
    orientation = _module_orientation(module_name)

    if orientation == "rows":
        rows_per_unit = 1 if module_name.startswith("mlp.") else int(head_dim)
        counts = allocate_axis_prune_counts(
            scores,
            items_per_unit=rows_per_unit,
            axis_length=weight.shape[1],
            target_ratio=ratio,
            min_ratio=min_unit_sparsity,
            max_ratio=max_unit_sparsity,
        )
        if counts.size != weight.shape[0]:
            raise ValueError(
                f"{module_name}: allocated row count {counts.size} != {weight.shape[0]}"
            )
        _mask_from_metric_variable_k_(weight, metric, counts, chunk_rows)
    else:
        columns_per_unit = 1 if module_name == "mlp.down_proj" else int(head_dim)
        counts = allocate_axis_prune_counts(
            scores,
            items_per_unit=columns_per_unit,
            axis_length=weight.shape[0],
            target_ratio=ratio,
            min_ratio=min_unit_sparsity,
            max_ratio=max_unit_sparsity,
        )
        if counts.size != weight.shape[1]:
            raise ValueError(
                f"{module_name}: allocated column count {counts.size} != {weight.shape[1]}"
            )
        _mask_from_metric_variable_k_columns_(
            weight, metric, counts, chunk_columns=chunk_rows
        )

    realized = _realized_unit_sparsities(
        module_name, weight, scores.size, head_dim
    )
    after_zero = int((weight == 0).sum().item())
    return {
        "module": module_name,
        "rows": int(weight.shape[0]),
        "columns": int(weight.shape[1]),
        "target_ratio": float(ratio),
        "zeros_before": before_zero,
        "zeros_after": after_zero,
        "actual_ratio": after_zero / weight.numel(),
        "unit_sparsity_min": float(realized.min()),
        "unit_sparsity_mean": float(realized.mean()),
        "unit_sparsity_max": float(realized.max()),
    }


def _row_sparsities(
    scores: np.ndarray,
    rows_per_unit: int,
    target_ratio: float,
    spread: float,
    temperature: float,
) -> np.ndarray:
    if not 0 < target_ratio < 1:
        raise ValueError("target_ratio must be in (0,1)")
    if rows_per_unit <= 0:
        raise ValueError("rows_per_unit must be positive")
    if not 0 <= spread < 1:
        raise ValueError("row spread must be in [0,1)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    percentile = _rank01(scores)
    weights = np.exp(-temperature * percentile)
    lower = max(0.0, target_ratio * (1.0 - spread))
    upper = min(0.999, target_ratio * (1.0 + spread))

    lo, hi = 0.0, 1000.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        current = np.clip(mid * weights, lower, upper).mean()
        if current < target_ratio:
            lo = mid
        else:
            hi = mid
    unit_sparsity = np.clip(((lo + hi) / 2.0) * weights, lower, upper)
    return np.repeat(unit_sparsity, rows_per_unit).astype(np.float64)


def allocate_row_prune_counts(
    scores: np.ndarray,
    rows_per_unit: int,
    columns: int,
    target_ratio: float,
    spread: float = 0.01,
    temperature: float = 1.0,
) -> np.ndarray:
    """Redistribute only a small part of the row budget across units."""
    row_sparsity = _row_sparsities(
        scores, rows_per_unit, target_ratio, spread, temperature
    )
    raw = row_sparsity * int(columns)
    counts = np.floor(raw).astype(np.int64)
    counts = np.clip(counts, 0, max(0, int(columns) - 1))
    target = int(round(target_ratio * counts.size * int(columns)))
    residual = target - int(counts.sum())
    fractional = raw - np.floor(raw)

    if residual > 0:
        order = np.lexsort((np.arange(counts.size), -fractional))
        for index in order:
            if residual <= 0:
                break
            if counts[index] < columns - 1:
                counts[index] += 1
                residual -= 1
    elif residual < 0:
        order = np.lexsort((np.arange(counts.size), fractional))
        for index in order:
            if residual >= 0:
                break
            if counts[index] > 0:
                counts[index] -= 1
                residual += 1

    cursor = 0
    while residual != 0 and counts.size:
        index = cursor % counts.size
        if residual > 0 and counts[index] < columns - 1:
            counts[index] += 1
            residual -= 1
        elif residual < 0 and counts[index] > 0:
            counts[index] -= 1
            residual += 1
        cursor += 1
        if cursor > counts.size * columns * 2:
            raise RuntimeError("unable to allocate requested row budget")
    return counts


def _mask_from_metric_variable_k_(
    weight: torch.Tensor,
    metric: torch.Tensor,
    row_counts: np.ndarray,
    chunk_rows: int,
) -> None:
    rows, _columns = weight.shape
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        local_counts = torch.as_tensor(
            row_counts[start:end], device=weight.device, dtype=torch.long
        )
        max_k = int(local_counts.max().item()) if local_counts.numel() else 0
        if max_k <= 0:
            continue
        indices = torch.topk(
            metric[start:end], k=max_k, dim=1, largest=False, sorted=True
        ).indices
        valid = (
            torch.arange(max_k, device=weight.device).unsqueeze(0)
            < local_counts.unsqueeze(1)
        )
        local_rows = torch.arange(
            end - start, device=weight.device
        ).unsqueeze(1).expand_as(indices)
        weight[start:end][local_rows[valid], indices[valid]] = 0


def _mask_from_metric_fixed_k_(
    weight: torch.Tensor,
    metric: torch.Tensor,
    ratio: float,
    chunk_rows: int,
) -> None:
    rows, columns = weight.shape
    k = int(columns * ratio)
    if k <= 0:
        return
    if k >= columns:
        raise ValueError("ratio would prune every weight in an output row")
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        indices = torch.topk(
            metric[start:end], k=k, dim=1, largest=False, sorted=False
        ).indices
        local_rows = torch.arange(
            end - start, device=weight.device
        ).unsqueeze(1).expand_as(indices)
        weight[start:end][local_rows, indices] = 0


def _module_orientation(module_name: str) -> str:
    if module_name in {
        "mlp.gate_proj", "mlp.up_proj", "self_attn.q_proj",
        "self_attn.k_proj", "self_attn.v_proj",
    }:
        return "rows"
    if module_name in {"mlp.down_proj", "self_attn.o_proj"}:
        return "columns"
    raise ValueError(f"unsupported module: {module_name}")


def apply_guided_wanda_module_(
    module_name: str,
    module: torch.nn.Module,
    input_scale: torch.Tensor,
    unit_scores: np.ndarray,
    ratio: float,
    head_dim: int,
    row_spread: float = 0.01,
    row_temperature: float = 1.0,
    guidance_strength: float = 0.01,
    chunk_rows: int = 256,
) -> dict:
    """Apply one exact-budget Wanda mask with gentle paper-score guidance."""
    weight = module.weight.data
    before_zero = int((weight == 0).sum().item())
    scale = torch.sqrt(torch.clamp(input_scale.float(), min=1e-12))
    if scale.numel() != weight.shape[1]:
        raise ValueError(
            f"{module_name}: activation scale {scale.numel()} != input width {weight.shape[1]}"
        )
    metric = weight.detach().float().abs() * scale.unsqueeze(0)
    orientation = _module_orientation(module_name)

    if module_name.startswith("mlp."):
        rows_per_unit = 1
    else:
        rows_per_unit = int(head_dim)

    if orientation == "rows":
        row_counts = allocate_row_prune_counts(
            unit_scores,
            rows_per_unit=rows_per_unit,
            columns=weight.shape[1],
            target_ratio=ratio,
            spread=row_spread,
            temperature=row_temperature,
        )
        if row_counts.size != weight.shape[0]:
            raise ValueError(
                f"{module_name}: score expansion {row_counts.size} != rows {weight.shape[0]}"
            )
        _mask_from_metric_variable_k_(weight, metric, row_counts, chunk_rows)
    else:
        factor = centered_rank_factor(unit_scores, guidance_strength)
        if module_name == "self_attn.o_proj":
            factor = np.repeat(factor, int(head_dim))
        if factor.size != weight.shape[1]:
            raise ValueError(
                f"{module_name}: contribution factor {factor.size} != columns {weight.shape[1]}"
            )
        metric = metric * torch.as_tensor(
            factor, device=metric.device, dtype=metric.dtype
        ).unsqueeze(0)
        _mask_from_metric_fixed_k_(weight, metric, ratio, chunk_rows)

    after_zero = int((weight == 0).sum().item())
    return {
        "module": module_name,
        "rows": int(weight.shape[0]),
        "columns": int(weight.shape[1]),
        "target_ratio": float(ratio),
        "zeros_before": before_zero,
        "zeros_after": after_zero,
        "actual_ratio": after_zero / weight.numel(),
    }


def apply_paper_wanda_weight_masks_(
    model: torch.nn.Module,
    scores_by_type: Mapping[str, Mapping[int, np.ndarray]],
    response_cache,
    targets: tuple[str, ...],
    mlp_ratio: float,
    attention_ratio: float,
    score_floor: float = 0.05,
    row_spread: float = 0.01,
    temperature: float = 1.0,
    guidance_strength: float = 0.05,
    chunk_rows: int = 256,
    layer_ratios: Mapping[str, Mapping[int, float]] | None = None,
) -> list[dict]:
    """Legacy non-sequential application; sequential mode is recommended."""
    del score_floor
    layers = transformer_layers(model)
    summaries: list[dict] = []
    with torch.no_grad():
        for layer_id, layer in enumerate(layers):
            layout = response_cache.attention_layouts[layer_id]
            num_heads = int(layout["num_heads"])
            num_kv_heads = int(layout["num_key_value_heads"])
            head_dim = int(layout["head_dim"])
            if "attention" in targets and num_heads != num_kv_heads:
                raise ValueError("guided attention masking currently expects MHA")
            module_names = []
            if "attention" in targets:
                module_names.extend(ATTENTION_MODULES)
            if "mlp" in targets:
                module_names.extend(MLP_MODULES)
            for module_name in module_names:
                module = get_submodule(layer, module_name)
                unit_type = "attention" if module_name.startswith("self_attn.") else "mlp"
                base_ratio = attention_ratio if unit_type == "attention" else mlp_ratio
                ratio = float(
                    layer_ratios.get(unit_type, {}).get(layer_id, base_ratio)
                    if layer_ratios is not None else base_ratio
                )
                input_scale = torch.as_tensor(
                    response_cache.load_activation_scale(layer_id, module_name),
                    device=module.weight.device,
                    dtype=torch.float32,
                )
                summary = apply_guided_wanda_module_(
                    module_name,
                    module,
                    input_scale,
                    np.asarray(scores_by_type[unit_type][layer_id]),
                    ratio,
                    head_dim,
                    row_spread=row_spread,
                    row_temperature=temperature,
                    guidance_strength=guidance_strength,
                    chunk_rows=chunk_rows,
                )
                summary["layer"] = layer_id
                summaries.append(summary)
    return summaries



def apply_paper_unit_budget_weight_masks_(
    model: torch.nn.Module,
    scores_by_type: Mapping[str, Mapping[int, np.ndarray]],
    response_cache,
    targets: tuple[str, ...],
    mlp_ratio: float,
    attention_ratio: float,
    min_unit_sparsity: float = 0.30,
    max_unit_sparsity: float = 0.70,
    chunk_rows: int = 256,
    layer_ratios: Mapping[str, Mapping[int, float]] | None = None,
) -> list[dict]:
    """Apply score-driven per-unit weight budgets without structural deletion."""
    layers = transformer_layers(model)
    summaries: list[dict] = []
    with torch.no_grad():
        for layer_id, layer in enumerate(layers):
            layout = response_cache.attention_layouts[layer_id]
            num_heads = int(layout["num_heads"])
            num_kv_heads = int(layout["num_key_value_heads"])
            head_dim = int(layout["head_dim"])
            if "attention" in targets and num_heads != num_kv_heads:
                raise ValueError("unit-budget attention masking currently expects MHA")

            module_names = []
            if "attention" in targets:
                module_names.extend(ATTENTION_MODULES)
            if "mlp" in targets:
                module_names.extend(MLP_MODULES)

            for module_name in module_names:
                module = get_submodule(layer, module_name)
                unit_type = "attention" if module_name.startswith("self_attn.") else "mlp"
                base_ratio = attention_ratio if unit_type == "attention" else mlp_ratio
                ratio = float(
                    layer_ratios.get(unit_type, {}).get(layer_id, base_ratio)
                    if layer_ratios is not None else base_ratio
                )
                input_scale = torch.as_tensor(
                    response_cache.load_activation_scale(layer_id, module_name),
                    device=module.weight.device,
                    dtype=torch.float32,
                )
                summary = apply_unit_budget_weight_module_(
                    module_name,
                    module,
                    input_scale,
                    np.asarray(scores_by_type[unit_type][layer_id]),
                    ratio,
                    head_dim,
                    min_unit_sparsity=min_unit_sparsity,
                    max_unit_sparsity=max_unit_sparsity,
                    chunk_rows=chunk_rows,
                )
                summary["layer"] = layer_id
                summaries.append(summary)
    return summaries

def write_weight_mask_summary(path: str | Path, rows: list[dict]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layer", "module", "rows", "columns", "target_ratio",
        "zeros_before", "zeros_after", "actual_ratio",
        "unit_sparsity_min", "unit_sparsity_mean", "unit_sparsity_max",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output
