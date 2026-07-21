from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Mapping

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


def rank_normalize(scores: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """Map arbitrary contribution scores to a stable positive rank factor.

    Low-contribution units receive ``floor`` and high-contribution units receive
    1.0. Rank normalization is used because MI, local MI and LCB can have
    different numeric scales and LCB values may be negative.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("scores must be non-empty")
    if not 0 < floor <= 1:
        raise ValueError("score floor must be in (0,1]")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    if values.size == 1:
        ranks[order] = 1.0
    else:
        ranks[order] = np.linspace(0.0, 1.0, values.size)
    return (floor + (1.0 - floor) * ranks).astype(np.float32)


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

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="stable")
    percentile = np.empty(values.size, dtype=np.float64)
    if values.size == 1:
        percentile[order] = 0.5
    else:
        percentile[order] = np.linspace(0.0, 1.0, values.size)

    # Low-contribution units have percentile close to zero and therefore get
    # a larger pruning weight. High-contribution units are protected.
    weights = np.exp(-temperature * percentile)
    lower = max(0.0, target_ratio * (1.0 - spread))
    upper = min(0.999, target_ratio * (1.0 + spread))
    if lower > target_ratio or upper < target_ratio:
        lower, upper = 0.0, 0.999

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
    spread: float = 0.8,
    temperature: float = 2.0,
) -> np.ndarray:
    """Allocate an exact matrix-level budget across output rows.

    Wanda normally removes the same number of weights from every output row.
    Here the total budget is unchanged, but lower-contribution structural units
    receive more row-wise sparsity and higher-contribution units receive less.
    """
    row_sparsity = _row_sparsities(
        scores,
        rows_per_unit=rows_per_unit,
        target_ratio=target_ratio,
        spread=spread,
        temperature=temperature,
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

    # The clipping limits can make a single pass insufficient only for unusual
    # parameter combinations. Finish deterministically if needed.
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
            raise RuntimeError("unable to allocate the requested row pruning budget")
    return counts


def _mask_rows_variable_k_(
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    row_counts: np.ndarray,
    chunk_rows: int,
) -> None:
    rows, columns = weight.shape
    if len(row_counts) != rows:
        raise ValueError(f"row count length {len(row_counts)} does not match weight rows {rows}")
    sqrt_scale = torch.sqrt(torch.clamp(input_scale.float(), min=1e-12))
    if sqrt_scale.numel() != columns:
        raise ValueError("activation scale width does not match matrix input dimension")

    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        local_counts = torch.as_tensor(row_counts[start:end], device=weight.device, dtype=torch.long)
        max_k = int(local_counts.max().item()) if local_counts.numel() else 0
        if max_k <= 0:
            continue
        metric = weight[start:end].detach().float().abs() * sqrt_scale.unsqueeze(0)
        indices = torch.topk(metric, k=max_k, dim=1, largest=False, sorted=True).indices
        valid = torch.arange(max_k, device=weight.device).unsqueeze(0) < local_counts.unsqueeze(1)
        local_rows = torch.arange(end - start, device=weight.device).unsqueeze(1).expand_as(indices)
        weight[start:end][local_rows[valid], indices[valid]] = 0
        del metric, indices, valid, local_rows


def _mask_rows_fixed_k_with_column_factor_(
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    column_factor: np.ndarray,
    ratio: float,
    chunk_rows: int,
) -> None:
    rows, columns = weight.shape
    k = int(columns * ratio)
    if k <= 0:
        return
    if k >= columns:
        raise ValueError("ratio would prune every input weight in an output row")
    sqrt_scale = torch.sqrt(torch.clamp(input_scale.float(), min=1e-12))
    factor = torch.as_tensor(column_factor, device=weight.device, dtype=torch.float32)
    if sqrt_scale.numel() != columns or factor.numel() != columns:
        raise ValueError("activation scale or contribution factor width mismatch")
    combined = sqrt_scale * factor
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        metric = weight[start:end].detach().float().abs() * combined.unsqueeze(0)
        indices = torch.topk(metric, k=k, dim=1, largest=False, sorted=False).indices
        local_rows = torch.arange(end - start, device=weight.device).unsqueeze(1).expand_as(indices)
        weight[start:end][local_rows, indices] = 0
        del metric, indices, local_rows


def _expanded_head_factor(scores: np.ndarray, head_dim: int, floor: float) -> np.ndarray:
    return np.repeat(rank_normalize(scores, floor=floor), int(head_dim))


def _module_orientation(module_name: str) -> str:
    if module_name in {"mlp.gate_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}:
        return "rows"
    if module_name in {"mlp.down_proj", "self_attn.o_proj"}:
        return "columns"
    raise ValueError(f"unsupported module: {module_name}")


def apply_paper_wanda_weight_masks_(
    model: torch.nn.Module,
    scores_by_type: Mapping[str, Mapping[int, np.ndarray]],
    response_cache,
    targets: tuple[str, ...],
    mlp_ratio: float,
    attention_ratio: float,
    score_floor: float = 0.05,
    row_spread: float = 0.8,
    temperature: float = 2.0,
    chunk_rows: int = 256,
) -> list[dict]:
    """Apply paper-guided Wanda-style unstructured masks to all chosen linears.

    The base importance is Wanda's ``|W| * sqrt(E[x^2])``. MI, granular-ball
    or LCB contribution scores then steer where the fixed per-matrix weight
    budget lands. No complete MLP channel or attention head is forced to zero.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    layers = transformer_layers(model)
    summaries: list[dict] = []

    with torch.no_grad():
        for layer_id, layer in enumerate(layers):
            layout = response_cache.attention_layouts[layer_id]
            num_heads = int(layout["num_heads"])
            num_kv_heads = int(layout["num_key_value_heads"])
            head_dim = int(layout["head_dim"])
            if "attention" in targets and num_heads != num_kv_heads:
                raise ValueError(
                    "paper-guided Wanda masking currently expects equal query and KV head counts; "
                    f"layer {layer_id} has {num_heads}/{num_kv_heads}"
                )

            module_names = []
            if "attention" in targets:
                module_names.extend(ATTENTION_MODULES)
            if "mlp" in targets:
                module_names.extend(MLP_MODULES)

            for module_name in module_names:
                module = get_submodule(layer, module_name)
                weight = module.weight.data
                before_zero = int((weight == 0).sum().item())
                input_scale_np = response_cache.load_activation_scale(layer_id, module_name)
                input_scale = torch.as_tensor(input_scale_np, device=weight.device, dtype=torch.float32)
                orientation = _module_orientation(module_name)

                if module_name.startswith("mlp."):
                    unit_scores = np.asarray(scores_by_type["mlp"][layer_id], dtype=np.float32)
                    ratio = float(mlp_ratio)
                    rows_per_unit = 1
                else:
                    unit_scores = np.asarray(scores_by_type["attention"][layer_id], dtype=np.float32)
                    ratio = float(attention_ratio)
                    rows_per_unit = head_dim

                if orientation == "rows":
                    if module_name in {"self_attn.k_proj", "self_attn.v_proj"}:
                        rows_per_unit = head_dim
                    row_counts = allocate_row_prune_counts(
                        unit_scores,
                        rows_per_unit=rows_per_unit,
                        columns=weight.shape[1],
                        target_ratio=ratio,
                        spread=row_spread,
                        temperature=temperature,
                    )
                    if row_counts.size != weight.shape[0]:
                        raise ValueError(
                            f"{module_name} rows={weight.shape[0]} but expanded score rows={row_counts.size}"
                        )
                    _mask_rows_variable_k_(weight, input_scale, row_counts, chunk_rows)
                else:
                    if module_name == "mlp.down_proj":
                        column_factor = rank_normalize(unit_scores, floor=score_floor)
                    else:
                        column_factor = _expanded_head_factor(unit_scores, head_dim, floor=score_floor)
                    _mask_rows_fixed_k_with_column_factor_(
                        weight,
                        input_scale,
                        column_factor,
                        ratio,
                        chunk_rows,
                    )

                after_zero = int((weight == 0).sum().item())
                summaries.append(
                    {
                        "layer": layer_id,
                        "module": module_name,
                        "rows": int(weight.shape[0]),
                        "columns": int(weight.shape[1]),
                        "target_ratio": ratio,
                        "zeros_before": before_zero,
                        "zeros_after": after_zero,
                        "actual_ratio": after_zero / weight.numel(),
                    }
                )
                print(
                    f"  [wanda-weight] layer={layer_id:02d} module={module_name:<20} "
                    f"target={ratio:.4f} actual={after_zero / weight.numel():.6f}",
                    flush=True,
                )
    return summaries


def write_weight_mask_summary(path: str | Path, rows: list[dict]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layer", "module", "rows", "columns", "target_ratio",
        "zeros_before", "zeros_after", "actual_ratio",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output
