from __future__ import annotations

from typing import Iterable, Mapping

import torch
from torch import nn

from .apply import transformer_layers


def _index(raw: Iterable[int], size: int, device: torch.device) -> torch.Tensor:
    pruned = torch.as_tensor(sorted(set(int(v) for v in raw)), dtype=torch.long, device=device)
    if pruned.numel() and (torch.any(pruned < 0) or torch.any(pruned >= size)):
        raise IndexError("structured-unit index out of range")
    keep_mask = torch.ones(size, dtype=torch.bool, device=device)
    keep_mask[pruned] = False
    keep = torch.arange(size, device=device)[keep_mask]
    if keep.numel() == 0:
        raise ValueError("cannot remove every structural unit")
    return keep


def _slice_linear(
    module: nn.Linear,
    row_indices: torch.Tensor | None = None,
    column_indices: torch.Tensor | None = None,
) -> nn.Linear:
    rows = (
        torch.arange(module.out_features, device=module.weight.device)
        if row_indices is None else row_indices.to(module.weight.device)
    )
    cols = (
        torch.arange(module.in_features, device=module.weight.device)
        if column_indices is None else column_indices.to(module.weight.device)
    )
    replacement = nn.Linear(
        int(cols.numel()),
        int(rows.numel()),
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.copy_(module.weight.index_select(0, rows).index_select(1, cols))
        if module.bias is not None:
            replacement.bias.copy_(module.bias.index_select(0, rows.to(module.bias.device)))
    return replacement



def _parameter_count(model: torch.nn.Module) -> int:
    parameters = list(model.parameters())
    if parameters:
        return int(sum(parameter.numel() for parameter in parameters))
    # Some lightweight test/demonstration wrappers keep transformer layers in
    # plain namespaces instead of registering them as submodules.
    return int(
        sum(
            parameter.numel()
            for layer in transformer_layers(model)
            for parameter in layer.parameters()
        )
    )


def materialize_structured_units_(
    model: torch.nn.Module,
    selections_by_type: Mapping[str, Mapping[int, Iterable[int]]],
    require_uniform: bool = True,
) -> dict:
    """Physically shrink MLP channels and standard-MHA heads.

    Unlike zero masks, this changes matrix shapes and can reduce eager PyTorch
    compute. The resulting attention shapes are an in-memory deployment form;
    standard Hugging Face ``from_pretrained`` does not reconstruct reduced
    attention projections without a custom model class. Uniformity is required
    by default so downstream exporters can consume one regular structure.
    """
    layers = transformer_layers(model)
    before = _parameter_count(model)
    kept_widths, kept_heads = [], []

    for layer_id, layer in enumerate(layers):
        if "mlp" in selections_by_type:
            mlp = layer.mlp
            width = int(mlp.down_proj.in_features)
            pruned = selections_by_type["mlp"].get(layer_id, ())
            keep = _index(pruned, width, mlp.down_proj.weight.device)
            if len(tuple(pruned)):
                mlp.gate_proj = _slice_linear(mlp.gate_proj, row_indices=keep)
                mlp.up_proj = _slice_linear(mlp.up_proj, row_indices=keep)
                mlp.down_proj = _slice_linear(mlp.down_proj, column_indices=keep)
                if hasattr(mlp, "intermediate_size"):
                    mlp.intermediate_size = int(keep.numel())
            kept_widths.append(int(keep.numel()))

        if "attention" in selections_by_type:
            attn = layer.self_attn
            heads = int(getattr(attn, "num_heads", 0) or getattr(model.config, "num_attention_heads", 0))
            kv_heads = int(getattr(attn, "num_key_value_heads", heads) or heads)
            if heads != kv_heads:
                raise ValueError("physical head surgery currently requires standard MHA, not GQA")
            head_dim = int(getattr(attn, "head_dim", 0) or attn.q_proj.out_features // heads)
            pruned = selections_by_type["attention"].get(layer_id, ())
            keep_heads = _index(pruned, heads, attn.q_proj.weight.device)
            if len(tuple(pruned)):
                offsets = torch.arange(head_dim, device=keep_heads.device)
                rows = (keep_heads[:, None] * head_dim + offsets[None, :]).reshape(-1)
                attn.q_proj = _slice_linear(attn.q_proj, row_indices=rows)
                attn.k_proj = _slice_linear(attn.k_proj, row_indices=rows)
                attn.v_proj = _slice_linear(attn.v_proj, row_indices=rows)
                attn.o_proj = _slice_linear(attn.o_proj, column_indices=rows)
                attn.num_heads = int(keep_heads.numel())
                attn.num_key_value_heads = int(keep_heads.numel())
                if hasattr(attn, "num_key_value_groups"):
                    attn.num_key_value_groups = 1
            kept_heads.append(int(keep_heads.numel()))

    if require_uniform:
        if kept_widths and len(set(kept_widths)) != 1:
            raise ValueError("non-uniform MLP widths cannot be saved with a standard single HF config")
        if kept_heads and len(set(kept_heads)) != 1:
            raise ValueError("non-uniform attention head counts cannot be saved with a standard single HF config")

    after = _parameter_count(model)
    summary = {
        "parameters_before": int(before),
        "parameters_after": int(after),
        "parameter_reduction_ratio": float(1.0 - after / before),
        "kept_mlp_widths": kept_widths,
        "kept_attention_heads": kept_heads,
        "uniform_structure": (not kept_widths or len(set(kept_widths)) == 1)
        and (not kept_heads or len(set(kept_heads)) == 1),
        "standard_hf_reload_supported": False,
        "deployment_note": (
            "Use the in-memory model or a custom exporter/runtime; standard "
            "Hugging Face model construction will not recreate reduced attention matrices."
        ),
    }
    model._paper_structural_summary = summary
    return summary
