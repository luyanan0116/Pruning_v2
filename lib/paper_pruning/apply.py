from __future__ import annotations

from typing import Iterable, Mapping

import torch


def transformer_layers(model: torch.nn.Module):
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise AttributeError("expected transformer layers at model.model.layers or model.layers")
    return layers


def _as_index_tensor(raw_indices: Iterable[int], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(list(raw_indices), dtype=torch.long, device=device)


def zero_mlp_channels_(model: torch.nn.Module, indices_by_layer: Mapping[int, Iterable[int]]) -> None:
    """Shape-preserving structured ablation of MLP intermediate channels.

    For every selected intermediate channel, the matching rows in ``gate_proj``
    and ``up_proj`` and the matching column in ``down_proj`` are zeroed.
    """
    layers = transformer_layers(model)
    with torch.no_grad():
        for layer_id, raw_indices in indices_by_layer.items():
            mlp = layers[int(layer_id)].mlp
            indices = _as_index_tensor(raw_indices, mlp.down_proj.weight.device)
            if indices.numel() == 0:
                continue
            unit_count = int(mlp.down_proj.in_features)
            if torch.any(indices < 0) or torch.any(indices >= unit_count):
                raise IndexError(f"layer {layer_id}: MLP channel index out of range")

            gate_indices = indices.to(mlp.gate_proj.weight.device)
            up_indices = indices.to(mlp.up_proj.weight.device)
            down_indices = indices.to(mlp.down_proj.weight.device)
            mlp.gate_proj.weight.index_fill_(0, gate_indices, 0)
            mlp.up_proj.weight.index_fill_(0, up_indices, 0)
            mlp.down_proj.weight.index_fill_(1, down_indices, 0)
            if getattr(mlp.gate_proj, "bias", None) is not None:
                mlp.gate_proj.bias.index_fill_(0, gate_indices, 0)
            if getattr(mlp.up_proj, "bias", None) is not None:
                mlp.up_proj.bias.index_fill_(0, up_indices, 0)


def _attention_layout(attn: torch.nn.Module) -> tuple[int, int, int]:
    num_heads = int(getattr(attn, "num_heads", 0) or 0)
    num_kv_heads = int(getattr(attn, "num_key_value_heads", num_heads) or num_heads)
    head_dim = int(getattr(attn, "head_dim", 0) or 0)

    q_out = int(attn.q_proj.out_features)
    if num_heads <= 0:
        if head_dim <= 0:
            raise AttributeError("attention module exposes neither num_heads nor head_dim")
        num_heads = q_out // head_dim
    if head_dim <= 0:
        if q_out % num_heads != 0:
            raise ValueError("q_proj output dimension is not divisible by num_heads")
        head_dim = q_out // num_heads
    if num_kv_heads <= 0:
        num_kv_heads = num_heads
    return num_heads, num_kv_heads, head_dim


def _expanded_head_rows(head_indices: torch.Tensor, head_dim: int) -> torch.Tensor:
    offsets = torch.arange(head_dim, device=head_indices.device, dtype=torch.long)
    return (head_indices[:, None] * head_dim + offsets[None, :]).reshape(-1)


def zero_attention_heads_(model: torch.nn.Module, indices_by_layer: Mapping[int, Iterable[int]]) -> None:
    """Zero complete LLaMA attention heads in q/k/v/o projections.

    This implementation is exact for standard multi-head attention where
    ``num_attention_heads == num_key_value_heads``. Llama-2-7B satisfies this.
    Grouped-query attention shares K/V heads across multiple query heads, so
    individual-head removal would no longer be a one-to-one structural unit;
    such models are rejected rather than silently pruning the wrong tensors.
    """
    layers = transformer_layers(model)
    with torch.no_grad():
        for layer_id, raw_indices in indices_by_layer.items():
            attn = layers[int(layer_id)].self_attn
            num_heads, num_kv_heads, head_dim = _attention_layout(attn)
            if num_heads != num_kv_heads:
                raise ValueError(
                    f"layer {layer_id}: grouped-query attention has {num_heads} query heads and "
                    f"{num_kv_heads} KV heads. Exact individual q/k/v/o head pruning currently "
                    "requires equal head counts. Llama-2-7B is supported."
                )

            indices = _as_index_tensor(raw_indices, attn.q_proj.weight.device)
            if indices.numel() == 0:
                continue
            if torch.any(indices < 0) or torch.any(indices >= num_heads):
                raise IndexError(f"layer {layer_id}: attention head index out of range")

            q_rows = _expanded_head_rows(indices, head_dim)
            k_rows = _expanded_head_rows(indices.to(attn.k_proj.weight.device), head_dim)
            v_rows = _expanded_head_rows(indices.to(attn.v_proj.weight.device), head_dim)
            o_cols = _expanded_head_rows(indices.to(attn.o_proj.weight.device), head_dim)

            attn.q_proj.weight.index_fill_(0, q_rows, 0)
            attn.k_proj.weight.index_fill_(0, k_rows, 0)
            attn.v_proj.weight.index_fill_(0, v_rows, 0)
            attn.o_proj.weight.index_fill_(1, o_cols, 0)

            for projection, rows in (
                (attn.q_proj, q_rows),
                (attn.k_proj, k_rows),
                (attn.v_proj, v_rows),
            ):
                if getattr(projection, "bias", None) is not None:
                    projection.bias.index_fill_(0, rows.to(projection.bias.device), 0)


def zero_structured_units_(
    model: torch.nn.Module,
    selections_by_type: Mapping[str, Mapping[int, Iterable[int]]],
) -> None:
    """Apply all requested structured masks without changing tensor shapes."""
    if "mlp" in selections_by_type:
        zero_mlp_channels_(model, selections_by_type["mlp"])
    if "attention" in selections_by_type:
        zero_attention_heads_(model, selections_by_type["attention"])
