from __future__ import annotations

import copy
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn


def transformer_layers(model: torch.nn.Module):
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise AttributeError("expected transformer layers at model.model.layers or model.layers")
    return layers


def _indices(values: Iterable[int], size: int, device: torch.device) -> torch.Tensor:
    idx = torch.as_tensor(sorted(set(int(v) for v in values)), dtype=torch.long, device=device)
    if idx.numel() and (torch.any(idx < 0) or torch.any(idx >= size)):
        raise IndexError(f"index out of range for size={size}")
    return idx


def _keep_from_prune(prune: Iterable[int], size: int, device: torch.device) -> torch.Tensor:
    p = _indices(prune, size, device)
    mask = torch.ones(size, dtype=torch.bool, device=device)
    if p.numel():
        mask[p] = False
    keep = torch.arange(size, device=device, dtype=torch.long)[mask]
    if keep.numel() < 1:
        raise ValueError("cannot prune every structural unit in a block")
    return keep


def _expanded_heads(head_indices: torch.Tensor, head_dim: int) -> torch.Tensor:
    offsets = torch.arange(head_dim, device=head_indices.device, dtype=torch.long)
    return (head_indices[:, None] * head_dim + offsets[None, :]).reshape(-1)


def _slice_linear_out_(linear: nn.Module, keep: torch.Tensor) -> None:
    device = linear.weight.device
    keep = keep.to(device)
    req = bool(linear.weight.requires_grad)
    linear.weight = nn.Parameter(
        linear.weight.detach().index_select(0, keep).contiguous(), requires_grad=req
    )
    if getattr(linear, "bias", None) is not None:
        breq = bool(linear.bias.requires_grad)
        linear.bias = nn.Parameter(
            linear.bias.detach().index_select(0, keep.to(linear.bias.device)).contiguous(),
            requires_grad=breq,
        )
    if hasattr(linear, "out_features"):
        linear.out_features = int(keep.numel())


def _slice_linear_in_(linear: nn.Module, keep: torch.Tensor) -> None:
    device = linear.weight.device
    keep = keep.to(device)
    req = bool(linear.weight.requires_grad)
    linear.weight = nn.Parameter(
        linear.weight.detach().index_select(1, keep).contiguous(), requires_grad=req
    )
    if hasattr(linear, "in_features"):
        linear.in_features = int(keep.numel())


def _attention_layout(attn: nn.Module) -> tuple[int, int, int, int]:
    num_heads = int(getattr(attn, "num_heads", 0) or 0)
    num_kv = int(getattr(attn, "num_key_value_heads", 0) or num_heads)
    head_dim = int(getattr(attn, "head_dim", 0) or 0)
    q_out = int(attn.q_proj.out_features)

    if num_heads <= 0:
        cfg = getattr(attn, "config", None)
        num_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
        num_kv = int(getattr(cfg, "num_key_value_heads", 0) or num_heads)
    if head_dim <= 0:
        if num_heads <= 0 or q_out % num_heads:
            raise ValueError("cannot infer attention head_dim")
        head_dim = q_out // num_heads
    if num_heads <= 0:
        num_heads = q_out // head_dim
    if num_kv <= 0 or num_heads % num_kv:
        raise ValueError(f"invalid Q/KV head layout {num_heads}/{num_kv}")
    return num_heads, num_kv, head_dim, num_heads // num_kv


def _localize_attention_config(attn: nn.Module, new_heads: int, new_kv: int) -> None:
    if hasattr(attn, "config"):
        cfg = copy.copy(attn.config)
        if hasattr(cfg, "num_attention_heads"):
            cfg.num_attention_heads = int(new_heads)
        if hasattr(cfg, "num_key_value_heads"):
            cfg.num_key_value_heads = int(new_kv)
        attn.config = cfg
    attn.num_heads = int(new_heads)
    attn.num_key_value_heads = int(new_kv)
    attn.num_key_value_groups = int(new_heads // new_kv)


def shrink_mlp_channels_(
    model: torch.nn.Module,
    prune_by_layer: Mapping[int, Iterable[int]],
) -> list[dict]:
    rows = []
    layers = transformer_layers(model)
    with torch.no_grad():
        for layer_id, prune in sorted(prune_by_layer.items()):
            mlp = layers[int(layer_id)].mlp
            total = int(mlp.down_proj.in_features)
            keep = _keep_from_prune(prune, total, mlp.down_proj.weight.device)
            _slice_linear_out_(mlp.gate_proj, keep)
            _slice_linear_out_(mlp.up_proj, keep)
            _slice_linear_in_(mlp.down_proj, keep)
            if hasattr(mlp, "intermediate_size"):
                mlp.intermediate_size = int(keep.numel())
            if hasattr(mlp, "config"):
                cfg = copy.copy(mlp.config)
                if hasattr(cfg, "intermediate_size"):
                    cfg.intermediate_size = int(keep.numel())
                mlp.config = cfg
            rows.append({
                "layer": int(layer_id), "unit_type": "mlp",
                "total": total, "kept": int(keep.numel()),
                "pruned": total - int(keep.numel()),
            })
    return rows


def shrink_attention_bundles_(
    model: torch.nn.Module,
    prune_by_layer: Mapping[int, Iterable[int]],
) -> list[dict]:
    """Physically remove complete MHA heads or complete GQA KV-sharing bundles."""
    rows = []
    layers = transformer_layers(model)
    with torch.no_grad():
        for layer_id, prune in sorted(prune_by_layer.items()):
            attn = layers[int(layer_id)].self_attn
            num_heads, num_kv, head_dim, q_per_kv = _attention_layout(attn)
            keep_bundle = _keep_from_prune(prune, num_kv, attn.k_proj.weight.device)

            # A GQA structural unit = one KV head + q_per_kv query heads.
            q_heads = []
            for bundle in keep_bundle.detach().cpu().tolist():
                q_heads.extend(range(bundle * q_per_kv, (bundle + 1) * q_per_kv))
            q_heads = torch.as_tensor(q_heads, dtype=torch.long, device=attn.q_proj.weight.device)
            q_rows = _expanded_heads(q_heads, head_dim)
            kv_rows = _expanded_heads(keep_bundle.to(attn.k_proj.weight.device), head_dim)
            o_cols = _expanded_heads(q_heads.to(attn.o_proj.weight.device), head_dim)

            _slice_linear_out_(attn.q_proj, q_rows)
            _slice_linear_out_(attn.k_proj, kv_rows)
            _slice_linear_out_(attn.v_proj, kv_rows.to(attn.v_proj.weight.device))
            _slice_linear_in_(attn.o_proj, o_cols)

            new_kv = int(keep_bundle.numel())
            new_heads = new_kv * q_per_kv
            _localize_attention_config(attn, new_heads, new_kv)
            rows.append({
                "layer": int(layer_id), "unit_type": "attention",
                "total": num_kv, "kept": new_kv, "pruned": num_kv - new_kv,
                "query_heads_per_kv": q_per_kv,
                "query_heads_kept": new_heads,
            })
    return rows


def zero_mlp_channels_(
    model: torch.nn.Module,
    prune_by_layer: Mapping[int, Iterable[int]],
) -> list[dict]:
    rows = []
    layers = transformer_layers(model)
    with torch.no_grad():
        for layer_id, prune in sorted(prune_by_layer.items()):
            mlp = layers[int(layer_id)].mlp
            total = int(mlp.down_proj.in_features)
            p = _indices(prune, total, mlp.down_proj.weight.device)
            if p.numel():
                mlp.gate_proj.weight.index_fill_(0, p.to(mlp.gate_proj.weight.device), 0)
                mlp.up_proj.weight.index_fill_(0, p.to(mlp.up_proj.weight.device), 0)
                mlp.down_proj.weight.index_fill_(1, p.to(mlp.down_proj.weight.device), 0)
                if getattr(mlp.gate_proj, "bias", None) is not None:
                    mlp.gate_proj.bias.index_fill_(0, p.to(mlp.gate_proj.bias.device), 0)
                if getattr(mlp.up_proj, "bias", None) is not None:
                    mlp.up_proj.bias.index_fill_(0, p.to(mlp.up_proj.bias.device), 0)
            rows.append({
                "layer": int(layer_id), "unit_type": "mlp",
                "total": total, "kept": total - int(p.numel()), "pruned": int(p.numel()),
            })
    return rows


def zero_attention_bundles_(
    model: torch.nn.Module,
    prune_by_layer: Mapping[int, Iterable[int]],
) -> list[dict]:
    rows = []
    layers = transformer_layers(model)
    with torch.no_grad():
        for layer_id, prune in sorted(prune_by_layer.items()):
            attn = layers[int(layer_id)].self_attn
            num_heads, num_kv, head_dim, q_per_kv = _attention_layout(attn)
            p = _indices(prune, num_kv, attn.k_proj.weight.device)
            q_heads = []
            for bundle in p.detach().cpu().tolist():
                q_heads.extend(range(bundle * q_per_kv, (bundle + 1) * q_per_kv))
            q_heads = torch.as_tensor(q_heads, dtype=torch.long, device=attn.q_proj.weight.device)
            q_rows = _expanded_heads(q_heads, head_dim) if q_heads.numel() else q_heads
            kv_rows = _expanded_heads(p.to(attn.k_proj.weight.device), head_dim) if p.numel() else p
            o_cols = _expanded_heads(q_heads.to(attn.o_proj.weight.device), head_dim) if q_heads.numel() else q_heads

            if q_rows.numel():
                attn.q_proj.weight.index_fill_(0, q_rows, 0)
                attn.o_proj.weight.index_fill_(1, o_cols, 0)
            if kv_rows.numel():
                attn.k_proj.weight.index_fill_(0, kv_rows, 0)
                attn.v_proj.weight.index_fill_(0, kv_rows.to(attn.v_proj.weight.device), 0)
            rows.append({
                "layer": int(layer_id), "unit_type": "attention",
                "total": num_kv, "kept": num_kv - int(p.numel()), "pruned": int(p.numel()),
                "query_heads_per_kv": q_per_kv,
            })
    return rows


def apply_structured_pruning_(
    model: torch.nn.Module,
    prune_by_type: Mapping[str, Mapping[int, Iterable[int]]],
    mode: str = "shrink",
) -> list[dict]:
    """Apply complete structural-unit pruning.

    mode="shrink": physically reduces q/k/v/o and gate/up/down tensor dimensions.
    mode="zero": shape-preserving complete-unit ablation for maximum HF compatibility.
    """
    if mode not in {"shrink", "zero"}:
        raise ValueError("mode must be shrink or zero")
    rows = []
    if mode == "shrink":
        if "attention" in prune_by_type:
            rows.extend(shrink_attention_bundles_(model, prune_by_type["attention"]))
        if "mlp" in prune_by_type:
            rows.extend(shrink_mlp_channels_(model, prune_by_type["mlp"]))
    else:
        if "attention" in prune_by_type:
            rows.extend(zero_attention_bundles_(model, prune_by_type["attention"]))
        if "mlp" in prune_by_type:
            rows.extend(zero_mlp_channels_(model, prune_by_type["mlp"]))
    return rows
