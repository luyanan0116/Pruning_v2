from __future__ import annotations

from typing import Iterable, Mapping

import torch


def transformer_layers(model: torch.nn.Module):
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise AttributeError("expected transformer layers at model.model.layers or model.layers")
    return layers


def zero_mlp_channels_(model: torch.nn.Module, indices_by_layer: Mapping[int, Iterable[int]]) -> None:
    """Shape-preserving structured ablation of MLP intermediate channels.

    For each selected channel, the matching gate/up projection rows and down
    projection column are zeroed together. This changes model behavior and gives
    a faithful PPL ablation, but physical tensor slicing is still needed for
    wall-clock speedup.
    """
    layers = transformer_layers(model)
    with torch.no_grad():
        for layer_id, raw_indices in indices_by_layer.items():
            mlp = layers[int(layer_id)].mlp
            indices = torch.as_tensor(
                list(raw_indices),
                dtype=torch.long,
                device=mlp.down_proj.weight.device,
            )
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
