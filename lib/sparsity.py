from __future__ import annotations

import torch
import torch.nn as nn


def find_linear_layers(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        return {name: module}
    result = {}
    for child_name, child in module.named_children():
        prefix = f"{name}.{child_name}" if name else child_name
        result.update(find_linear_layers(child, prefix))
    return result


def check_transformer_weight_sparsity(model: nn.Module) -> float:
    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = False
    layers = getattr(getattr(model, "model", model), "layers")
    zero_count = 0
    total_count = 0
    try:
        for layer_id, layer in enumerate(layers):
            subset = find_linear_layers(layer)
            layer_zero = 0
            layer_total = 0
            for module in subset.values():
                weight = module.weight.data
                count = int((weight == 0).sum().item())
                layer_zero += count
                layer_total += int(weight.numel())
                zero_count += count
                total_count += int(weight.numel())
            print(f"layer {layer_id} sparsity {layer_zero / max(1, layer_total):.6f}")
    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
    return float(zero_count) / float(max(1, total_count))
