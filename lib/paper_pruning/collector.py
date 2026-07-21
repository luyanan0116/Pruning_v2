from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


CACHE_VERSION = 3
UNIT_TYPES = ("mlp", "attention")
LINEAR_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass(frozen=True)
class ResponseCache:
    root: Path
    num_observations: int
    num_layers: int
    response_length: int
    unit_counts: Dict[str, Dict[int, int]]
    attention_layouts: Dict[int, Dict[str, int]]
    events: np.ndarray
    scenario_ids: np.ndarray
    losses: np.ndarray
    linear_modules: tuple[str, ...]

    def layer_path(self, layer_id: int, unit_type: str = "mlp") -> Path:
        if unit_type not in UNIT_TYPES:
            raise ValueError(f"unknown unit type: {unit_type}")
        return self.root / f"layer_{int(layer_id):03d}_{unit_type}_responses.npy"

    def load_layer(self, layer_id: int, unit_type: str = "mlp", mmap_mode: str = "r") -> np.ndarray:
        return np.load(self.layer_path(layer_id, unit_type), mmap_mode=mmap_mode)

    def activation_scale_path(self, layer_id: int, module_name: str) -> Path:
        if module_name not in self.linear_modules:
            raise ValueError(f"unknown cached linear module: {module_name}")
        safe_name = module_name.replace(".", "__")
        return self.root / f"layer_{int(layer_id):03d}_{safe_name}_input_scale.npy"

    def load_activation_scale(self, layer_id: int, module_name: str, mmap_mode: str = "r") -> np.ndarray:
        return np.load(self.activation_scale_path(layer_id, module_name), mmap_mode=mmap_mode)


def parse_scenario_ratios(raw: str | Sequence[float]) -> List[float]:
    if isinstance(raw, str):
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [float(value) for value in raw]
    if not values or any(not 0 < value <= 1 for value in values):
        raise ValueError("scenario ratios must be comma-separated values in (0,1]")
    return sorted(set(values))


def _quantile_events(losses: np.ndarray, scenario_ids: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2:
        raise ValueError("event_bins must be >= 2")
    labels = np.zeros(losses.size, dtype=np.int64)
    for scenario in np.unique(scenario_ids):
        indices = np.flatnonzero(scenario_ids == scenario)
        values = losses[indices]
        if indices.size < bins:
            order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
            labels[indices] = np.minimum((order * bins) // max(1, indices.size), bins - 1)
            continue
        quantiles = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
        edges = np.unique(quantiles)
        labels[indices] = np.digitize(values, edges, right=False)
    if np.unique(labels).size < 2:
        median = float(np.median(losses))
        labels = (losses > median).astype(np.int64)
    if np.unique(labels).size < 2:
        labels[np.arange(labels.size) % 2 == 1] = 1
    return labels


def _transformer_layers(model: torch.nn.Module):
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise AttributeError("expected transformer layers at model.model.layers or model.layers")
    return layers


def _input_device(model: torch.nn.Module) -> torch.device:
    embedding = model.get_input_embeddings()
    return embedding.weight.device


def _attention_layout(layer: torch.nn.Module, model: torch.nn.Module) -> Dict[str, int]:
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        raise AttributeError("transformer layer has no self_attn module")
    config = getattr(model, "config", None)
    num_heads = int(
        getattr(attn, "num_heads", 0)
        or getattr(config, "num_attention_heads", 0)
        or 0
    )
    num_kv_heads = int(
        getattr(attn, "num_key_value_heads", 0)
        or getattr(config, "num_key_value_heads", 0)
        or num_heads
    )
    head_dim = int(getattr(attn, "head_dim", 0) or 0)
    q_out = int(attn.q_proj.out_features)
    if num_heads <= 0:
        if head_dim <= 0:
            raise AttributeError("unable to infer attention head count")
        num_heads = q_out // head_dim
    if head_dim <= 0:
        if q_out % num_heads != 0:
            raise ValueError("q_proj output dimension is not divisible by num_heads")
        head_dim = q_out // num_heads
    if num_kv_heads <= 0:
        num_kv_heads = num_heads
    if q_out != num_heads * head_dim:
        raise ValueError(
            f"attention layout mismatch: q_proj.out_features={q_out}, "
            f"num_heads={num_heads}, head_dim={head_dim}"
        )
    return {
        "num_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def load_response_cache(cache_dir: str | Path) -> ResponseCache:
    root = Path(cache_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    version = int(metadata.get("cache_version", 1))
    if version != CACHE_VERSION:
        raise ValueError(
            f"response cache version {version} is incompatible with the MLP+attention "
            f"collector (expected {CACHE_VERSION}). Re-run with --paper_overwrite_cache."
        )
    events = np.load(root / "events.npy")
    scenarios = np.load(root / "scenario_ids.npy")
    losses = np.load(root / "losses.npy")
    raw_counts = metadata["unit_counts"]
    unit_counts = {
        unit_type: {int(key): int(value) for key, value in layer_map.items()}
        for unit_type, layer_map in raw_counts.items()
    }
    attention_layouts = {
        int(key): {name: int(value) for name, value in layout.items()}
        for key, layout in metadata["attention_layouts"].items()
    }
    cache = ResponseCache(
        root=root,
        num_observations=int(metadata["num_observations"]),
        num_layers=int(metadata["num_layers"]),
        response_length=int(metadata["response_length"]),
        unit_counts=unit_counts,
        attention_layouts=attention_layouts,
        events=events,
        scenario_ids=scenarios,
        losses=losses,
        linear_modules=tuple(metadata.get("linear_modules", LINEAR_MODULES)),
    )
    for unit_type in UNIT_TYPES:
        for layer_id in range(cache.num_layers):
            if not cache.layer_path(layer_id, unit_type).exists():
                raise FileNotFoundError(cache.layer_path(layer_id, unit_type))
    for layer_id in range(cache.num_layers):
        for module_name in cache.linear_modules:
            if not cache.activation_scale_path(layer_id, module_name).exists():
                raise FileNotFoundError(cache.activation_scale_path(layer_id, module_name))
    return cache


def collect_gradient_response_cache(
    model: torch.nn.Module,
    dataloader: Sequence,
    cache_dir: str | Path,
    scenario_ratios: str | Sequence[float] = (0.5, 0.75, 1.0),
    response_length: int = 32,
    event_bins: int = 3,
    overwrite: bool = False,
) -> ResponseCache:
    """Collect task-gradient responses for MLP channels and attention heads.

    MLP channel response:
        ``r_i(t) = z_i(t) * dL/dz_i(t)`` at the input of ``mlp.down_proj``.

    Attention-head response:
        ``r_h(t) = sum_d z_h,d(t) * dL/dz_h,d(t)`` at the input of
        ``self_attn.o_proj``. Summation over the head dimension turns a complete
        attention head into one structured unit while preserving sequence
        position for the later DCT/MI pipeline.
    """
    root = Path(cache_dir)
    metadata_path = root / "metadata.json"
    if metadata_path.exists() and not overwrite:
        return load_response_cache(root)
    root.mkdir(parents=True, exist_ok=True)

    ratios = parse_scenario_ratios(scenario_ratios)
    if response_length < 8:
        raise ValueError("response_length must be >= 8")
    layers = _transformer_layers(model)
    num_layers = len(layers)
    observations = len(dataloader) * len(ratios)
    if observations < 4:
        raise ValueError("at least four sample/scenario observations are required")

    unit_counts: Dict[str, Dict[int, int]] = {"mlp": {}, "attention": {}}
    attention_layouts: Dict[int, Dict[str, int]] = {}
    maps: Dict[str, Dict[int, np.memmap]] = {"mlp": {}, "attention": {}}
    for layer_id, layer in enumerate(layers):
        down_proj = getattr(getattr(layer, "mlp", None), "down_proj", None)
        if down_proj is None or not hasattr(down_proj, "in_features"):
            raise AttributeError(f"layer {layer_id} has no Llama/Mistral-style mlp.down_proj")
        mlp_count = int(down_proj.in_features)
        layout = _attention_layout(layer, model)
        head_count = int(layout["num_heads"])
        unit_counts["mlp"][layer_id] = mlp_count
        unit_counts["attention"][layer_id] = head_count
        attention_layouts[layer_id] = layout
        maps["mlp"][layer_id] = np.lib.format.open_memmap(
            root / f"layer_{layer_id:03d}_mlp_responses.npy",
            mode="w+",
            dtype=np.float16,
            shape=(observations, mlp_count, response_length),
        )
        maps["attention"][layer_id] = np.lib.format.open_memmap(
            root / f"layer_{layer_id:03d}_attention_responses.npy",
            mode="w+",
            dtype=np.float16,
            shape=(observations, head_count, response_length),
        )

    activation_sums: Dict[int, Dict[str, torch.Tensor]] = {}
    activation_counts: Dict[int, Dict[str, int]] = {}
    for layer_id, layer in enumerate(layers):
        activation_sums[layer_id] = {}
        activation_counts[layer_id] = {}
        for module_name in LINEAR_MODULES:
            module = layer
            for part in module_name.split("."):
                module = getattr(module, part)
            activation_sums[layer_id][module_name] = torch.zeros(
                int(module.in_features), device=module.weight.device, dtype=torch.float32
            )
            activation_counts[layer_id][module_name] = 0

    losses = np.zeros(observations, dtype=np.float64)
    scenario_ids = np.zeros(observations, dtype=np.int64)
    written = {
        unit_type: np.zeros((observations, num_layers), dtype=bool)
        for unit_type in UNIT_TYPES
    }
    state = {"row": -1}
    handles = []

    def register_response_hook(
        module: torch.nn.Module,
        layer_id: int,
        unit_type: str,
        transform,
    ) -> None:
        def pre_hook(_module, inputs):
            activation = inputs[0]
            if not activation.requires_grad:
                raise RuntimeError(
                    f"{unit_type} response tensor has no gradient. The collector requires "
                    "model.enable_input_require_grads() or trainable embeddings."
                )
            row = int(state["row"])
            detached_activation = activation.detach()

            def capture(gradient):
                sequence_response = transform(detached_activation.float(), gradient.detach().float())
                pooled = F.adaptive_avg_pool1d(sequence_response, response_length).mean(dim=0)
                maps[unit_type][layer_id][row] = pooled.cpu().numpy().astype(np.float16, copy=False)
                written[unit_type][row, layer_id] = True
                return gradient

            activation.register_hook(capture)
            return None

        handles.append(module.register_forward_pre_hook(pre_hook))

    def register_activation_hook(module: torch.nn.Module, layer_id: int, module_name: str) -> None:
        def pre_hook(_module, inputs):
            activation = inputs[0].detach().float()
            reduce_dims = tuple(range(activation.ndim - 1))
            activation_sums[layer_id][module_name].add_(activation.square().sum(dim=reduce_dims))
            activation_counts[layer_id][module_name] += int(activation.numel() // activation.shape[-1])
            return None

        handles.append(module.register_forward_pre_hook(pre_hook))

    for layer_id, layer in enumerate(layers):
        for module_name in LINEAR_MODULES:
            module = layer
            for part in module_name.split("."):
                module = getattr(module, part)
            register_activation_hook(module, layer_id, module_name)

        register_response_hook(
            layer.mlp.down_proj,
            layer_id,
            "mlp",
            lambda activation, gradient: (activation * gradient).transpose(1, 2),
        )
        layout = attention_layouts[layer_id]
        num_heads = layout["num_heads"]
        head_dim = layout["head_dim"]

        def attention_transform(activation, gradient, h=num_heads, d=head_dim):
            if activation.shape[-1] != h * d:
                raise ValueError(
                    f"o_proj input width {activation.shape[-1]} does not match {h} heads x {d}"
                )
            response = (activation * gradient).reshape(activation.shape[0], activation.shape[1], h, d)
            return response.sum(dim=-1).transpose(1, 2)

        register_response_hook(
            layer.self_attn.o_proj,
            layer_id,
            "attention",
            attention_transform,
        )

    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = False
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    input_grad_enabled = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        input_grad_enabled = True
    else:
        embedding = model.get_input_embeddings()

        def require_embedding_grad(_module, _inputs, output):
            output.requires_grad_(True)

        handles.append(embedding.register_forward_hook(require_embedding_grad))

    model.eval()
    device = _input_device(model)
    row = 0
    try:
        for sample_index, batch in enumerate(dataloader):
            base_ids = batch[0]
            if base_ids.ndim == 1:
                base_ids = base_ids.unsqueeze(0)
            for scenario_id, ratio in enumerate(ratios):
                length = max(8, int(round(base_ids.shape[1] * ratio)))
                input_ids = base_ids[:, :length].to(device)
                state["row"] = row
                model.zero_grad(set_to_none=True)
                outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                loss = outputs.loss
                losses[row] = float(loss.detach().cpu())
                scenario_ids[row] = scenario_id
                loss.backward()
                for unit_type in UNIT_TYPES:
                    if not written[unit_type][row].all():
                        missing = np.flatnonzero(~written[unit_type][row]).tolist()
                        raise RuntimeError(
                            f"{unit_type} gradient response missing for layers {missing} at observation {row}"
                        )
                del outputs, loss, input_ids
                row += 1
                print(f"[paper response] observation {row}/{observations}", flush=True)
                if torch.cuda.is_available() and row % 8 == 0:
                    torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()
        if input_grad_enabled and hasattr(model, "disable_input_require_grads"):
            model.disable_input_require_grads()
        for parameter, requires_grad in zip(model.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache

    for unit_type in UNIT_TYPES:
        for mmap in maps[unit_type].values():
            mmap.flush()
    for layer_id in range(num_layers):
        for module_name in LINEAR_MODULES:
            count = activation_counts[layer_id][module_name]
            if count <= 0:
                raise RuntimeError(f"no activation statistics collected for layer {layer_id} {module_name}")
            mean_square = (activation_sums[layer_id][module_name] / float(count)).detach().cpu().numpy()
            safe_name = module_name.replace(".", "__")
            np.save(
                root / f"layer_{layer_id:03d}_{safe_name}_input_scale.npy",
                mean_square.astype(np.float32, copy=False),
            )
    events = _quantile_events(losses, scenario_ids, event_bins)
    np.save(root / "events.npy", events)
    np.save(root / "scenario_ids.npy", scenario_ids)
    np.save(root / "losses.npy", losses)
    metadata = {
        "cache_version": CACHE_VERSION,
        "num_observations": observations,
        "num_layers": num_layers,
        "response_length": response_length,
        "unit_counts": {
            unit_type: {str(key): value for key, value in layer_map.items()}
            for unit_type, layer_map in unit_counts.items()
        },
        "attention_layouts": {
            str(key): value for key, value in attention_layouts.items()
        },
        "linear_modules": list(LINEAR_MODULES),
        "scenario_ratios": ratios,
        "event_bins": event_bins,
        "response_definitions": {
            "mlp": "down_proj_input * d(causal_lm_loss)/d(down_proj_input)",
            "attention": "sum_head_dim(o_proj_input * d(causal_lm_loss)/d(o_proj_input))",
        },
        "activation_scale_definition": "mean over calibration tokens of linear-module input squared",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_response_cache(root)
