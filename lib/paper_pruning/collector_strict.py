from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


CACHE_VERSION = 7
UNIT_TYPES = ("mlp", "attention")


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
    base_sample_ids: np.ndarray
    losses: np.ndarray
    position_losses: np.ndarray

    def layer_path(self, layer_id: int, unit_type: str = "mlp") -> Path:
        if unit_type not in UNIT_TYPES:
            raise ValueError(f"unknown unit type: {unit_type}")
        return self.root / f"layer_{int(layer_id):03d}_{unit_type}_responses.npy"

    def load_layer(self, layer_id: int, unit_type: str = "mlp", mmap_mode: str = "r") -> np.ndarray:
        return np.load(self.layer_path(layer_id, unit_type), mmap_mode=mmap_mode)


def parse_scenario_ratios(raw: str | Sequence[float]) -> List[float]:
    if isinstance(raw, str):
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [float(value) for value in raw]
    if not values or any(not 0 < value <= 1 for value in values):
        raise ValueError("scenario ratios must be values in (0,1]")
    return sorted(set(values))


def _quantile_events(values: np.ndarray, bins: int) -> np.ndarray:
    """Unified discrete task-event scale from position-derived next-token NLL."""
    if bins < 2:
        raise ValueError("event_bins must be >= 2")
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < bins:
        order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
        labels = np.minimum((order * bins) // max(1, x.size), bins - 1)
    else:
        quantiles = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
        labels = np.digitize(x, np.unique(quantiles), right=False)
    labels = labels.astype(np.int64, copy=False)
    if np.unique(labels).size < 2:
        labels = (x > float(np.median(x))).astype(np.int64)
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
    return model.get_input_embeddings().weight.device


def attention_layout(layer: torch.nn.Module, model: torch.nn.Module) -> Dict[str, int]:
    """Return a structural attention layout.

    A structural attention unit is one KV head plus every query head that shares
    that KV head. For ordinary MHA this degenerates to one Q/K/V/O head.
    """
    attn = layer.self_attn
    cfg = getattr(model, "config", None)
    num_heads = int(
        getattr(attn, "num_heads", 0)
        or getattr(cfg, "num_attention_heads", 0)
        or 0
    )
    num_kv_heads = int(
        getattr(attn, "num_key_value_heads", 0)
        or getattr(cfg, "num_key_value_heads", 0)
        or num_heads
    )
    head_dim = int(getattr(attn, "head_dim", 0) or 0)
    q_out = int(attn.q_proj.out_features)

    if num_heads <= 0:
        if head_dim <= 0:
            raise AttributeError("unable to infer attention head count")
        num_heads = q_out // head_dim
    if head_dim <= 0:
        if q_out % num_heads:
            raise ValueError("q_proj output dimension is not divisible by num_heads")
        head_dim = q_out // num_heads
    if num_kv_heads <= 0 or num_heads % num_kv_heads:
        raise ValueError(
            f"attention Q/KV layout must have an integer sharing group, got {num_heads}/{num_kv_heads}"
        )
    if q_out != num_heads * head_dim:
        raise ValueError("q_proj output dimension does not match inferred attention layout")

    return {
        "num_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "query_heads_per_kv": num_heads // num_kv_heads,
        "structural_units": num_kv_heads,
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
            f"response cache version {version} is incompatible with strict structured collector "
            f"(expected {CACHE_VERSION}); rebuild with --paper_overwrite_cache"
        )

    unit_counts = {
        unit_type: {int(k): int(v) for k, v in layer_map.items()}
        for unit_type, layer_map in metadata["unit_counts"].items()
    }
    layouts = {
        int(k): {name: int(value) for name, value in layout.items()}
        for k, layout in metadata["attention_layouts"].items()
    }
    cache = ResponseCache(
        root=root,
        num_observations=int(metadata["num_observations"]),
        num_layers=int(metadata["num_layers"]),
        response_length=int(metadata["response_length"]),
        unit_counts=unit_counts,
        attention_layouts=layouts,
        events=np.load(root / "events.npy"),
        scenario_ids=np.load(root / "scenario_ids.npy"),
        base_sample_ids=np.load(root / "base_sample_ids.npy"),
        losses=np.load(root / "losses.npy"),
        position_losses=np.load(root / "position_losses.npy"),
    )
    for unit_type in UNIT_TYPES:
        for layer_id in range(cache.num_layers):
            if not cache.layer_path(layer_id, unit_type).exists():
                raise FileNotFoundError(cache.layer_path(layer_id, unit_type))
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
    """Collect the paper's task-loss gradient responses.

    MLP: |dL/dA_j(t)| at the input of down_proj.
    Attention: L2 norm of dL/dO over the complete structural head bundle.
    No weight magnitude or Wanda activation-scale statistic is collected here.
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
    layouts: Dict[int, Dict[str, int]] = {}
    maps: Dict[str, Dict[int, np.memmap]] = {"mlp": {}, "attention": {}}

    for layer_id, layer in enumerate(layers):
        mlp_count = int(layer.mlp.down_proj.in_features)
        layout = attention_layout(layer, model)
        attention_count = int(layout["structural_units"])
        unit_counts["mlp"][layer_id] = mlp_count
        unit_counts["attention"][layer_id] = attention_count
        layouts[layer_id] = layout
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
            shape=(observations, attention_count, response_length),
        )

    losses = np.zeros(observations, dtype=np.float64)
    position_losses = np.zeros((observations, response_length), dtype=np.float32)
    scenario_ids = np.zeros(observations, dtype=np.int64)
    base_sample_ids = np.zeros(observations, dtype=np.int64)
    written = {
        unit_type: np.zeros((observations, num_layers), dtype=bool)
        for unit_type in UNIT_TYPES
    }
    state = {"row": -1}
    handles = []

    def register_response_hook(module, layer_id, unit_type, transform):
        def pre_hook(_module, inputs):
            activation = inputs[0]
            if not activation.requires_grad:
                raise RuntimeError(
                    f"{unit_type} response tensor has no gradient; enable input gradients"
                )
            row = int(state["row"])

            def capture(gradient):
                sequence_response = transform(gradient.detach().float())
                pooled = F.adaptive_avg_pool1d(
                    sequence_response, response_length
                ).mean(dim=0)
                maps[unit_type][layer_id][row] = (
                    pooled.cpu().numpy().astype(np.float16, copy=False)
                )
                written[unit_type][row, layer_id] = True
                return gradient

            activation.register_hook(capture)
            return None

        handles.append(module.register_forward_pre_hook(pre_hook))

    for layer_id, layer in enumerate(layers):
        register_response_hook(
            layer.mlp.down_proj,
            layer_id,
            "mlp",
            lambda gradient: gradient.abs().transpose(1, 2),
        )

        layout = layouts[layer_id]
        num_heads = layout["num_heads"]
        num_kv_heads = layout["num_key_value_heads"]
        q_per_kv = layout["query_heads_per_kv"]
        head_dim = layout["head_dim"]

        def attention_transform(
            gradient,
            h=num_heads,
            kv=num_kv_heads,
            group=q_per_kv,
            d=head_dim,
        ):
            if gradient.shape[-1] != h * d:
                raise ValueError("o_proj input width does not match attention layout")
            g = gradient.reshape(gradient.shape[0], gradient.shape[1], kv, group, d)
            response = torch.sqrt(torch.clamp(g.square().sum(dim=(-1, -2)), min=0.0))
            return response.transpose(1, 2)

        register_response_hook(
            layer.self_attn.o_proj,
            layer_id,
            "attention",
            attention_transform,
        )

    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = False

    original_requires_grad = [p.requires_grad for p in model.parameters()]
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

                shift_logits = outputs.logits[:, :-1, :].float().contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                token_nll = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    reduction="none",
                ).reshape(shift_labels.shape)

                pooled_nll = F.adaptive_avg_pool1d(
                    token_nll.unsqueeze(1), response_length
                ).squeeze(1).mean(dim=0)
                position_losses[row] = pooled_nll.detach().cpu().numpy().astype(np.float32)
                losses[row] = float(pooled_nll.mean().detach().cpu())
                scenario_ids[row] = scenario_id
                base_sample_ids[row] = sample_index

                loss.backward()
                for unit_type in UNIT_TYPES:
                    if not written[unit_type][row].all():
                        missing = np.flatnonzero(~written[unit_type][row]).tolist()
                        raise RuntimeError(
                            f"{unit_type} gradient response missing for layers {missing} at observation {row}"
                        )
                row += 1
                print(f"[strict paper response] observation {row}/{observations}", flush=True)
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

    # One event per sample-scenario observation. It is derived from the complete
    # position-level next-token NLL vector, which is also retained on disk.
    events = _quantile_events(position_losses.mean(axis=1), event_bins)
    np.save(root / "events.npy", events)
    np.save(root / "scenario_ids.npy", scenario_ids)
    np.save(root / "base_sample_ids.npy", base_sample_ids)
    np.save(root / "losses.npy", losses)
    np.save(root / "position_losses.npy", position_losses)

    metadata = {
        "cache_version": CACHE_VERSION,
        "num_observations": observations,
        "num_layers": num_layers,
        "response_length": response_length,
        "unit_counts": {
            t: {str(k): v for k, v in layer_map.items()}
            for t, layer_map in unit_counts.items()
        },
        "attention_layouts": {str(k): v for k, v in layouts.items()},
        "scenario_ratios": ratios,
        "event_bins": event_bins,
        "response_definitions": {
            "mlp": "abs(d(causal_lm_loss)/d(down_proj_input))",
            "attention": "L2 gradient norm over one KV head and all query heads sharing it",
        },
        "task_event_definition": "global quantile bins of mean pooled position-level next-token NLL",
        "wanda_statistics": "none",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return load_response_cache(root)
