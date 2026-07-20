from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ResponseCache:
    root: Path
    num_observations: int
    num_layers: int
    response_length: int
    unit_counts: Dict[int, int]
    events: np.ndarray
    scenario_ids: np.ndarray
    losses: np.ndarray

    def layer_path(self, layer_id: int) -> Path:
        return self.root / f"layer_{int(layer_id):03d}_responses.npy"

    def load_layer(self, layer_id: int, mmap_mode: str = "r") -> np.ndarray:
        return np.load(self.layer_path(layer_id), mmap_mode=mmap_mode)


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


def load_response_cache(cache_dir: str | Path) -> ResponseCache:
    root = Path(cache_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    events = np.load(root / "events.npy")
    scenarios = np.load(root / "scenario_ids.npy")
    losses = np.load(root / "losses.npy")
    unit_counts = {int(key): int(value) for key, value in metadata["unit_counts"].items()}
    cache = ResponseCache(
        root=root,
        num_observations=int(metadata["num_observations"]),
        num_layers=int(metadata["num_layers"]),
        response_length=int(metadata["response_length"]),
        unit_counts=unit_counts,
        events=events,
        scenario_ids=scenarios,
        losses=losses,
    )
    for layer_id in range(cache.num_layers):
        if not cache.layer_path(layer_id).exists():
            raise FileNotFoundError(cache.layer_path(layer_id))
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
    """Collect r_i(t)=z_i(t)*dL/dz_i(t) for every MLP channel and layer.

    ``z`` is the input to ``mlp.down_proj``, i.e. the post-gating MLP channel
    activation. Responses are adaptively pooled along sequence position before
    being written as float16 memory maps, keeping CPU memory bounded.
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

    unit_counts: Dict[int, int] = {}
    maps: Dict[int, np.memmap] = {}
    for layer_id, layer in enumerate(layers):
        down_proj = getattr(getattr(layer, "mlp", None), "down_proj", None)
        if down_proj is None or not hasattr(down_proj, "in_features"):
            raise AttributeError(f"layer {layer_id} has no Llama/Mistral-style mlp.down_proj")
        unit_count = int(down_proj.in_features)
        unit_counts[layer_id] = unit_count
        maps[layer_id] = np.lib.format.open_memmap(
            root / f"layer_{layer_id:03d}_responses.npy",
            mode="w+",
            dtype=np.float16,
            shape=(observations, unit_count, response_length),
        )

    losses = np.zeros(observations, dtype=np.float64)
    scenario_ids = np.zeros(observations, dtype=np.int64)
    written = np.zeros((observations, num_layers), dtype=bool)
    state = {"row": -1}
    handles = []

    for layer_id, layer in enumerate(layers):
        down_proj = layer.mlp.down_proj

        def make_pre_hook(current_layer: int):
            def pre_hook(_module, inputs):
                activation = inputs[0]
                if not activation.requires_grad:
                    raise RuntimeError(
                        "MLP response tensor has no gradient. The collector requires "
                        "model.enable_input_require_grads() or trainable embeddings."
                    )
                row = int(state["row"])
                detached_activation = activation.detach()

                def capture(gradient):
                    response = (detached_activation.float() * gradient.detach().float()).transpose(1, 2)
                    pooled = F.adaptive_avg_pool1d(response, response_length).mean(dim=0)
                    maps[current_layer][row] = pooled.cpu().numpy().astype(np.float16, copy=False)
                    written[row, current_layer] = True
                    return gradient

                activation.register_hook(capture)
                return None

            return pre_hook

        handles.append(down_proj.register_forward_pre_hook(make_pre_hook(layer_id)))

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
                if not written[row].all():
                    missing = np.flatnonzero(~written[row]).tolist()
                    raise RuntimeError(f"gradient response missing for layers {missing} at observation {row}")
                del outputs, loss, input_ids
                row += 1
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

    for mmap in maps.values():
        mmap.flush()
    events = _quantile_events(losses, scenario_ids, event_bins)
    np.save(root / "events.npy", events)
    np.save(root / "scenario_ids.npy", scenario_ids)
    np.save(root / "losses.npy", losses)
    metadata = {
        "num_observations": observations,
        "num_layers": num_layers,
        "response_length": response_length,
        "unit_counts": {str(key): value for key, value in unit_counts.items()},
        "scenario_ratios": ratios,
        "event_bins": event_bins,
        "response_definition": "down_proj_input * d(causal_lm_loss)/d(down_proj_input)",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_response_cache(root)
