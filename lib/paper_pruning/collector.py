from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


CACHE_VERSION = 9
UNIT_TYPES = ("mlp", "attention")


@dataclass(frozen=True)
class ResponseCache:
    """Native-length gradient responses after exact DCT and fine-bin energy.

    Raw response sequences can have different lengths across scenarios. The
    formula only needs their fine-bin energy after DCT, so the cache stores
    ``[observations, units, fine_bins]`` and never resamples the sequence axis.
    """

    root: Path
    num_observations: int
    num_layers: int
    fine_bins: int
    unit_counts: Dict[str, Dict[int, int]]
    attention_layouts: Dict[int, Dict[str, int]]
    events: np.ndarray
    scenario_ids: np.ndarray
    base_sample_ids: np.ndarray
    losses: np.ndarray
    position_losses: np.ndarray
    position_events: np.ndarray
    position_offsets: np.ndarray
    native_sequence_lengths: np.ndarray
    scenario_definitions: tuple[dict, ...]
    observation_metadata: tuple[dict, ...]
    cache_signature: dict

    def layer_path(self, layer_id: int, unit_type: str = "mlp") -> Path:
        if unit_type not in UNIT_TYPES:
            raise ValueError(f"unknown unit type: {unit_type}")
        return self.root / f"layer_{int(layer_id):03d}_{unit_type}_fine_energy.npy"

    def load_layer(self, layer_id: int, unit_type: str = "mlp", mmap_mode: str = "r") -> np.ndarray:
        return np.load(self.layer_path(layer_id, unit_type), mmap_mode=mmap_mode)

    def position_loss_slice(self, observation: int) -> np.ndarray:
        start, end = self.position_offsets[observation : observation + 2]
        return self.position_losses[int(start) : int(end)]

    def position_event_slice(self, observation: int) -> np.ndarray:
        start, end = self.position_offsets[observation : observation + 2]
        return self.position_events[int(start) : int(end)]


def parse_scenario_ratios(raw: str | Sequence[float]) -> List[float]:
    values = (
        [float(part.strip()) for part in raw.split(",") if part.strip()]
        if isinstance(raw, str)
        else [float(value) for value in raw]
    )
    if not values or any(not 0 < value <= 1 for value in values):
        raise ValueError("scenario ratios must be values in (0,1]")
    return sorted(set(values))


def parse_scenario_crops(raw: str | Sequence[str]) -> List[str]:
    values = (
        [part.strip().lower() for part in raw.split(",") if part.strip()]
        if isinstance(raw, str)
        else [str(value).strip().lower() for value in raw]
    )
    aliases = {"left": "prefix", "middle": "center", "right": "suffix"}
    values = [aliases.get(value, value) for value in values]
    allowed = {"prefix", "center", "suffix"}
    if not values or any(value not in allowed for value in values):
        raise ValueError("scenario crops must contain prefix, center and/or suffix")
    return list(dict.fromkeys(values))


def _crop_input(base_ids: torch.Tensor, length: int, crop: str) -> torch.Tensor:
    total = int(base_ids.shape[1])
    length = min(max(8, int(length)), total)
    if crop == "prefix":
        start = 0
    elif crop == "center":
        start = max(0, (total - length) // 2)
    elif crop == "suffix":
        start = total - length
    else:
        raise ValueError(f"unknown scenario crop: {crop}")
    return base_ids[:, start : start + length]


def _normalize_signature(signature: Mapping | None) -> dict:
    if signature is None:
        return {}
    return json.loads(json.dumps(dict(signature), ensure_ascii=False, sort_keys=True, default=str))


def _quantile_events(values: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2:
        raise ValueError("event_bins must be >= 2")
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0:
        raise ValueError("cannot build events from an empty array")
    if data.size < bins:
        order = np.argsort(np.argsort(data, kind="stable"), kind="stable")
        labels = np.minimum((order * bins) // max(1, data.size), bins - 1)
    else:
        edges = np.unique(np.quantile(data, np.linspace(0, 1, bins + 1)[1:-1]))
        labels = np.digitize(data, edges, right=False)
    labels = labels.astype(np.int64, copy=False)
    if np.unique(labels).size < 2:
        labels = (data > float(np.median(data))).astype(np.int64)
    if np.unique(labels).size < 2 and labels.size > 1:
        labels[1::2] = 1
    return labels


def _transformer_layers(model: torch.nn.Module):
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise AttributeError("expected transformer layers at model.model.layers or model.layers")
    return layers


def _input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def _attention_layout(layer: torch.nn.Module, model: torch.nn.Module) -> Dict[str, int]:
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        raise AttributeError("transformer layer has no self_attn")
    config = getattr(model, "config", None)
    num_heads = int(getattr(attn, "num_heads", 0) or getattr(config, "num_attention_heads", 0) or 0)
    num_kv_heads = int(
        getattr(attn, "num_key_value_heads", 0)
        or getattr(config, "num_key_value_heads", 0)
        or num_heads
    )
    head_dim = int(getattr(attn, "head_dim", 0) or 0)
    q_out = int(attn.q_proj.out_features)
    if num_heads <= 0 and head_dim > 0:
        num_heads = q_out // head_dim
    if head_dim <= 0 and num_heads > 0:
        if q_out % num_heads:
            raise ValueError("q_proj output is not divisible by the number of heads")
        head_dim = q_out // num_heads
    if num_heads <= 0 or head_dim <= 0 or q_out != num_heads * head_dim:
        raise ValueError("unable to infer a valid attention-head layout")
    return {
        "num_heads": num_heads,
        "num_key_value_heads": num_kv_heads or num_heads,
        "head_dim": head_dim,
    }


def _batch_record(batch, sample_index: int) -> tuple[torch.Tensor, object, object, dict]:
    if isinstance(batch, Mapping):
        ids = torch.as_tensor(batch["input_ids"], dtype=torch.long)
        base_id = batch.get("base_sample_id", sample_index)
        scenario_id = batch.get("scenario_id", "dataset")
        metadata = dict(batch.get("metadata", {}))
    else:
        ids = batch[0] if isinstance(batch, (tuple, list)) else batch
        ids = torch.as_tensor(ids, dtype=torch.long)
        base_id = sample_index
        scenario_id = "dataset"
        metadata = {}
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("each calibration item must contain one token sequence")
    return ids, base_id, scenario_id, metadata


def _torch_proposal_dct(sequence: torch.Tensor, eps: float) -> torch.Tensor:
    """Exact displayed DCT-II cosine sum, computed with an FFT identity."""
    x = sequence.float()
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    standardized = (x - mean) / (std + eps)
    length = standardized.shape[-1]
    reordered = torch.cat(
        [standardized[..., ::2], torch.flip(standardized[..., 1::2], dims=[-1])],
        dim=-1,
    )
    spectrum = torch.fft.fft(reordered, dim=-1)
    frequency = torch.arange(length, device=x.device, dtype=x.dtype)
    phase = torch.exp(-1j * torch.pi * frequency / (2 * length))
    return (spectrum * phase).real


def _torch_fine_energy(sequence: torch.Tensor, fine_bins: int, eps: float) -> torch.Tensor:
    coefficients = _torch_proposal_dct(sequence, eps)
    length = coefficients.shape[-1]
    bin_count = min(int(fine_bins), length)
    quotient, remainder = divmod(length, bin_count)
    sizes = [quotient + (index < remainder) for index in range(bin_count)]
    result = []
    start = 0
    for size in sizes:
        end = start + size
        result.append(coefficients[..., start:end].square().sum(dim=-1))
        start = end
    return torch.stack(result, dim=-1)


def load_response_cache(cache_dir: str | Path, expected_signature: Mapping | None = None) -> ResponseCache:
    root = Path(cache_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    version = int(metadata.get("cache_version", 0))
    if version != CACHE_VERSION:
        raise ValueError(
            f"response cache version {version} is incompatible with version {CACHE_VERSION}; rebuild the cache"
        )
    stored_signature = _normalize_signature(metadata.get("cache_signature", {}))
    requested_signature = _normalize_signature(expected_signature)
    if requested_signature and stored_signature != requested_signature:
        raise ValueError("response cache signature does not match model/tokenizer/data/scenario settings")

    unit_counts = {
        unit_type: {int(layer): int(count) for layer, count in layer_map.items()}
        for unit_type, layer_map in metadata["unit_counts"].items()
    }
    attention_layouts = {
        int(layer): {name: int(value) for name, value in layout.items()}
        for layer, layout in metadata["attention_layouts"].items()
    }
    cache = ResponseCache(
        root=root,
        num_observations=int(metadata["num_observations"]),
        num_layers=int(metadata["num_layers"]),
        fine_bins=int(metadata["fine_bins"]),
        unit_counts=unit_counts,
        attention_layouts=attention_layouts,
        events=np.load(root / "events.npy"),
        scenario_ids=np.load(root / "scenario_ids.npy"),
        base_sample_ids=np.load(root / "base_sample_ids.npy"),
        losses=np.load(root / "losses.npy"),
        position_losses=np.load(root / "position_losses.npy"),
        position_events=np.load(root / "position_events.npy"),
        position_offsets=np.load(root / "position_offsets.npy"),
        native_sequence_lengths=np.load(root / "native_sequence_lengths.npy"),
        scenario_definitions=tuple(metadata.get("scenario_definitions", [])),
        observation_metadata=tuple(metadata.get("observation_metadata", [])),
        cache_signature=stored_signature,
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
    scenario_crops: str | Sequence[str] = ("prefix", "center", "suffix"),
    fine_bins: int = 16,
    event_bins: int = 3,
    eps: float = 1e-8,
    overwrite: bool = False,
    cache_signature: Mapping | None = None,
) -> ResponseCache:
    """Collect native-length task-gradient responses and exact fine-bin energy.

    No sequence pooling or interpolation is applied. Every sample/scenario is
    standardized and transformed at its own native context length; only the
    final contiguous frequency-bin sums have a common dimension.
    """
    root = Path(cache_dir)
    if (root / "metadata.json").exists() and not overwrite:
        return load_response_cache(root, expected_signature=cache_signature)
    root.mkdir(parents=True, exist_ok=True)

    ratios = parse_scenario_ratios(scenario_ratios)
    crops = parse_scenario_crops(scenario_crops)
    transforms = [(ratio, crop) for ratio in ratios for crop in crops]
    if fine_bins < 2:
        raise ValueError("fine_bins must be >= 2")
    layers = _transformer_layers(model)
    num_layers = len(layers)
    observations = len(dataloader) * len(transforms)
    if observations < 4:
        raise ValueError("at least four sample/scenario observations are required")

    unit_counts: Dict[str, Dict[int, int]] = {"mlp": {}, "attention": {}}
    attention_layouts: Dict[int, Dict[str, int]] = {}
    maps: Dict[str, Dict[int, np.memmap]] = {"mlp": {}, "attention": {}}
    for layer_id, layer in enumerate(layers):
        down_proj = getattr(getattr(layer, "mlp", None), "down_proj", None)
        if down_proj is None:
            raise AttributeError(f"layer {layer_id} lacks mlp.down_proj")
        mlp_count = int(down_proj.in_features)
        layout = _attention_layout(layer, model)
        unit_counts["mlp"][layer_id] = mlp_count
        unit_counts["attention"][layer_id] = int(layout["num_heads"])
        attention_layouts[layer_id] = layout
        maps["mlp"][layer_id] = np.lib.format.open_memmap(
            root / f"layer_{layer_id:03d}_mlp_fine_energy.npy",
            mode="w+",
            dtype=np.float32,
            shape=(observations, mlp_count, fine_bins),
        )
        maps["attention"][layer_id] = np.lib.format.open_memmap(
            root / f"layer_{layer_id:03d}_attention_fine_energy.npy",
            mode="w+",
            dtype=np.float32,
            shape=(observations, int(layout["num_heads"]), fine_bins),
        )

    losses = np.zeros(observations, dtype=np.float64)
    scenario_ids = np.zeros(observations, dtype=np.int64)
    base_sample_ids = np.zeros(observations, dtype=np.int64)
    native_sequence_lengths = np.zeros(observations, dtype=np.int32)
    position_loss_rows: list[np.ndarray] = []
    observation_metadata: list[dict] = []
    scenario_key_to_id: dict[str, int] = {}
    base_key_to_id: dict[str, int] = {}
    written = {
        unit_type: np.zeros((observations, num_layers), dtype=bool)
        for unit_type in UNIT_TYPES
    }
    state = {"row": -1}
    handles = []

    def register_response_hook(module, layer_id: int, unit_type: str, transform) -> None:
        def pre_hook(_module, inputs):
            activation = inputs[0]
            if not activation.requires_grad:
                raise RuntimeError("response tensor has no gradient; enable input gradients")
            row = int(state["row"])

            def capture(gradient):
                sequence_response = transform(gradient.detach().float())
                fine_energy = _torch_fine_energy(sequence_response, fine_bins, eps).mean(dim=0)
                maps[unit_type][layer_id][row] = fine_energy.cpu().numpy().astype(np.float32, copy=False)
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
        layout = attention_layouts[layer_id]
        head_count, head_dim = layout["num_heads"], layout["head_dim"]

        def attention_transform(gradient, heads=head_count, dimension=head_dim):
            if gradient.shape[-1] != heads * dimension:
                raise ValueError("attention output-gradient width does not match the head layout")
            response = gradient.reshape(gradient.shape[0], gradient.shape[1], heads, dimension)
            return torch.linalg.vector_norm(response, ord=2, dim=-1).transpose(1, 2)

        register_response_hook(layer.self_attn.o_proj, layer_id, "attention", attention_transform)

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
        handles.append(embedding.register_forward_hook(lambda _m, _i, output: output.requires_grad_(True)))

    model.eval()
    device = _input_device(model)
    row = 0
    try:
        for sample_index, batch in enumerate(dataloader):
            base_ids, raw_base_id, raw_scenario_id, metadata = _batch_record(batch, sample_index)
            base_key = json.dumps(raw_base_id, ensure_ascii=False, sort_keys=True, default=str)
            if base_key not in base_key_to_id:
                base_key_to_id[base_key] = len(base_key_to_id)
            encoded_base_id = base_key_to_id[base_key]
            for ratio, crop in transforms:
                length = max(8, int(round(base_ids.shape[1] * ratio)))
                input_ids = _crop_input(base_ids, length, crop).to(device)
                if input_ids.shape[1] < fine_bins:
                    raise ValueError(
                        f"scenario sequence length {input_ids.shape[1]} is smaller than fine_bins={fine_bins}; "
                        "reduce the number of fine bins or use longer scenario inputs"
                    )
                composite_scenario = {
                    "source": raw_scenario_id,
                    "ratio": float(ratio),
                    "crop": crop,
                    **metadata,
                }
                scenario_key = json.dumps(composite_scenario, ensure_ascii=False, sort_keys=True, default=str)
                if scenario_key not in scenario_key_to_id:
                    scenario_key_to_id[scenario_key] = len(scenario_key_to_id)
                encoded_scenario_id = scenario_key_to_id[scenario_key]

                state["row"] = row
                model.zero_grad(set_to_none=True)
                outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                shift_logits = outputs.logits[:, :-1, :].float().contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                token_nll = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    reduction="none",
                ).reshape(shift_labels.shape)
                token_values = token_nll.squeeze(0).detach().cpu().numpy().astype(np.float32)
                position_loss_rows.append(token_values)
                losses[row] = float(token_values.mean())
                native_sequence_lengths[row] = int(input_ids.shape[1])
                scenario_ids[row] = encoded_scenario_id
                base_sample_ids[row] = encoded_base_id
                observation_metadata.append(
                    {
                        "observation": row,
                        "base_sample_id": raw_base_id,
                        "scenario_id": raw_scenario_id,
                        "scenario_index": encoded_scenario_id,
                        "ratio": float(ratio),
                        "crop": crop,
                        "sequence_length": int(input_ids.shape[1]),
                        **metadata,
                    }
                )
                outputs.loss.backward()
                for unit_type in UNIT_TYPES:
                    if not written[unit_type][row].all():
                        missing = np.flatnonzero(~written[unit_type][row]).tolist()
                        raise RuntimeError(f"missing {unit_type} gradient responses in layers {missing}")
                row += 1
                print(f"[response] observation {row}/{observations}", flush=True)
                del outputs, shift_logits, shift_labels, token_nll, input_ids
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

    events = _quantile_events(losses, event_bins)
    position_offsets = np.zeros(observations + 1, dtype=np.int64)
    for index, values in enumerate(position_loss_rows):
        position_offsets[index + 1] = position_offsets[index] + values.size
    position_losses = np.concatenate(position_loss_rows).astype(np.float32, copy=False)
    position_events = _quantile_events(position_losses, event_bins)
    np.save(root / "events.npy", events)
    np.save(root / "scenario_ids.npy", scenario_ids)
    np.save(root / "base_sample_ids.npy", base_sample_ids)
    np.save(root / "losses.npy", losses)
    np.save(root / "position_losses.npy", position_losses)
    np.save(root / "position_events.npy", position_events)
    np.save(root / "position_offsets.npy", position_offsets)
    np.save(root / "native_sequence_lengths.npy", native_sequence_lengths)

    scenario_definitions = [
        {"scenario_index": value, **json.loads(key)}
        for key, value in scenario_key_to_id.items()
    ]
    metadata = {
        "cache_version": CACHE_VERSION,
        "num_observations": observations,
        "num_layers": num_layers,
        "fine_bins": fine_bins,
        "unit_counts": {
            unit_type: {str(layer): count for layer, count in layer_map.items()}
            for unit_type, layer_map in unit_counts.items()
        },
        "attention_layouts": {str(layer): layout for layer, layout in attention_layouts.items()},
        "scenario_ratios": ratios,
        "scenario_crops": crops,
        "scenario_definitions": scenario_definitions,
        "observation_metadata": observation_metadata,
        "event_bins": event_bins,
        "cache_signature": _normalize_signature(cache_signature),
        "response_definitions": {
            "mlp": "abs(d(causal_lm_loss)/d(ffn_intermediate_activation_j))",
            "attention": "l2_head_dim(d(causal_lm_loss)/d(attention_head_output))",
        },
        "frequency_processing": (
            "native-length per-sample standardization; displayed unnormalised DCT-II; "
            "squared coefficients; contiguous fine-bin sums; no sequence resampling"
        ),
        "task_event_definition": "quantile_bin(mean(position_level_next_token_nll))",
        "position_event_definition": "quantile_bin(position_level_next_token_nll)",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_response_cache(root, expected_signature=cache_signature)
