from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from .weight_budget import WeightBudget, _attention_layout, _layers


TARGET_MODULE_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass
class ActivationStats:
    scaler_row: torch.Tensor
    nsamples: int = 0

    @classmethod
    def for_linear(cls, layer: nn.Linear) -> "ActivationStats":
        return cls(torch.zeros(layer.weight.shape[1], dtype=torch.float32, device=layer.weight.device), 0)

    def add_batch(self, inp: torch.Tensor) -> None:
        # Matches the official Wanda WrappedGPT.add_batch definition: accumulate
        # squared L2 norm of each input feature, normalized by batch count.
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        tmp = int(inp.shape[0])
        flat = inp.reshape(-1, inp.shape[-1]).t().float()
        self.scaler_row.mul_(self.nsamples / float(self.nsamples + tmp))
        self.nsamples += tmp
        self.scaler_row.add_(torch.norm(flat, p=2, dim=1).pow(2) / float(self.nsamples))


class _CalibrationCaptureStop(RuntimeError):
    pass


def _get_module(layer: nn.Module, dotted_name: str) -> nn.Module:
    cur = layer
    for part in dotted_name.split("."):
        cur = getattr(cur, part)
    return cur


def _layer_device(layer: nn.Module) -> torch.device:
    try:
        return next(layer.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_tree(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_tree(v, device) for v in value)
    if isinstance(value, list):
        return [_move_tree(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _move_tree(v, device) for k, v in value.items()}
    return value


def _detach_tree(value, device: torch.device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, tuple):
        return tuple(_detach_tree(v, device) for v in value)
    if isinstance(value, list):
        return [_detach_tree(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _detach_tree(v, device) for k, v in value.items()}
    return value


def _choose_storage_device(model: nn.Module, nsamples: int, seqlen: int, mode: str) -> torch.device:
    mode = mode.lower()
    if mode not in {"auto", "cuda", "cpu"}:
        raise ValueError("activation storage must be auto, cuda or cpu")
    if mode == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda:0")

    hidden = int(model.config.hidden_size)
    dtype = next(model.parameters()).dtype
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    # We hold both inps and outs; reserve extra headroom for model kernels/sorts.
    required = 2 * nsamples * seqlen * hidden * bytes_per
    try:
        free_bytes, _ = torch.cuda.mem_get_info(0)
        if required < 0.35 * free_bytes:
            return torch.device("cuda:0")
    except Exception:
        pass
    return torch.device("cpu")


def prepare_calibration_inputs(
    model: nn.Module,
    dataloader,
    nsamples: int,
    seqlen: int,
    storage_mode: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, object], torch.device]:
    """Capture first decoder-layer inputs for sequential Wanda-style pruning.

    This follows the official Wanda flow but allows CPU offload when 128 x 4096
    hidden-state buffers would otherwise pressure GPU memory.
    """
    layers = _layers(model)
    first_layer = layers[0]
    embed_device = _layer_device(first_layer)
    device_map = getattr(model, "hf_device_map", {}) or {}
    if "model.embed_tokens" in device_map:
        embed_device = torch.device(device_map["model.embed_tokens"])
    storage = _choose_storage_device(model, nsamples, seqlen, storage_mode)
    dtype = next(model.parameters()).dtype
    hidden = int(model.config.hidden_size)
    inps = torch.empty((nsamples, seqlen, hidden), dtype=dtype, device=storage)
    outs = torch.empty_like(inps)
    cache: dict[str, object] = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            idx = int(cache["i"])
            if idx >= nsamples:
                raise _CalibrationCaptureStop()
            inps[idx].copy_(inp[0].detach().to(storage))
            cache["i"] = idx + 1
            # Position/mask kwargs are sequence-geometry dependent and constant
            # for the fixed-length, no-padding C4 calibration samples.
            if cache["kwargs"] is None:
                cache["kwargs"] = _detach_tree(kwargs, storage)
            raise _CalibrationCaptureStop()

    use_cache = bool(getattr(model.config, "use_cache", False))
    model.config.use_cache = False
    layers[0] = Catcher(first_layer)
    try:
        for batch in dataloader:
            if int(cache["i"]) >= nsamples:
                break
            try:
                model(batch[0].to(embed_device), use_cache=False)
            except _CalibrationCaptureStop:
                pass
    finally:
        layers[0] = first_layer
        model.config.use_cache = use_cache

    captured = int(cache["i"])
    if captured != nsamples:
        raise RuntimeError(f"captured {captured} calibration samples, expected {nsamples}")
    kwargs = cache["kwargs"] or {}
    return inps, outs, kwargs, storage


def _forward_layer(layer: nn.Module, hidden: torch.Tensor, kwargs: Mapping[str, object]):
    result = layer(hidden, **kwargs)
    if isinstance(result, tuple):
        return result[0]
    if hasattr(result, "last_hidden_state"):
        return result.last_hidden_state
    if torch.is_tensor(result):
        return result
    try:
        return result[0]
    except Exception as exc:  # pragma: no cover - defensive HF compatibility
        raise TypeError(f"unsupported decoder-layer output type: {type(result)!r}") from exc


def _metric(module: nn.Linear, stats: ActivationStats) -> torch.Tensor:
    # Frozen-stat modes keep scaler_row on CPU to avoid retaining one GPU tensor
    # per projection for every decoder layer. Move only the small feature vector
    # back to the module device when its mask is computed.
    scaler = stats.scaler_row.to(module.weight.device)
    scale = torch.sqrt(torch.clamp(scaler, min=0.0)).reshape(1, -1)
    return torch.abs(module.weight.data).float() * scale


def _stats_to_cpu(stats: Mapping[str, ActivationStats]) -> dict[str, ActivationStats]:
    return {
        name: ActivationStats(item.scaler_row.detach().cpu().clone(), int(item.nsamples))
        for name, item in stats.items()
    }


def _mask_variable_per_row(metric: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    rows, cols = metric.shape
    counts = counts.to(metric.device, dtype=torch.int64).clamp_(0, cols)
    max_count = int(counts.max().item()) if counts.numel() else 0
    if max_count == 0:
        return torch.zeros_like(metric, dtype=torch.bool)
    # topk is substantially cheaper than materializing a full argsort when the
    # requested sparsity is around 50%; ties are extremely rare for the Wanda metric.
    indices = torch.topk(metric, k=max_count, dim=1, largest=False, sorted=True).indices
    valid = torch.arange(max_count, device=metric.device).unsqueeze(0) < counts.unsqueeze(1)
    mask = torch.zeros_like(metric, dtype=torch.bool)
    mask.scatter_(1, indices, valid)
    return mask


def _mask_fixed_fraction_per_row(metric: torch.Tensor, sparsity: float) -> torch.Tensor:
    k = int(metric.shape[1] * float(sparsity))
    counts = torch.full((metric.shape[0],), k, device=metric.device, dtype=torch.int64)
    return _mask_variable_per_row(metric, counts)


def _split_total_counts(total: np.ndarray, parts: int) -> list[np.ndarray]:
    base = total // parts
    rem = total % parts
    return [base + (rem > i).astype(np.int64) for i in range(parts)]


def _head_counts_to_row_counts(head_counts: np.ndarray, head_dim: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for count in head_counts.astype(np.int64, copy=False):
        base = int(count) // head_dim
        rem = int(count) % head_dim
        arr = np.full(head_dim, base, dtype=np.int64)
        if rem:
            arr[:rem] += 1
        rows.append(arr)
    return np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)


def _group_budget_for_layer(budget: WeightBudget, layer_id: int, unit_type: str) -> np.ndarray | None:
    items = [
        (key.unit_id, int(count))
        for key, count in zip(budget.keys, budget.prune_counts)
        if key.layer_id == layer_id and key.unit_type == unit_type
    ]
    if not items:
        return None
    items.sort()
    return np.asarray([count for _, count in items], dtype=np.int64)


def _apply_mask_(module: nn.Linear, mask: torch.Tensor) -> int:
    pruned = int(mask.sum().item())
    with torch.no_grad():
        module.weight.data[mask] = 0
    return pruned


def _collect_stats_for_layer(
    layer: nn.Module,
    inps: torch.Tensor,
    outs: torch.Tensor,
    layer_kwargs: Mapping[str, object],
    nsamples: int,
) -> dict[str, ActivationStats]:
    modules = {name: _get_module(layer, name) for name in TARGET_MODULE_NAMES}
    stats = {name: ActivationStats.for_linear(module) for name, module in modules.items()}
    handles = []
    for name, module in modules.items():
        def make_hook(key):
            def hook(_module, inp, _out):
                stats[key].add_batch(inp[0].detach())
            return hook
        handles.append(module.register_forward_hook(make_hook(name)))

    dev = _layer_device(layer)
    kwargs_dev = _move_tree(layer_kwargs, dev)
    try:
        for j in range(nsamples):
            hidden = inps[j].unsqueeze(0).to(dev)
            with torch.no_grad():
                out = _forward_layer(layer, hidden, kwargs_dev)
            outs[j].copy_(out[0].detach().to(outs.device))
    finally:
        for handle in handles:
            handle.remove()
    return stats


def _propagate_pruned_layer(
    layer: nn.Module,
    inps: torch.Tensor,
    outs: torch.Tensor,
    layer_kwargs: Mapping[str, object],
    nsamples: int,
) -> None:
    dev = _layer_device(layer)
    kwargs_dev = _move_tree(layer_kwargs, dev)
    for j in range(nsamples):
        hidden = inps[j].unsqueeze(0).to(dev)
        with torch.no_grad():
            out = _forward_layer(layer, hidden, kwargs_dev)
        outs[j].copy_(out[0].detach().to(outs.device))


def _prune_layer_wanda_uniform(layer: nn.Module, stats: Mapping[str, ActivationStats], sparsity: float) -> list[dict]:
    rows = []
    for name in TARGET_MODULE_NAMES:
        module = _get_module(layer, name)
        mask = _mask_fixed_fraction_per_row(_metric(module, stats[name]), sparsity)
        pruned = _apply_mask_(module, mask)
        rows.append({"module": name, "weights": int(module.weight.numel()), "pruned": pruned})
    return rows


def _prune_layer_paper_budget(
    model: nn.Module,
    layer_id: int,
    layer: nn.Module,
    stats: Mapping[str, ActivationStats],
    budget: WeightBudget,
    anchor_sparsity: float,
) -> list[dict]:
    """Apply paper unit quotas while keeping Wanda's per-output behavior where possible.

    For the transposed ownership matrices (MLP down_proj and attention o_proj),
    a standard per-output Wanda mask is applied first. The remaining exact paper
    unit quota is absorbed by gate/up and q/k/v respectively. This avoids the
    activation cancellation that would occur if down/o were pruned purely by
    column/head-column ranking.
    """
    summaries: list[dict] = []

    mlp_total = _group_budget_for_layer(budget, layer_id, "mlp")
    if mlp_total is not None:
        down = layer.mlp.down_proj
        down_mask = _mask_fixed_fraction_per_row(_metric(down, stats["mlp.down_proj"]), anchor_sparsity)
        down_per_unit = down_mask.sum(dim=0).detach().cpu().numpy().astype(np.int64)
        remaining = mlp_total - down_per_unit
        capacity = int(layer.mlp.gate_proj.in_features + layer.mlp.up_proj.in_features)
        if np.any(remaining < 0) or np.any(remaining > capacity):
            raise RuntimeError(
                "paper MLP quota became infeasible after the Wanda down_proj anchor; "
                "narrow the unit sparsity range or change anchor_sparsity"
            )
        gate_counts, up_counts = _split_total_counts(remaining, 2)
        gate = layer.mlp.gate_proj
        up = layer.mlp.up_proj
        gate_mask = _mask_variable_per_row(
            _metric(gate, stats["mlp.gate_proj"]),
            torch.as_tensor(gate_counts, device=gate.weight.device),
        )
        up_mask = _mask_variable_per_row(
            _metric(up, stats["mlp.up_proj"]),
            torch.as_tensor(up_counts, device=up.weight.device),
        )
        for name, module, mask in (
            ("mlp.gate_proj", gate, gate_mask),
            ("mlp.up_proj", up, up_mask),
            ("mlp.down_proj", down, down_mask),
        ):
            summaries.append({"module": name, "weights": int(module.weight.numel()), "pruned": _apply_mask_(module, mask)})

        actual = gate_mask.sum(1).detach().cpu().numpy() + up_mask.sum(1).detach().cpu().numpy() + down_per_unit
        if not np.array_equal(actual.astype(np.int64), mlp_total):
            raise RuntimeError("MLP unit quota mismatch after activation-aware masking")

    attn_total = _group_budget_for_layer(budget, layer_id, "attention")
    if attn_total is not None:
        attn = layer.self_attn
        heads, _, head_dim = _attention_layout(layer, model)
        o_mask = _mask_fixed_fraction_per_row(_metric(attn.o_proj, stats["self_attn.o_proj"]), anchor_sparsity)
        o_per_head = (
            o_mask.view(o_mask.shape[0], heads, head_dim)
            .sum(dim=(0, 2)).detach().cpu().numpy().astype(np.int64)
        )
        remaining = attn_total - o_per_head
        segment_size = head_dim * attn.q_proj.in_features
        if np.any(remaining < 0) or np.any(remaining > 3 * segment_size):
            raise RuntimeError(
                "paper attention quota became infeasible after the Wanda o_proj anchor; "
                "narrow the unit sparsity range or change anchor_sparsity"
            )
        q_counts, k_counts, v_counts = _split_total_counts(remaining, 3)
        masks = {}
        for short, counts in (("q", q_counts), ("k", k_counts), ("v", v_counts)):
            module = getattr(attn, f"{short}_proj")
            row_counts = _head_counts_to_row_counts(counts, head_dim)
            masks[short] = _mask_variable_per_row(
                _metric(module, stats[f"self_attn.{short}_proj"]),
                torch.as_tensor(row_counts, device=module.weight.device),
            )
        for name, module, mask in (
            ("self_attn.q_proj", attn.q_proj, masks["q"]),
            ("self_attn.k_proj", attn.k_proj, masks["k"]),
            ("self_attn.v_proj", attn.v_proj, masks["v"]),
            ("self_attn.o_proj", attn.o_proj, o_mask),
        ):
            summaries.append({"module": name, "weights": int(module.weight.numel()), "pruned": _apply_mask_(module, mask)})

        actual_q = masks["q"].view(heads, head_dim, -1).sum(dim=(1, 2)).detach().cpu().numpy()
        actual_k = masks["k"].view(heads, head_dim, -1).sum(dim=(1, 2)).detach().cpu().numpy()
        actual_v = masks["v"].view(heads, head_dim, -1).sum(dim=(1, 2)).detach().cpu().numpy()
        actual = actual_q + actual_k + actual_v + o_per_head
        if not np.array_equal(actual.astype(np.int64), attn_total):
            raise RuntimeError("attention unit quota mismatch after activation-aware masking")

    return summaries



FROZEN_STATS_CACHE_VERSION = 1


def _frozen_stats_signature(model: nn.Module, nsamples: int, seqlen: int) -> dict:
    layers = _layers(model)
    module_dims = []
    for layer in layers:
        module_dims.append({
            name: int(_get_module(layer, name).weight.shape[1])
            for name in TARGET_MODULE_NAMES
        })
    return {
        "cache_version": FROZEN_STATS_CACHE_VERSION,
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
        "num_layers": int(len(layers)),
        "module_input_dims": module_dims,
    }


def _save_frozen_stats_cache(
    cache_dir: str | Path,
    model: nn.Module,
    frozen_stats: Sequence[Mapping[str, ActivationStats]],
    nsamples: int,
    seqlen: int,
) -> None:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    signature = _frozen_stats_signature(model, nsamples, seqlen)
    for layer_id, stats in enumerate(frozen_stats):
        payload = {
            name: {
                "scaler_row": item.scaler_row.detach().cpu(),
                "nsamples": int(item.nsamples),
            }
            for name, item in stats.items()
        }
        torch.save(payload, root / f"layer_{layer_id:03d}.pt")
    (root / "metadata.json").write_text(
        json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_frozen_stats_cache(
    cache_dir: str | Path,
    model: nn.Module,
    nsamples: int,
    seqlen: int,
) -> list[dict[str, ActivationStats]] | None:
    root = Path(cache_dir)
    meta_path = root / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        actual = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    expected = _frozen_stats_signature(model, nsamples, seqlen)
    if actual != expected:
        print("[wanda stats cache] metadata mismatch; recomputing frozen dense stats", flush=True)
        return None
    result: list[dict[str, ActivationStats]] = []
    for layer_id in range(len(_layers(model))):
        path = root / f"layer_{layer_id:03d}.pt"
        if not path.exists():
            print(f"[wanda stats cache] missing {path.name}; recomputing", flush=True)
            return None
        payload = torch.load(path, map_location="cpu")
        stats = {
            name: ActivationStats(
                scaler_row=item["scaler_row"].detach().cpu().float(),
                nsamples=int(item["nsamples"]),
            )
            for name, item in payload.items()
        }
        if set(stats) != set(TARGET_MODULE_NAMES):
            print(f"[wanda stats cache] invalid module set in {path.name}; recomputing", flush=True)
            return None
        result.append(stats)
    print(f"[wanda stats cache] reused frozen dense stats from {root}", flush=True)
    return result


def sequential_wanda_prune_(
    model: nn.Module,
    dataloader,
    nsamples: int,
    seqlen: int,
    sparsity: float = 0.50,
    budget: WeightBudget | None = None,
    storage_mode: str = "auto",
    prune_order: str = "forward",
    frozen_stats_cache_dir: str | Path | None = None,
) -> list[dict]:
    """Wanda pruning with configurable inter-layer pruning chronology.

    ``forward`` is the original V8.2 behavior: collect current-layer statistics,
    prune that layer, then propagate its sparse output so downstream layers see
    activations produced by the already-pruned prefix.

    ``reverse`` and ``joint`` freeze Wanda activation statistics on the dense
    model before any mask is applied. Those dense statistics can therefore be
    cached and safely reused across reverse/joint and band on/off experiments,
    provided the calibration sample count/length and model architecture match.
    """
    if abs(float(sparsity) - 0.5) > 1e-12:
        raise ValueError("V8.2 currently targets exactly 50% weight sparsity")
    prune_order = str(prune_order).lower()
    if prune_order not in {"forward", "reverse", "joint"}:
        raise ValueError("prune_order must be one of: forward, reverse, joint")

    layers = _layers(model)
    use_cache = bool(getattr(model.config, "use_cache", False))
    model.config.use_cache = False
    summaries: list[dict] = []

    def prune_one(layer_id: int, layer: nn.Module, stats: Mapping[str, ActivationStats]) -> None:
        if budget is None:
            layer_rows = _prune_layer_wanda_uniform(layer, stats, sparsity)
        else:
            layer_rows = _prune_layer_paper_budget(
                model, layer_id, layer, stats, budget, anchor_sparsity=sparsity
            )
        for row in layer_rows:
            row["layer"] = layer_id
            row["prune_order"] = prune_order
            summaries.append(row)

    try:
        if prune_order == "forward":
            inps, outs, layer_kwargs, storage = prepare_calibration_inputs(
                model, dataloader, nsamples=nsamples, seqlen=seqlen, storage_mode=storage_mode
            )
            print(
                f"[wanda mask] mode={prune_order}, calibration nsamples={nsamples}, seqlen={seqlen}, "
                f"activation_storage={storage}",
                flush=True,
            )
            for layer_id, layer in enumerate(layers):
                print(
                    f"[wanda mask] layer {layer_id + 1}/{len(layers)} collect -> prune -> propagate",
                    flush=True,
                )
                stats = _collect_stats_for_layer(layer, inps, outs, layer_kwargs, nsamples)
                prune_one(layer_id, layer, stats)
                _propagate_pruned_layer(layer, inps, outs, layer_kwargs, nsamples)
                inps, outs = outs, inps
                del stats
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            frozen_stats = None
            if frozen_stats_cache_dir:
                frozen_stats = _load_frozen_stats_cache(
                    frozen_stats_cache_dir, model, nsamples=nsamples, seqlen=seqlen
                )

            if frozen_stats is None:
                inps, outs, layer_kwargs, storage = prepare_calibration_inputs(
                    model, dataloader, nsamples=nsamples, seqlen=seqlen, storage_mode=storage_mode
                )
                print(
                    f"[wanda mask] mode={prune_order}, calibration nsamples={nsamples}, seqlen={seqlen}, "
                    f"activation_storage={storage}",
                    flush=True,
                )
                frozen_stats = []
                for layer_id, layer in enumerate(layers):
                    print(
                        f"[wanda mask] freeze dense stats layer {layer_id + 1}/{len(layers)}",
                        flush=True,
                    )
                    stats = _collect_stats_for_layer(layer, inps, outs, layer_kwargs, nsamples)
                    frozen_stats.append(_stats_to_cpu(stats))
                    inps, outs = outs, inps
                    del stats
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if frozen_stats_cache_dir:
                    _save_frozen_stats_cache(
                        frozen_stats_cache_dir, model, frozen_stats,
                        nsamples=nsamples, seqlen=seqlen,
                    )
                    print(
                        f"[wanda stats cache] saved frozen dense stats to {frozen_stats_cache_dir}",
                        flush=True,
                    )

            if prune_order == "reverse":
                apply_ids = range(len(layers) - 1, -1, -1)
            else:
                apply_ids = range(len(layers))
            for step, layer_id in enumerate(apply_ids, start=1):
                print(
                    f"[wanda mask] {prune_order} apply {step}/{len(layers)}: layer {layer_id + 1}",
                    flush=True,
                )
                prune_one(layer_id, layers[layer_id], frozen_stats[layer_id])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        model.config.use_cache = use_cache

    total_pruned = int(sum(int(r["pruned"]) for r in summaries))
    total_weights = int(sum(int(r["weights"]) for r in summaries))
    if budget is None:
        expected = int(round(total_weights * sparsity))
    else:
        expected = int(budget.target_pruned_weights)
    if total_pruned != expected:
        raise RuntimeError(f"activation-aware masker pruned {total_pruned}, expected {expected}")
    return summaries
