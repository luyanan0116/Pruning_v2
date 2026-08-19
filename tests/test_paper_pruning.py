from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from lib.paper_pruning.collector import collect_gradient_response_cache, load_response_cache
from lib.paper_pruning.config import FrequencyConfig, GranularBallConfig, LCBConfig, PipelineConfig
from lib.paper_pruning.granular_ball import build_multigranularity_hierarchy, layer_localization_features
from lib.paper_pruning.mi import build_frequency_spectrum
from lib.paper_pruning.pipeline import score_layer
from lib.paper_pruning.weight_budget import (
    UnitKey,
    allocate_weight_budget,
    apply_weight_budget_,
    unit_segments,
)
from lib.sparsity import check_transformer_weight_sparsity


def synthetic_responses(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, units, length = 72, 24, 32
    events = np.repeat(np.arange(3), n // 3)
    scenarios = np.tile(np.arange(3), n // 3)
    x = rng.normal(0, 0.35, size=(n, units, length)).astype(np.float32)
    t = np.linspace(0, 2 * np.pi, length, endpoint=False)
    for unit in range(units):
        if unit < 8:
            x[:, unit] += events[:, None] * np.sin(t)[None, :] * (0.15 + unit / 40)
        elif unit < 16:
            x[:, unit] += (events == 2)[:, None] * np.cos(4 * t)[None, :] * 0.6
        else:
            x[:, unit] += scenarios[:, None] * np.cos(2 * t)[None, :] * 0.25
    return x, events, scenarios


def test_frequency_and_granular_hierarchy():
    x, events, _ = synthetic_responses()
    spectrum = build_frequency_spectrum(x, events, FrequencyConfig(fine_bins=8, target_bands=4))
    assert spectrum.band_energy.shape == (72, 24, 4)
    assert spectrum.band_mi.shape == (24, 4)
    features = layer_localization_features(spectrum.band_energy)
    cfg = GranularBallConfig(
        purity_thresholds=(0.55, 0.65, 0.75),
        min_ball_size=6,
        max_balls=20,
        min_event_classes=1,
    )
    hierarchy = build_multigranularity_hierarchy(features, events, cfg)
    counts = [len(balls) for _, balls in hierarchy]
    assert counts == sorted(counts)


def test_true_repeated_lcb_has_variance():
    x, events, scenarios = synthetic_responses(2)
    base_ids = np.repeat(np.arange(24), 3)
    cfg = PipelineConfig(
        frequency=FrequencyConfig(fine_bins=8, target_bands=4, mi_neighbors=2, random_state=4),
        granular_ball=GranularBallConfig(
            purity_thresholds=(0.55, 0.65, 0.75),
            min_ball_size=6,
            max_balls=16,
            min_event_classes=2,
            min_event_count_per_class=2,
            localization_mode="layer_shared",
            kde_scope="none",
        ),
        lcb=LCBConfig(
            repeats=5,
            sample_fraction=0.8,
            scenario_fraction=2 / 3,
            lcb_lambda=0.5,
            random_state=7,
        ),
    )
    result = score_layer(x, events, scenarios, cfg, layer_id=3, base_sample_ids=base_ids)
    assert result.bootstrap_scores.shape == (5, 24)
    assert np.any(result.lcb_std > 0)
    assert not np.allclose(result.mi_score, result.granular_score)
    assert np.allclose(result.lcb_score, result.lcb_mean - 0.5 * result.lcb_std)


class ToyMLP(torch.nn.Module):
    def __init__(self, hidden=4, intermediate=6):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)


class ToyAttention(torch.nn.Module):
    def __init__(self, hidden=4, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_heads
        self.head_dim = hidden // num_heads
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        mixed = (self.q_proj(x) + self.k_proj(x) + self.v_proj(x)) / 3
        return self.o_proj(mixed)


class ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()

    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp.down_proj(torch.sigmoid(self.mlp.gate_proj(x)) * self.mlp.up_proj(x))


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyLayer(), ToyLayer()])


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(13, 4)
        self.model = TinyBackbone()
        self.lm_head = torch.nn.Linear(4, 13, bias=False)
        self.config = SimpleNamespace(
            use_cache=True,
            num_attention_heads=2,
            num_key_value_heads=2,
        )

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, labels=None, use_cache=False):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )
        return SimpleNamespace(loss=loss, logits=logits)


def test_collector_contains_only_paper_response_cache(tmp_path):
    model = TinyCausalLM()
    dataloader = []
    for offset in range(4):
        ids = (torch.arange(12).unsqueeze(0) + offset) % 13
        dataloader.append((ids, ids.clone()))
    root = tmp_path / "cache"
    cache = collect_gradient_response_cache(
        model,
        dataloader,
        root,
        scenario_ratios="0.75,1.0",
        response_length=8,
        event_bins=2,
    )
    assert cache.num_observations == 8
    assert cache.load_layer(0, "mlp").shape == (8, 6, 8)
    assert cache.load_layer(0, "attention").shape == (8, 2, 8)


def test_v7_response_cache_version_is_accepted(tmp_path):
    model = TinyCausalLM()
    data = []
    for offset in range(4):
        ids = (torch.arange(12).unsqueeze(0) + offset) % 13
        data.append((ids, ids.clone()))
    root = tmp_path / "cache"
    collect_gradient_response_cache(model, data, root, scenario_ratios="1.0", response_length=8, event_bins=2)
    meta_path = root / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["cache_version"] = 5
    meta_path.write_text(json.dumps(meta))
    loaded = load_response_cache(root)
    assert loaded.cache_version == 5
    assert loaded.load_layer(1, "mlp").shape[1] == 6


def _evidence(model, reverse=False):
    rng = np.random.default_rng(3)
    result = {"mlp": {}, "attention": {}}
    for layer_id, layer in enumerate(model.model.layers):
        for unit_type, count in (
            ("mlp", layer.mlp.down_proj.in_features),
            ("attention", layer.self_attn.num_heads),
        ):
            score = np.linspace(0.1, 1.0, count, dtype=np.float64)
            if reverse:
                score = score[::-1].copy()
            bands = np.abs(rng.normal(size=(count, 4))) + 0.05
            result[unit_type][layer_id] = {"score": score, "bands": bands}
    return result


def _unit_nonzero_count(model, key):
    layer = model.model.layers[key.layer_id]
    total = 0
    for segment in unit_segments(model, key):
        module = layer
        for part in segment.module_name.split("."):
            module = getattr(module, part)
        block = module.weight.data[
            segment.row_start:segment.row_end,
            segment.col_start:segment.col_end,
        ]
        total += int(torch.count_nonzero(block).item())
    return total


def test_paper_nonuniform_budget_is_exact_50_and_never_removes_a_unit():
    model = TinyCausalLM()
    budget = allocate_weight_budget(
        model,
        "paper_mi_gb_lcb",
        _evidence(model),
        ("mlp", "attention"),
        target_sparsity=0.50,
        allocation="paper_nonuniform",
        min_unit_sparsity=0.25,
        max_unit_sparsity=0.75,
    )
    assert budget.prune_counts.sum() * 2 == budget.total_weights
    transformer_weights = sum(
        module.weight.numel()
        for layer in model.model.layers
        for module in (
            layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.o_proj,
            layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj,
        )
    )
    assert budget.total_weights == transformer_weights
    assert np.all(budget.prune_counts < budget.costs)
    apply_weight_budget_(model, budget, seed=11, chunk_size=1024)
    assert abs(check_transformer_weight_sparsity(model) - 0.5) < 1e-12
    assert all(_unit_nonzero_count(model, key) > 0 for key in budget.keys)


def test_score_direction_changes_nonuniform_weight_budget():
    a = TinyCausalLM()
    b = TinyCausalLM()
    budget_a = allocate_weight_budget(
        a, "paper_mi", _evidence(a, reverse=False), ("mlp", "attention"),
        target_sparsity=0.50, allocation="paper_nonuniform",
        min_unit_sparsity=0.25, max_unit_sparsity=0.75,
    )
    budget_b = allocate_weight_budget(
        b, "paper_mi", _evidence(b, reverse=True), ("mlp", "attention"),
        target_sparsity=0.50, allocation="paper_nonuniform",
        min_unit_sparsity=0.25, max_unit_sparsity=0.75,
    )
    assert not np.array_equal(budget_a.prune_counts, budget_b.prune_counts)
    assert budget_a.prune_counts.sum() == budget_b.prune_counts.sum()


def test_uniform_weight_budget_is_exact_50():
    model = TinyCausalLM()
    budget = allocate_weight_budget(
        model,
        "paper_mi",
        _evidence(model),
        ("mlp", "attention"),
        target_sparsity=0.50,
        allocation="uniform",
    )
    assert budget.prune_counts.sum() * 2 == budget.total_weights
    apply_weight_budget_(model, budget, seed=7, chunk_size=1024)
    assert abs(check_transformer_weight_sparsity(model) - 0.5) < 1e-12


def test_full_coverage_budget_keeps_exact_global_target():
    model = TinyCausalLM()
    budget = allocate_weight_budget(
        model,
        "paper_full",
        _evidence(model),
        ("mlp", "attention"),
        target_sparsity=0.50,
        allocation="paper_nonuniform",
        min_unit_sparsity=0.30,
        max_unit_sparsity=0.70,
        coverage_ratio=0.90,
        coverage_alpha=0.20,
        greedy_batches=8,
    )
    assert budget.prune_counts.sum() * 2 == budget.total_weights
    assert np.min(budget.prune_counts / budget.costs) >= 0.30 - 1e-9
    assert np.max(budget.prune_counts / budget.costs) <= 0.70 + 1e-9
