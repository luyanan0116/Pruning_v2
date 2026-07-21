from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from lib.paper_pruning.apply import zero_attention_heads_, zero_mlp_channels_, zero_structured_units_
from lib.paper_pruning.collector import collect_gradient_response_cache
from lib.paper_pruning.config import FrequencyConfig, GranularBallConfig, LCBConfig, PipelineConfig
from lib.paper_pruning.granular_ball import build_multigranularity_hierarchy, layer_localization_features
from lib.paper_pruning.mi import build_frequency_spectrum
from lib.paper_pruning.pipeline import score_layer
from lib.paper_pruning.selection import resolve_prune_count, select_bottom_k, split_into_steps
from lib.paper_pruning.wanda_weight import (
    ALL_LINEAR_MODULES,
    allocate_row_prune_counts,
    apply_paper_wanda_weight_masks_,
)


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


def test_frequency_spectrum_and_selection():
    x, events, _ = synthetic_responses()
    cfg = FrequencyConfig(fine_bins=8, target_bands=4, mi_neighbors=3)
    spectrum = build_frequency_spectrum(x, events, cfg)
    assert spectrum.band_energy.shape == (72, 24, 4)
    assert spectrum.band_mi.shape == (24, 4)
    assert np.isfinite(spectrum.total_mi).all()
    chosen = select_bottom_k(spectrum.total_mi, 5)
    assert chosen.shape == (5,)
    assert len(split_into_steps(chosen, 2)) == 3
    assert resolve_prune_count(24, 0.25, 0) == 6
    assert resolve_prune_count(24, 0.25, 7) == 7


def test_purity_thresholds_create_nested_ball_counts():
    x, events, _ = synthetic_responses(1)
    spectrum = build_frequency_spectrum(x, events, FrequencyConfig(fine_bins=8, target_bands=4))
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
    assert counts[-1] >= counts[0]
    assert all(0.0 <= ball.purity <= 1.0 for _, balls in hierarchy for ball in balls)


def test_three_ablation_paths_and_lcb_variance():
    x, events, scenarios = synthetic_responses(2)
    cfg = PipelineConfig(
        frequency=FrequencyConfig(fine_bins=8, target_bands=4, mi_neighbors=2, random_state=4),
        granular_ball=GranularBallConfig(
            purity_thresholds=(0.55, 0.65, 0.75),
            min_ball_size=6,
            max_balls=16,
            min_event_classes=1,
        ),
        lcb=LCBConfig(repeats=5, sample_fraction=0.75, lcb_lambda=1.0, random_state=7),
    )
    result = score_layer(x, events, scenarios, cfg, layer_id=3)
    assert result.bootstrap_scores.shape == (5, 24)
    assert np.any(result.lcb_std > 0)
    assert not np.allclose(result.mi_score, result.granular_score)
    assert not np.allclose(result.granular_score, result.lcb_score)


class ToyMLP(torch.nn.Module):
    def __init__(self, hidden=4, intermediate=6):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=True)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=True)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)


class ToyAttention(torch.nn.Module):
    def __init__(self, hidden=4, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_heads
        self.head_dim = hidden // num_heads
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=True)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=True)
        self.v_proj = torch.nn.Linear(hidden, hidden, bias=True)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        # The collector only needs a differentiable, head-concatenated o_proj input.
        mixed = (self.q_proj(x) + self.k_proj(x) + self.v_proj(x)) / 3
        return self.o_proj(mixed)


class ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()

    def forward(self, x):
        x = x + self.self_attn(x)
        gate = torch.sigmoid(self.mlp.gate_proj(x))
        up = self.mlp.up_proj(x)
        return x + self.mlp.down_proj(gate * up)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=[ToyLayer(), ToyLayer()])


def test_structured_mlp_and_attention_zeroing():
    model = ToyModel()
    zero_structured_units_(
        model,
        {
            "mlp": {0: [1, 4], 1: [2]},
            "attention": {0: [1], 1: [0]},
        },
    )
    layer0 = model.model.layers[0]
    assert torch.count_nonzero(layer0.mlp.gate_proj.weight[[1, 4]]) == 0
    assert torch.count_nonzero(layer0.mlp.up_proj.weight[[1, 4]]) == 0
    assert torch.count_nonzero(layer0.mlp.down_proj.weight[:, [1, 4]]) == 0
    assert torch.count_nonzero(layer0.mlp.gate_proj.bias[[1, 4]]) == 0

    # Head 1 corresponds to rows/columns 2:4 for hidden=4, heads=2.
    assert torch.count_nonzero(layer0.self_attn.q_proj.weight[2:4]) == 0
    assert torch.count_nonzero(layer0.self_attn.k_proj.weight[2:4]) == 0
    assert torch.count_nonzero(layer0.self_attn.v_proj.weight[2:4]) == 0
    assert torch.count_nonzero(layer0.self_attn.o_proj.weight[:, 2:4]) == 0


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
            _name_or_path="tiny",
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
        shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
        shift_labels = input_ids[:, 1:].reshape(-1)
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
        return SimpleNamespace(loss=loss, logits=logits)


def test_gradient_response_collector_mlp_and_attention(tmp_path):
    model = TinyCausalLM()
    dataloader = []
    for offset in range(4):
        ids = (torch.arange(12).unsqueeze(0) + offset) % 13
        dataloader.append((ids, ids.clone()))
    cache = collect_gradient_response_cache(
        model,
        dataloader,
        tmp_path / "cache",
        scenario_ratios="0.75,1.0",
        response_length=8,
        event_bins=2,
    )
    assert cache.num_observations == 8
    assert cache.load_layer(0, "mlp").shape == (8, 6, 8)
    assert cache.load_layer(0, "attention").shape == (8, 2, 8)
    assert cache.unit_counts["mlp"][0] == 6
    assert cache.unit_counts["attention"][0] == 2
    assert np.unique(cache.events).size == 2
    assert np.isfinite(cache.losses).all()
    for module_name in ALL_LINEAR_MODULES:
        scale = cache.load_activation_scale(0, module_name)
        assert scale.shape == (getattr(getattr(model.model.layers[0], module_name.split('.')[0]), module_name.split('.')[1]).in_features,)
        assert np.isfinite(scale).all()
        assert np.all(scale >= 0)


class FakeResponseCache:
    def __init__(self, model):
        self.attention_layouts = {
            layer_id: {"num_heads": 2, "num_key_value_heads": 2, "head_dim": 2}
            for layer_id, _ in enumerate(model.model.layers)
        }
        self.scales = {}
        for layer_id, layer in enumerate(model.model.layers):
            for module_name in ALL_LINEAR_MODULES:
                module = layer
                for part in module_name.split("."):
                    module = getattr(module, part)
                self.scales[(layer_id, module_name)] = np.linspace(0.5, 1.5, module.in_features).astype(np.float32)

    def load_activation_scale(self, layer_id, module_name):
        return self.scales[(layer_id, module_name)]


def _toy_scores(model, reverse=False):
    result = {"mlp": {}, "attention": {}}
    for layer_id, layer in enumerate(model.model.layers):
        mlp = np.arange(layer.mlp.down_proj.in_features, dtype=np.float32)
        attn = np.arange(layer.self_attn.num_heads, dtype=np.float32)
        if reverse:
            mlp = mlp[::-1].copy()
            attn = attn[::-1].copy()
        result["mlp"][layer_id] = mlp
        result["attention"][layer_id] = attn
    return result


def test_row_budget_is_exact_and_score_aware():
    scores = np.arange(6, dtype=np.float32)
    counts = allocate_row_prune_counts(
        scores, rows_per_unit=1, columns=10, target_ratio=0.5, spread=0.8, temperature=2.0
    )
    assert counts.sum() == 30
    assert counts[0] > counts[-1]


def test_wanda_weight_masks_cover_attention_and_mlp():
    torch.manual_seed(1)
    model = ToyModel()
    cache = FakeResponseCache(model)
    summaries = apply_paper_wanda_weight_masks_(
        model,
        _toy_scores(model),
        cache,
        targets=("mlp", "attention"),
        mlp_ratio=0.5,
        attention_ratio=0.5,
        chunk_rows=2,
    )
    assert len(summaries) == 2 * 7
    for layer in model.model.layers:
        for module_name in ALL_LINEAR_MODULES:
            module = layer
            for part in module_name.split("."):
                module = getattr(module, part)
            ratio = float((module.weight == 0).sum().item()) / module.weight.numel()
            assert abs(ratio - 0.5) <= 1.0 / module.weight.numel()


def test_paper_scores_change_wanda_weight_masks():
    torch.manual_seed(7)
    left = ToyModel()
    torch.manual_seed(7)
    right = ToyModel()
    apply_paper_wanda_weight_masks_(
        left, _toy_scores(left), FakeResponseCache(left), ("mlp", "attention"), 0.5, 0.5, chunk_rows=2
    )
    apply_paper_wanda_weight_masks_(
        right, _toy_scores(right, reverse=True), FakeResponseCache(right), ("mlp", "attention"), 0.5, 0.5, chunk_rows=2
    )
    differences = 0
    for left_layer, right_layer in zip(left.model.layers, right.model.layers):
        for module_name in ALL_LINEAR_MODULES:
            a, b = left_layer, right_layer
            for part in module_name.split("."):
                a, b = getattr(a, part), getattr(b, part)
            differences += int(torch.count_nonzero((a.weight == 0) != (b.weight == 0)).item())
    assert differences > 0
