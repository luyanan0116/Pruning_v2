from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.fft import dct
from torch import nn

from lib.paper_pruning.apply import zero_attention_heads_, zero_mlp_channels_
from lib.paper_pruning.collector import collect_gradient_response_cache
from lib.paper_pruning.config import (
    BudgetConfig,
    FrequencyConfig,
    GranularBallConfig,
    LCBConfig,
    PipelineConfig,
)
from lib.paper_pruning.global_budget import coverage_aware_multi_budget_indices
from lib.paper_pruning.materialize import materialize_structured_units_
from lib.paper_pruning.granular_ball import (
    granularity_weights_from_repeat_variance,
    unit_localization_features,
)
from lib.paper_pruning.mi import (
    adaptive_merge_adjacent_bins,
    dct_frequency_energy,
    estimate_knn_mi_matrix,
    proposal_dct,
)
from lib.paper_pruning.pipeline import score_layer
from lib.paper_pruning.scenarios import load_scenario_manifest


def test_proposal_dct_matches_displayed_cosine_sum():
    rng = np.random.default_rng(0)
    responses = rng.normal(size=(3, 2, 12)).astype(np.float32)
    standardized = (responses - responses.mean(-1, keepdims=True)) / (
        responses.std(-1, keepdims=True) + 1e-8
    )
    manual = np.zeros_like(standardized)
    length = standardized.shape[-1]
    for k in range(length):
        cosine = np.cos(np.pi * (2 * np.arange(length) + 1) * k / (2 * length))
        manual[..., k] = np.sum(standardized * cosine, axis=-1)
    assert np.allclose(proposal_dct(responses), manual, atol=2e-5)
    assert np.allclose(proposal_dct(responses), dct(standardized, type=2, axis=-1) / 2, atol=2e-5)


def test_frequency_energy_is_squared_and_binned():
    responses = np.asarray([[[1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]]])
    cfg = FrequencyConfig(fine_bins=4, target_bands=2)
    coefficients = proposal_dct(responses, cfg.eps)
    groups = np.array_split(np.arange(8), 4)
    expected = np.stack([(coefficients[..., group] ** 2).sum(-1) for group in groups], axis=-1)
    assert np.allclose(dct_frequency_energy(responses, cfg), expected)


def test_entropy_decomposition_mi_detects_dependency():
    rng = np.random.default_rng(1)
    events = np.repeat([0, 1], 100)
    dependent = events + rng.normal(scale=0.12, size=events.size)
    independent = rng.normal(size=events.size)
    values = np.column_stack([dependent, independent])
    estimates = estimate_knn_mi_matrix(values, events, FrequencyConfig(mi_neighbors=4))
    assert estimates[0] > estimates[1] + 0.15
    assert np.all(estimates >= 0)


def test_adaptive_merge_uses_probe_information_loss():
    rng = np.random.default_rng(2)
    samples, units, bins = 80, 6, 8
    events = np.repeat([0, 1], samples // 2)
    energy = rng.gamma(2.0, 0.2, size=(samples, units, bins)).astype(np.float32)
    energy[events == 1, :, :2] += 2.5
    ranges, relevance, probes = adaptive_merge_adjacent_bins(
        energy,
        events,
        FrequencyConfig(fine_bins=bins, target_bands=3, probe_units=4, mi_neighbors=3),
    )
    assert len(ranges) == 3
    assert ranges[0][0] == 0 and ranges[-1][1] == bins
    assert all(ranges[index][1] == ranges[index + 1][0] for index in range(len(ranges) - 1))
    assert relevance.shape == (bins,)
    assert 1 <= probes.size <= units


def _synthetic_responses(seed: int = 3):
    rng = np.random.default_rng(seed)
    samples, units, length = 48, 5, 32
    base_ids = np.repeat(np.arange(12), 4)
    scenarios = np.tile(np.arange(4), 12)
    events = ((base_ids + scenarios) % 2).astype(np.int64)
    t = np.arange(length)
    responses = rng.normal(scale=0.25, size=(samples, units, length))
    low = np.cos(np.pi * t / length)
    high = np.cos(np.pi * 12 * (2 * t + 1) / (2 * length))
    responses[:, 0, :] += np.where(events[:, None] == 0, low, high)
    responses[:, 1, :] += np.where(events[:, None] == 0, high, low)
    return responses.astype(np.float32), events, scenarios, base_ids


def test_full_scoring_pipeline_outputs_granular_and_lcb_scores():
    responses, events, scenarios, base_ids = _synthetic_responses()
    config = PipelineConfig(
        frequency=FrequencyConfig(
            fine_bins=8,
            target_bands=3,
            mi_neighbors=2,
            probe_units=3,
        ),
        granular_ball=GranularBallConfig(
            purity_thresholds=(0.60, 0.75),
            min_ball_size=4,
            max_balls=8,
            max_depth=3,
            workers=1,
            kde_scope="none",
        ),
        lcb=LCBConfig(
            repeats=3,
            sample_fraction=0.8,
            scenario_fraction=0.75,
            lcb_lambda=1.0,
            workers=1,
        ),
        budget=BudgetConfig(coverage_ratio=0.8, coverage_alpha=0.2),
    )
    result = score_layer(
        responses,
        events,
        scenarios,
        config,
        layer_id=0,
        base_sample_ids=base_ids,
    )
    assert result.mi_score.shape == (responses.shape[1],)
    assert result.granular_band_mi.shape == (responses.shape[1], 3)
    assert result.bootstrap_scores.shape == (3, responses.shape[1])
    assert result.bootstrap_level_band_scores.shape[:3] == (3, 2, responses.shape[1])
    assert np.isclose(result.granularity_weights.sum(), 1.0)
    assert np.allclose(result.lcb_score, result.lcb_mean - result.lcb_std)
    assert result.global_spectrum.band_ranges



def test_particle_space_uses_literal_raw_band_response_vectors():
    values = np.asarray([[1.0, 10.0], [3.0, 40.0], [8.0, 90.0]])
    assert np.array_equal(unit_localization_features(values), values)


def test_variance_adaptive_fusion_prefers_stable_granularity():
    rng = np.random.default_rng(123)
    stable = 1.0 + rng.normal(scale=0.01, size=(12, 1, 4, 3))
    unstable = 1.0 + rng.normal(scale=0.5, size=(12, 1, 4, 3))
    repeats = np.concatenate([stable, unstable], axis=1)
    weights, variances = granularity_weights_from_repeat_variance(
        repeats, GranularBallConfig(purity_thresholds=(0.60, 0.80))
    )
    assert variances[0] < variances[1]
    assert weights[0] > weights[1]
    assert np.isclose(weights.sum(), 1.0)

def _brute_greedy(scores, bands, costs, limits, cfg):
    selected = []
    spent = np.zeros(costs.shape[1])
    coverage = np.zeros(bands.shape[1])
    targets = cfg.coverage_ratio * bands.sum(0)
    effective = np.mean(costs / limits[None, :], axis=1)
    while True:
        feasible = [
            i for i in range(len(scores))
            if i not in selected and np.all(spent + costs[i] <= limits + 1e-12)
        ]
        if not feasible:
            break
        under = np.maximum(0, targets - coverage)
        gains = {
            i: (scores[i] + cfg.coverage_alpha * bands[i].dot(under)) / effective[i]
            for i in feasible
        }
        best = max(feasible, key=lambda i: (gains[i], -i))
        if not cfg.fill_budget and np.all(coverage >= targets) and gains[best] <= 0:
            break
        selected.append(best)
        spent += costs[best]
        coverage += bands[best]
    return np.asarray(sorted(selected)), spent, coverage / np.maximum(bands.sum(0), 1e-12)


def test_lazy_global_greedy_matches_full_rescoring():
    scores = np.asarray([1.2, 0.9, 0.7, 0.3, 0.2])
    bands = np.asarray([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.6, 0.4],
        [0.2, 0.7],
        [0.5, 0.1],
    ])
    costs = np.asarray([
        [2.0, 3.0],
        [2.0, 2.0],
        [1.0, 2.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ])
    limits = np.asarray([5.0, 7.0])
    cfg = BudgetConfig(coverage_ratio=0.55, coverage_alpha=0.4, fill_budget=True)
    expected, expected_spent, expected_coverage = _brute_greedy(scores, bands, costs, limits, cfg)
    keep, spent, coverage, _ = coverage_aware_multi_budget_indices(
        scores, bands, costs, limits, cfg
    )
    assert np.array_equal(keep, expected)
    assert np.allclose(spent, expected_spent)
    assert np.allclose(coverage, expected_coverage)
    assert np.all(spent <= limits + 1e-12)


class FakeAttention(nn.Module):
    def __init__(self, hidden: int = 8, heads: int = 2):
        super().__init__()
        self.num_heads = heads
        self.num_key_value_heads = heads
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.o_proj(x)


class FakeMLP(nn.Module):
    def __init__(self, hidden: int = 8, width: int = 12):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, width, bias=False)
        self.up_proj = nn.Linear(hidden, width, bias=False)
        self.down_proj = nn.Linear(width, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()
        self.mlp = FakeMLP()

    def forward(self, x):
        return x + self.self_attn(x) + self.mlp(x)


class FakeLM(nn.Module):
    def __init__(self, layers: int = 2, vocab: int = 17):
        super().__init__()
        self.embed = nn.Embedding(vocab, 8)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer() for _ in range(layers)])
        self.lm_head = nn.Linear(8, vocab, bias=False)
        self.config = SimpleNamespace(
            use_cache=False,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=layers,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=vocab,
            model_type="fake",
            _name_or_path="fake",
        )

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, labels=None, use_cache=False):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x)
        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        return SimpleNamespace(logits=logits, loss=loss)


def test_gradient_collector_captures_complete_structural_units(tmp_path):
    torch.manual_seed(0)
    model = FakeLM(layers=2)
    loader = [(torch.randint(0, 17, (1, 24)), None) for _ in range(4)]
    cache = collect_gradient_response_cache(
        model,
        loader,
        tmp_path / "cache",
        scenario_ratios="1.0",
        scenario_crops="prefix",
        fine_bins=4,
        event_bins=2,
        overwrite=True,
        cache_signature={"test": 1},
    )
    assert cache.load_layer(0, "mlp").shape == (4, 12, 4)
    assert cache.load_layer(0, "attention").shape == (4, 2, 4)
    assert cache.position_offsets.shape == (5,)
    assert cache.position_loss_slice(0).shape == (23,)
    assert cache.position_event_slice(0).shape == (23,)
    assert np.unique(cache.events).size == 2


def test_complete_unit_zeroing_changes_all_coupled_tensors():
    model = FakeLM(layers=1)
    layer = model.model.layers[0]
    zero_mlp_channels_(model, {0: [2]})
    assert torch.count_nonzero(layer.mlp.gate_proj.weight[2]) == 0
    assert torch.count_nonzero(layer.mlp.up_proj.weight[2]) == 0
    assert torch.count_nonzero(layer.mlp.down_proj.weight[:, 2]) == 0

    zero_attention_heads_(model, {0: [1]})
    rows = slice(4, 8)
    assert torch.count_nonzero(layer.self_attn.q_proj.weight[rows]) == 0
    assert torch.count_nonzero(layer.self_attn.k_proj.weight[rows]) == 0
    assert torch.count_nonzero(layer.self_attn.v_proj.weight[rows]) == 0
    assert torch.count_nonzero(layer.self_attn.o_proj.weight[:, rows]) == 0


def test_physical_surgery_reduces_unit_dimensions():
    model = FakeLM(layers=1)
    before = sum(parameter.numel() for parameter in model.parameters())
    summary = materialize_structured_units_(
        model,
        {"mlp": {0: [0, 1]}, "attention": {0: [1]}},
        require_uniform=True,
    )
    layer = model.model.layers[0]
    assert layer.mlp.down_proj.in_features == 10
    assert layer.self_attn.num_heads == 1
    assert layer.self_attn.q_proj.out_features == 4
    assert summary["parameters_after"] < before


class FakeTokenizer:
    def __call__(self, text, return_tensors, truncation, max_length, add_special_tokens):
        del return_tensors, truncation, add_special_tokens
        ids = [min(30, ord(char) % 31) for char in text][:max_length]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


def test_scenario_manifest_preserves_paired_sample_and_scenario_ids(tmp_path):
    path = tmp_path / "scenarios.jsonl"
    records = [
        {"text": "abcdefghijk", "base_sample_id": "a", "scenario_id": "short", "task": "x"},
        {"text": "abcdefghijkl", "base_sample_id": "a", "scenario_id": "long", "task": "x"},
    ]
    path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    loaded = load_scenario_manifest(path, FakeTokenizer(), max_length=16)
    assert len(loaded) == 2
    assert loaded[0]["base_sample_id"] == loaded[1]["base_sample_id"]
    assert {item["scenario_id"] for item in loaded} == {"short", "long"}


def test_source_tree_contains_no_removed_legacy_component_name():
    root = Path(__file__).resolve().parents[1]
    prohibited = "wan" + "da"
    hits = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".sh", ".txt"}:
            if prohibited in path.read_text(encoding="utf-8", errors="ignore").lower():
                hits.append(str(path.relative_to(root)))
    assert hits == []


def _strict_ablation_config():
    return PipelineConfig(
        frequency=FrequencyConfig(
            fine_bins=8,
            target_bands=3,
            mi_neighbors=2,
            probe_units=3,
        ),
        granular_ball=GranularBallConfig(
            purity_thresholds=(0.60, 0.75),
            granularity_weight_mode="equal",
            min_ball_size=4,
            max_balls=8,
            max_depth=3,
            workers=1,
            kde_scope="none",
        ),
        lcb=LCBConfig(
            repeats=3,
            sample_fraction=0.8,
            scenario_fraction=0.75,
            lcb_lambda=1.0,
            workers=1,
        ),
        budget=BudgetConfig(coverage_ratio=0.8, coverage_alpha=0.2),
    )


def test_strict_mi_stage_does_not_execute_granular_ball_or_lcb():
    responses, events, scenarios, base_ids = _synthetic_responses(seed=31)
    result = score_layer(
        responses,
        events,
        scenarios,
        _strict_ablation_config(),
        base_sample_ids=base_ids,
        method="paper_mi",
    )
    assert result.executed_stages == ("mutual_information",)
    assert result.bootstrap_scores.shape[0] == 0
    assert np.isnan(result.granular_score).all()
    assert np.isnan(result.lcb_score).all()
    assert np.array_equal(result.selected_score, result.mi_score)


def test_strict_mi_gb_stage_does_not_execute_repeats_or_lcb():
    responses, events, scenarios, base_ids = _synthetic_responses(seed=32)
    result = score_layer(
        responses,
        events,
        scenarios,
        _strict_ablation_config(),
        base_sample_ids=base_ids,
        method="paper_mi_gb",
    )
    assert result.executed_stages == (
        "mutual_information",
        "granular_ball",
        "multi_granularity_fusion",
    )
    assert result.bootstrap_scores.shape[0] == 0
    assert np.isnan(result.lcb_score).all()
    assert np.allclose(result.granularity_weights, [0.5, 0.5])
    assert np.array_equal(result.selected_score, result.granular_score)


def test_strict_full_stage_adds_repeated_estimation_and_lcb_only():
    responses, events, scenarios, base_ids = _synthetic_responses(seed=33)
    result = score_layer(
        responses,
        events,
        scenarios,
        _strict_ablation_config(),
        base_sample_ids=base_ids,
        method="paper_mi_gb_lcb",
    )
    assert "dual_source_repeated_estimation" in result.executed_stages
    assert "lcb" in result.executed_stages
    assert result.bootstrap_scores.shape[0] == 3
    assert np.allclose(result.granularity_weights, [0.5, 0.5])
    assert np.allclose(result.lcb_score, result.lcb_mean - result.lcb_std)
    assert np.array_equal(result.selected_score, result.lcb_score)
