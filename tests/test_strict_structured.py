import numpy as np
import torch
import torch.nn as nn

from lib.paper_pruning.apply_strict import apply_structured_pruning_
from lib.paper_pruning.budget_strict import select_keep_set_eq11_13
from lib.paper_pruning.config import BudgetConfig


def test_eq11_13_hard_coverage_and_budget():
    scores = np.array([2.0, 1.8, 1.5, 1.0, 0.7, 0.5])
    bands = np.array([
        [4.0, 0.0],
        [2.0, 0.1],
        [0.5, 3.0],
        [0.0, 2.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ])
    costs = np.ones(6)
    cfg = BudgetConfig(coverage_ratio=0.50, coverage_alpha=1.0, greedy_batches=16)
    result = select_keep_set_eq11_13(scores, bands, costs, budget=4.0, cfg=cfg)
    assert result.spent_cost <= 4.0 + 1e-9
    assert np.all(result.achieved_coverage >= 0.50 - 1e-9)


class ToyMLP(nn.Module):
    def __init__(self, hidden=8, intermediate=6):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class ToyAttn(nn.Module):
    def __init__(self, hidden=8, q_heads=4, kv_heads=2, head_dim=2):
        super().__init__()
        self.num_heads = q_heads
        self.num_key_value_heads = kv_heads
        self.num_key_value_groups = q_heads // kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(q_heads * head_dim, hidden, bias=False)


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttn()
        self.mlp = ToyMLP()


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer()])


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBackbone()


def test_physical_complete_ffn_and_gqa_bundle_shrink():
    model = ToyModel()
    rows = apply_structured_pruning_(
        model,
        {
            "mlp": {0: [1, 4]},
            "attention": {0: [1]},
        },
        mode="shrink",
    )
    layer = model.model.layers[0]
    assert layer.mlp.gate_proj.out_features == 4
    assert layer.mlp.up_proj.out_features == 4
    assert layer.mlp.down_proj.in_features == 4

    # One of two KV bundles removed. Each bundle owns two query heads.
    assert layer.self_attn.num_key_value_heads == 1
    assert layer.self_attn.num_heads == 2
    assert layer.self_attn.q_proj.out_features == 4
    assert layer.self_attn.k_proj.out_features == 2
    assert layer.self_attn.v_proj.out_features == 2
    assert layer.self_attn.o_proj.in_features == 4
    assert sum(r["pruned"] for r in rows) == 3
