# 青基2026 + sample 严格流程到代码映射

## 主流程

| 论文流程 | v7 文件/函数 |
|---|---|
| 梯度响应 | `paper_pruning/collector_strict.py::collect_gradient_response_cache` |
| 样本内标准化 | `paper_pruning/mi.py::standardize_responses` |
| DCT | `paper_pruning/mi.py::dct_frequency_energy` |
| 频带能量 | `paper_pruning/mi.py::dct_frequency_energy` |
| 任务驱动频带合并 | `paper_pruning/mi.py::adaptive_merge_adjacent_bins` |
| MI | `paper_pruning/mi.py::calculate_band_mi` |
| 粒球 | `paper_pruning/granular_ball.py::build_multigranularity_hierarchy` |
| 多粒度融合 | `paper_pruning/granular_ball.py::multi_granularity_local_mi` |
| 双源重复估计 | `paper_pruning/resampling.py::dual_source_bootstrap_indices` + `pipeline.py::_bootstrap_index_sets` |
| LCB | `paper_pruning/pipeline.py::score_layer` |
| Eq.(11) 频带硬覆盖 | `paper_pruning/budget_strict.py::select_keep_set_eq11_13` |
| Eq.(12) 欠覆盖权重 | 同上 `deficit = max(0,target-current)` |
| Eq.(13) 预算边际增益 | 同上 `LCB/cost + alpha*(I @ deficit)/cost` |
| K | `prune_paper_strict.py` 输出 `keep_set_K.json` |
| 完整 Head/FFN Channel | `paper_pruning/apply_strict.py::apply_structured_pruning_` |
| 结构剪枝率 | `structural_pruning_manifest.json` |
| PP/PPL | `main.py` 在剪后模型上调用 `eval_ppl()` |

## 两个实现口径说明

- Attention：MHA 一个 head 就是一个结构单元；GQA 为保证 q/k/v/o 能一起物理裁剪，结构单元定义成一个 KV head 及其共享 query-head bundle。
- Eq.(11) 的 `C_b(K)` 用稳健逐频带贡献构造；LCB 用作保留目标。这样避免把不确定性惩罚重复施加到频带覆盖本身。
- `sparsity_ratio` 在严格 paper path 中解释为候选结构单元 **cost-weighted 剪枝率**；默认 `cost=params`，也可用 `uniform` 查看纯结构单元比例。
