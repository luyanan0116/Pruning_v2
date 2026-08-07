# Pruning paper-aligned v7 strict structured

本版本把 **Wanda 与论文主流程彻底分开**：

- `--prune_method wanda`：只作为普通 Wanda 权重级基线保留。
- `--prune_method paper_mi_gb_lcb` / `paper_full`：不调用任何 Wanda metric、Wanda activation scale 或 Wanda mask。
- 论文主路径最终生成全局保留集合 `K`，并剪完整 Attention 结构单元和 FFN 中间通道。

## 严格主流程

```text
梯度响应
→ 样本内标准化
→ DCT
→ 细粒度频带能量
→ 任务驱动相邻频带合并
→ MI
→ 粒球局部化
→ 多粒度融合
→ 样本-场景双源重复估计
→ LCB
→ Eq.(11) 频带硬覆盖
→ Eq.(12) 欠覆盖权重
→ Eq.(13) 预算边际增益
→ K
→ 完整 Head/GQA bundle + FFN Channel
→ 结构剪枝率
→ PPL
```

## 推荐运行

```bash
bash scripts/run_fast_paper_ablation.sh
```

或：

```bash
python main.py \
  --model /path/to/model \
  --prune_method paper_mi_gb_lcb \
  --sparsity_ratio 0.15 \
  --paper_apply_mode shrink \
  --paper_budget_metric params
```

## 输出

```text
paper_structured_report/
  keep_set_K.json
  prune_indices.json
  all_contribution_scores_strict.csv
  budget_selection_eq11_13.json
  structural_pruning_manifest.json
```

`structural_pruning_manifest.json` 同时报告：
1. 完整结构单元剪枝率；
2. 候选结构 cost-weighted 剪枝率；
3. shrink 模式下整模型真实参数量下降比例。

## Wanda 保留范围

请看 `WANDA_AUDIT.md`。v6 中 Wanda 与论文主流程耦合的代码已经移出 active path，并原样归档在 `legacy/v6_wanda_paper/`，便于回查。普通 Wanda 基线仍由 `lib/prune.py::prune_wanda()` 与 `scripts/run_wanda_baseline.sh` 提供。

## shrink 与 zero

- `--paper_apply_mode shrink`：物理缩小线性层张量维度，最符合“完整结构剪枝”。
- `--paper_apply_mode zero`：完整单元置零但不改变张量形状，适合作为 Hugging Face 兼容性对照。

注意：若各层保留维度不同，`shrink` 后不能仅凭一个全局 HF config 无损重建。代码会保存结构 manifest；PPL 评测在实际剪枝后的内存模型上完成。
