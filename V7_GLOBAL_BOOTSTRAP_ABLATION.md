# v7: 双源 Bootstrap + 稳健粒球 + 跨层全局预算 + 干净四级消融

本版本针对 v6 中后续模块对 PPL/mask 影响过弱的问题，做了四项核心修正。

## 1. 真正的双源 Bootstrap 与 LCB

默认 `--paper_lcb_repeats 10`。每次重复先对 base calibration sample 重采样，再对该 base sample 的 scenario realization 重采样；若缺少 base/scenario 标识，则回退到事件分层 bootstrap。

LCB 使用重复估计经验分布：`LCB = mean - lambda * std`。`--paper_lcb_repeats 1` 只保留为显式单次调试模式。

## 2. 粒球稳健性修正

- split 时拒绝会产生局部事件类别样本数不足的子球；
- invalid ball 不再导致整个粒度的贡献按缺失质量缩小，剩余 valid ball 权重会重新归一化；
- 默认多粒度融合改为 `inverse_sqrt_dispersion`；
- 默认把不同粒度原始融合权重比例限制在 `5x`，避免“假多粒度”退化为单一粒度。

## 3. 跨层全局预算 / 覆盖选择

默认 `--paper_global_budget`。同一 unit type（MLP channel 或 attention head）的各层先统一拼接，再在保持总剪枝数完全不变的前提下全局选 keep/prune。这样不再强制每一层都剪相同比例。

Full 的 coverage 贪心每一轮都同时包含 scalar LCB 与 band deficit，不再先做一段 coverage-only 选择。

## 4. 干净的四级消融

- `paper_mi`: MI scalar score only；
- `paper_mi_gb`: granular-ball scalar score only；
- `paper_mi_gb_lcb`: true LCB scalar score only；
- `paper_full`: LCB + frequency coverage + global budget。

前三个方法不会再偷偷经过 coverage，只有 `paper_full` 使用 `--paper_final_keep_strategy`。

## 推荐运行

```bash
bash scripts/run_v7_clean_ablation.sh
```

快速功能检查可用：

```bash
LCB_REPEATS=3 bash scripts/run_v7_clean_ablation.sh --paper_score_nsamples 32
```

## 新增报告

- `global_budget_summary.json`: 全局总预算、每层实际剪枝数量和各方法 coverage；
- `global_selection_status.csv`: 四种方法最终每个结构单元的 prune flag；
- `mask_overlap.csv`: 现在包含 LCB vs Full 的 mask overlap；
- `ablation_ppl_summary.csv`: 四级消融 PPL。
