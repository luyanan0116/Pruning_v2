# Pruning paper-aligned v7

当前版本已恢复真正的样本-场景双源 bootstrap/LCB，并把配置生成改为跨层全局预算。消融固定为 **MI → MI+GB → MI+GB+LCB → Full(LCB+频带覆盖)**。完整改动见 `V7_GLOBAL_BOOTSTRAP_ABLATION.md`，推荐入口为：

```bash
bash scripts/run_v7_clean_ablation.sh
```

关键新增参数：

```text
--paper_lcb_repeats 10
--paper_lcb_sample_fraction 0.8
--paper_lcb_scenario_fraction 0.67
--paper_min_event_count_per_ball 2
--paper_gb_fusion_mode inverse_sqrt_dispersion
--paper_gb_fusion_max_ratio 5
--paper_global_budget
--paper_global_layer_spread 0.10
```

测试：`PYTHONPATH=. pytest -q`。当前包内测试覆盖真实 LCB 方差、invalid-ball 重新归一化、融合权重上限和 coverage+scalar 联合贪心。

---

# Pruning paper-aligned v6 fast

This release keeps the v5 paper-aligned MI → granular-ball → repeated LCB → coverage/budget pipeline, while accelerating the strict per-unit granular-ball implementation. See `FAST_ACCELERATION_REPORT.md`.

Recommended command for the paths used in this project:

```bash
bash scripts/run_fast_paper_ablation.sh
```

Important fast-path controls:

```text
--paper_fast_small_mi       exact Numba JIT for small ball-level kNN MI
--paper_kde_scope probe     KDE only on representative units; does not affect masks/PPL
--paper_gb_workers 16-32    total unit-local CPU worker budget
--paper_gb_chunk_size 64    amortizes thread scheduling overhead single-pass LCB-compatible scoring (no repeats)
```

# Pruning Paper-Aligned v5

基于 Wanda 工程骨架实现的论文对齐版本，主要对应：

- 频域互信息任务贡献谱；
- 粒球多粒度局部互信息；
- 样本-场景双源重复估计与 LCB；
- 频带覆盖与预算约束下的贪心配置；
- 顺序 Wanda 50% 权重剪枝低 PPL 验证路径；
- 完整注意力头/FFN 通道结构化剪枝验证路径。

## 首先阅读

- `PAPER_ALIGNMENT_REPORT.md`：逐公式、逐要求核对和修改说明。
- `RUN_TARGET_65.md`：0.5 稀疏度与低 PPL 的运行命令和诊断阶梯。
- `scripts/run_wanda_baseline.sh`：先确认服务器上的普通 Wanda 基线。
- `scripts/run_target65_ablation.sh`：运行 MI、MI+GB、MI+GB+LCB 三组实验。

## 重要口径

`wanda_weight` 表示七个线性矩阵中的 50% 权重稀疏，与 Wanda 的 PPL 口径接近。

`structured_unit` 表示删除完整注意力头和 FFN 中间通道，更贴近论文“结构单元”语义，但完整结构单元 50% 剪除通常远比 Wanda 50% 权重稀疏激进。

## 安装

```bash
conda activate lya_pruningv2
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 测试

```bash
PYTHONPATH=. pytest -q
```

当前代码包包含 12 个单元测试，覆盖频域 MI、粒球嵌套划分、双源 bootstrap、LCB、覆盖预算、注意力/MLP mask 与 Wanda 回退一致性。

## 快速启动

```bash
bash scripts/run_wanda_baseline.sh
bash scripts/run_target65_ablation.sh
```

脚本中的模型和数据路径已经按以下服务器路径填写：

```text
/root/dw2/Lya/models/Llama-2-7b
/root/dw2/Lya/dataset/dataset_c4
/root/dw2/Lya/dataset/dataset_wikitext-raw
```

## 主要结果文件

```text
results/.../ablation_ppl_summary.csv
results/.../shared_score_report/all_contribution_scores.csv
results/.../shared_score_report/granular_ball_summary.csv
results/.../shared_score_report/prune_indices.json
results/.../shared_score_report/prune_batches_step10.json
results/.../shared_score_report/mask_overlap.csv
results/.../shared_score_report/weight_mask_summary_*.csv
```

## 真实性说明

本代码包在本地完成了静态检查和单元测试，但没有用户服务器上的 Llama-2-7B、C4、WikiText-2 和 A100，因此没有宣称已经复现 PPL=6.5。请先运行普通 Wanda 基线，再运行论文引导消融。
