# Pruning v4: 论文评分 + Wanda 风格全线性层权重剪枝

本版本保留三组论文消融评分：

1. `paper_mi`：频域互信息；
2. `paper_mi_gb`：频域互信息 + 多粒度粒球局部化；
3. `paper_mi_gb_lcb`：频域互信息 + 粒球 + 重复估计 LCB。

同时新增两种掩码口径：

| 参数 | 作用 |
|---|---|
| `--paper_mask_style wanda_weight` | 推荐。像 Wanda 一样对 Transformer 层中的线性权重做非结构化剪枝，覆盖注意力 `q/k/v/o` 和 MLP `gate/up/down`。|
| `--paper_mask_style structured_unit` | 论文结构单元消融。完整置零 MLP 中间通道和注意力头，50% 时非常激进。|

## Wanda 风格与“剪 50% 注意力头”的区别

Wanda 的 50% 指每个线性权重矩阵中的权重稀疏度，不是删除 50% 完整注意力头。官方 Wanda 会递归处理 Transformer block 内全部 `nn.Linear`，包括：

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
```

本项目的 `wanda_weight` 模式采用：

```text
基础权重重要性 = |W| × sqrt(校准输入二阶矩)
```

再用 MI、MI+粒球或 MI+粒球+LCB 的结构贡献分数调节权重预算落点：

- `gate_proj`、`up_proj`：低贡献 MLP 单元的输出行获得更高稀疏度；
- `down_proj`：低贡献 MLP 单元对应输入列更容易被剪；
- `q_proj`、`k_proj`、`v_proj`：低贡献注意力头对应行获得更高稀疏度；
- `o_proj`：低贡献注意力头对应输入列更容易被剪。

总权重预算保持不变，但不会强制整头或整通道完全归零，因此 50% 时通常远比结构化删除 50% 头和通道稳定。

## 50% MLP + Attention 权重剪枝命令

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u run_paper_ablation.py \
  --model /root/dw2/Lya/models/Llama-2-7b \
  --c4_path /root/dw2/Lya/dataset/dataset_c4 \
  --wikitext2_path /root/dw2/Lya/dataset/dataset_wikitext-raw \
  --cache_dir /root/dw2/Lya/models/cache \
  --output_dir results/paper_wanda_all_linear_s050 \
  --seed 0 \
  --sparsity_ratio 0.50 \
  --paper_mask_style wanda_weight \
  --paper_prune_targets mlp,attention \
  --mlp_sparsity_ratio 0.50 \
  --attention_sparsity_ratio 0.50 \
  --prune_per_layer 0 \
  --attention_prune_per_layer 0 \
  --paper_calib_dataset c4 \
  --paper_score_nsamples 16 \
  --paper_calib_seqlen 256 \
  --paper_response_length 16 \
  --paper_event_bins 3 \
  --paper_num_bins 8 \
  --paper_num_bands 3 \
  --paper_mi_neighbors 3 \
  --paper_scenario_ratios 0.5,1.0 \
  --paper_purity_thresholds 0.60,0.75,0.90 \
  --paper_min_ball_size 4 \
  --paper_max_balls 32 \
  --paper_sample_fraction 0.8 \
  --n_samples_lcb 5 \
  --lcb_lambda 1.0 \
  --paper_wanda_score_floor 0.05 \
  --paper_wanda_row_spread 0.8 \
  --paper_wanda_temperature 2.0 \
  --paper_wanda_chunk_rows 256 \
  --paper_plot_layers first,middle,last \
  --overwrite
```

关键参数：

```text
--paper_mask_style wanda_weight
--paper_prune_targets mlp,attention
--mlp_sparsity_ratio 0.50
--attention_sparsity_ratio 0.50
```

这表示 MLP 的三个线性矩阵和 Attention 的四个线性矩阵都做 50% 权重剪枝，最终 `check_sparsity()` 应接近 `0.5000`。

## 正式实验建议

流程验证可先使用上面的 `16 × 256 × 2 场景 × 5 次 LCB`。正式结果建议逐步提高到：

```text
--paper_score_nsamples 32
--paper_calib_seqlen 512
--paper_scenario_ratios 0.5,0.75,1.0
--n_samples_lcb 10
```

这些参数会显著增加计算时间。稀疏度只影响最终掩码，不会显著减少 MI、粒球和 LCB 的评分成本。

## 输出文件

`shared_score_report` 中包括：

```text
all_contribution_scores.csv
  每层、每个 MLP 单元/注意力头的 MI、粒球贡献、LCB 均值、标准差与 LCB。

unit_scores.npz
  三组方法用于 Wanda 风格权重掩码的完整单位贡献向量。

weight_mask_summary_paper_mi.csv
weight_mask_summary_paper_mi_gb.csv
weight_mask_summary_paper_mi_gb_lcb.csv
  每层每个 q/k/v/o、gate/up/down 矩阵的目标和实际权重稀疏度。

prune_indices.json
  低贡献结构单元的排序优先集合。wanda_weight 模式下它用于解释预算落点，
  不表示这些完整头或通道被整体置零。

granular_ball_summary.csv
layer_*_balls.png
layer_*_lcb.png
```

## 原版 Wanda 基线

项目仍保留原始 `--prune_method wanda`。该方法直接使用 Wanda 的权重与激活度量，不使用论文的 MI、粒球或 LCB：

```bash
C4_PATH=/root/dw2/Lya/dataset/dataset_c4 \
WIKITEXT2_PATH=/root/dw2/Lya/dataset/dataset_wikitext-raw \
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model /root/dw2/Lya/models/Llama-2-7b \
  --cache_dir /root/dw2/Lya/models/cache \
  --prune_method wanda \
  --sparsity_ratio 0.50 \
  --sparsity_type unstructured \
  --save results/wanda_baseline
```

## 缓存兼容性

v4 响应缓存除任务梯度响应外，还保存七个线性模块的输入二阶矩。旧版 v2/v3 缓存会被拒绝。首次运行必须使用新输出目录，或加：

```text
--overwrite
```

## 测试

```bash
PYTHONPATH=. pytest -q
```

当前测试覆盖频域互信息、粒球层级、LCB、MLP/Attention 梯度响应、七个线性模块的激活统计、结构化掩码和 Wanda 风格全线性层权重掩码。
