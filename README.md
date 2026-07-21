# Pruning

面向 Llama / Mistral 类模型的频域互信息、粒球局部化与 LCB 稳健结构化剪枝实验代码。

本版本新增了与论文技术路线对应的三组消融：

| 方法 | 命令参数 | 实际评分路径 |
|---|---|---|
| 只使用互信息 | `paper_mi` | 梯度响应 → 样本内标准化 → DCT → 频率桶能量 → 相邻频带合并 → MI |
| 互信息 + 粒球 | `paper_mi_gb` | 上述步骤 → 每层粒球局部化 → 球内 MI → 多粒度融合 |
| 互信息 + 粒球 + LCB | `paper_mi_gb_lcb` | 上述步骤 → 事件/场景分层重复抽样 → 每次重建粒球 → 均值、标准差、LCB |

旧参数 `--prune_method lcb` 已映射到 `paper_mi_gb_lcb`。

## 关键修改

### 1. 任务损失梯度响应

剪枝单元统一定义为每层 MLP 中间通道。代码在 `mlp.down_proj` 输入处采集：

```text
r_i(t) = z_i(t) × ∂L_task / ∂z_i(t)
```

其中 `z_i(t)` 是第 `i` 个 MLP 通道在序列位置 `t` 的输出，`L_task` 为因果语言模型损失。响应序列会自适应压缩并保存为内存映射文件，避免一次性占满 CPU 内存。

### 2. 频域互信息

实现位置：`lib/paper_pruning/mi.py`

流程包括：

1. 每个样本、每个结构单元单独标准化；
2. 沿序列位置执行正交 DCT-II；
3. 按 DCT 系数平方和统计频率桶能量；
4. 根据任务事件相关性合并相邻频率桶；
5. 使用 kNN 互信息作为主估计器，并用高斯核密度互信息作为辅助估计器；
6. 对频带贡献加权汇总。

任务事件 `Y` 默认由每条校准序列的因果语言模型损失进行分位数离散化得到。为了避免上下文长度直接决定事件类别，事件分箱在每个场景内部独立完成。

### 3. 粒球多粒度局部化

实现位置：`lib/paper_pruning/granular_ball.py`

粒球按层构建，而不是整个模型只构建一次，也不是每个通道单独聚类一次：

```text
每一层一套响应空间
同层通道共享粒球划分
各通道在球内分别估计互信息
```

粒球分裂同时考虑：

- 事件纯度是否达到当前阈值；
- 球半径是否足够紧致；
- 分裂后事件纯度是否提升；
- 分裂后平均半径是否下降；
- 子球最小样本数和局部事件类别数。

`0.65,0.75,0.85` 三个阈值使用嵌套层级，细粒度在粗粒度结果上继续分裂，因此粒球数量不会随纯度阈值升高而减少。

### 4. LCB 稳健排序

每次重复估计都会重新执行：

```text
事件 + 场景联合分层抽样
→ 重新合并频带
→ 重新构建粒球
→ 重新计算局部互信息
```

最终计算：

```text
LCB_i = mean(S_i) - lambda × std(S_i)
```

所有通道的全局 MI、粒球贡献、重复估计均值、标准差和 LCB 都会导出。

### 5. 每层按 10 个索引分批

`--paper_prune_step 10` 会把每层最终选择的索引按 10 个一批写入：

```text
prune_batches_step10.json
```

最终剪枝数量有两种模式：

```text
--prune_per_layer 10
```

表示每层总共剪 10 个 MLP 通道。

```text
--prune_per_layer 0 --sparsity_ratio 0.15
```

表示每层剪除 15% 的 MLP 通道，但索引仍按每批 10 个导出。为了观察明显 PPL 差异，推荐使用固定比例，而不是只剪 10 个通道。只剪 10 个通道通常过于轻微，PPL 四舍五入后可能完全相同。

## 一键运行三组消融

```bash
export C4_PATH=/你的路径/dataset_c4
export WIKITEXT2_PATH=/你的路径/dataset_wikitext-raw

python run_paper_ablation.py \
  --model /你的路径/Llama-2-7b-hf \
  --cache_dir llm_weights \
  --sparsity_ratio 0.15 \
  --prune_per_layer 0 \
  --output_dir results/paper_ablation \
  --paper_score_nsamples 32 \
  --paper_calib_seqlen 512 \
  --paper_scenario_ratios 0.5,0.75,1.0 \
  --n_samples_lcb 10
```

脚本会为三个方法分别重新加载原始模型，避免在同一个模型上连续剪枝。同时复用同一份梯度响应缓存与同一套三路评分结果，保证消融公平。

结果汇总：

```text
results/paper_ablation/ablation_ppl_summary.csv
```

## 单独运行一种方法

```bash
python main.py \
  --model /你的路径/Llama-2-7b-hf \
  --prune_method paper_mi_gb_lcb \
  --sparsity_type unstructured \
  --sparsity_ratio 0.15 \
  --prune_per_layer 0 \
  --paper_prune_step 10 \
  --paper_cache_dir results/shared_response_cache \
  --paper_report_dir results/shared_score_report \
  --save results/logs
```

将 `paper_mi_gb_lcb` 替换为 `paper_mi` 或 `paper_mi_gb` 即可运行另外两组。

## 输出文件

`--paper_report_dir` 下会生成：

```text
all_contribution_scores.csv
  每层、每个 MLP 通道的 MI、粒球贡献、LCB 均值、标准差、LCB 和剪枝标记

granular_ball_summary.csv
  每层、每个粒度、每个粒球的大小、纯度、半径、深度和主事件

prune_indices.json
  三种方法每层最终剪枝索引

prune_batches_step10.json
  每层按 10 个索引分批后的剪枝计划

mask_overlap.csv
  三种方法剪枝掩码的 Jaccard 相似度和变化索引数量

layer_000_balls.png
layer_000_lcb.png
  浅层、中层、深层的局部粒球图和 LCB 贡献图
```

如果三组 PPL 仍然相同，先检查 `mask_overlap.csv`。当两组掩码完全一致时，PPL 相同是正常结果。代码不会通过人为扰动分数来伪造递减曲线。

## 结构化剪枝口径

当前实现同步置零：

```text
gate_proj 对应行
up_proj 对应行
down_proj 对应列
```

这是形状保持的结构单元消融，适合比较 PPL 与验证排序。若要获得真实推理加速，还需要在确认最终索引后物理裁切权重张量，并同步更新模型配置。

## 测试

```bash
PYTHONPATH=. pytest -q tests/test_paper_pruning.py
```

当前测试覆盖：频域互信息、纯度驱动粒球层级、三路消融差异、LCB 方差、梯度响应采集以及 MLP 通道同步置零。

## Structured MLP + attention-head pruning (v3)

The paper-aligned methods now treat both LLaMA MLP intermediate channels and complete attention heads as prunable structural units.

- MLP response: input of `mlp.down_proj` multiplied by its task-loss gradient.
- Attention response: input of `self_attn.o_proj`, reshaped by head, multiplied by its gradient and summed over `head_dim`.
- MLP mask: zero matching `gate_proj`/`up_proj` rows and `down_proj` columns.
- Attention mask: zero matching `q_proj`/`k_proj`/`v_proj` rows and `o_proj` columns.
- Tensor shapes are preserved for PPL ablation. Physical slicing is still required for wall-clock speedup.

The default paper targets are now:

```text
--paper_prune_targets mlp,attention
```

With standard Llama-2-7B attention, this command gives 50% structured sparsity in both parts:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u run_paper_ablation.py \
  --model /root/dw2/Lya/models/Llama-2-7b \
  --c4_path /root/dw2/Lya/dataset/dataset_c4 \
  --wikitext2_path /root/dw2/Lya/dataset/dataset_wikitext-raw \
  --cache_dir /root/dw2/Lya/models/cache \
  --output_dir results/paper_ablation_mlp_attn_s050 \
  --sparsity_ratio 0.50 \
  --mlp_sparsity_ratio 0.50 \
  --attention_sparsity_ratio 0.50 \
  --prune_per_layer 0 \
  --attention_prune_per_layer 0 \
  --paper_prune_targets mlp,attention \
  --paper_score_nsamples 16 \
  --paper_calib_seqlen 256 \
  --paper_response_length 16 \
  --paper_num_bins 8 \
  --paper_num_bands 3 \
  --paper_scenario_ratios 0.5,1.0 \
  --paper_min_ball_size 4 \
  --paper_max_balls 32 \
  --n_samples_lcb 5 \
  --overwrite
```

Old MLP-only response caches are intentionally rejected. Use a new output directory or pass `--overwrite` so attention responses are collected.

The exact q/k/v/o head-mask implementation currently requires `num_attention_heads == num_key_value_heads`. This includes Llama-2-7B. Grouped-query-attention models are rejected to avoid silently applying an invalid K/V mask.
