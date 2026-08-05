# 频域互信息结构化剪枝：两份研究方案联合对齐版

本代码只保留两份研究方案中的结构单元流程：

1. 对注意力头输出和前馈中间通道计算任务损失梯度响应；
2. 按样本、按结构单元在序列维标准化；
3. 对每条样本—场景按原生序列长度执行 DCT-II，不在 DCT 前做池化、插值或长度重采样；
4. 对频域系数平方并汇聚为细粒度频率桶；
5. 根据任务事件互信息，以最小合并损失贪心合并相邻频率桶；
6. 计算各频带互信息贡献谱与加权总贡献；
7. 在每个结构单元自己的频带响应空间中构建多粒度粒球；
8. 在粒球内估计局部互信息，并按样本占比汇聚；
9. 对粒球多粒度结果进行融合；完整方法可按重复估计方差自适应确定融合权重，严格消融时固定为等权或手工权重；
10. 完整方法对基础样本和输入场景进行双源重复抽样，计算均值、标准差与 LCB；
11. 在频带覆盖和参数量、FLOPs、显存、KV-cache、实测时延预算下生成全局保留集合；
12. 删除完整注意力头或完整前馈通道，并开展剪前/剪后贡献谱、排序和端到端性能验证。

代码中不存在权重级非结构化评分、权重掩码引导或相关回退路径。

## 完整方法与消融方法

- `paper_mi`：频域互信息贡献谱；
- `paper_mi_gb`：频域互信息 + 粒球多粒度局部化；
- `paper_mi_gb_lcb`：频域互信息 + 粒球多粒度局部化 + 样本—场景双源重复估计 + LCB。该项对应完整流程。


## 三组 PPL 严格消融

推荐直接运行：

```bash
python run_paper_ablation.py \
  --model /path/to/model \
  --c4_path /path/to/c4 \
  --wikitext2_path /path/to/wikitext-2-raw \
  --prune_ratio 0.15 \
  --paper_granularity_weight_mode equal \
  --paper_lcb_repeats 20 \
  --paper_score_nsamples 128 \
  --paper_calib_seqlen 2048 \
  --seqlen 2048 \
  --output_dir outputs/strict_ablation
```

这条命令会依次启动三个全新的模型实例，并共享同一份梯度响应缓存：

1. `paper_mi`：只执行频域互信息，不构粒球，不重复估计，不计算 LCB；
2. `paper_mi_gb`：执行互信息、粒球局部化和多粒度融合，不重复估计，不计算 LCB；
3. `paper_mi_gb_lcb`：在第二组基础上增加样本—场景重复估计、均值/标准差和 LCB。

严格消融默认使用 `equal` 融合权重，使第二组和第三组之间唯一新增的评分步骤是重复估计与 LCB。输出包括：

- `ablation_summary.csv`：三组 PPL、相对未剪枝模型增量、相对 MI 和前一阶段的增量；
- `ablation_summary.json`：实验协议和实际执行模块；
- 每组 `score_report/executed_stage_manifest.json`：逐层记录真正运行过的模块，用于检查消融污染；
- 每组 `run_result.json`：完整评测结果。

若要运行研究方案中的方差自适应粒度融合，可在单独的完整方法实验中使用：

```bash
--paper_granularity_weight_mode repeat_variance
```

不建议在严格三阶段消融中使用该模式，因为第二组会为估计粒度权重调用重复估计，此时“MI+粒球”和“MI+粒球+LCB”之间不再只相差 LCB。

## 安装

```bash
pip install -r requirements.txt
```

## 一次运行完整闭环

```bash
python main.py \
  --model /path/to/model \
  --c4_path /path/to/c4 \
  --wikitext2_path /path/to/wikitext-2-raw \
  --prune_method paper_mi_gb_lcb \
  --prune_ratio 0.15 \
  --paper_mask_style structured_zero \
  --paper_score_nsamples 128 \
  --paper_calib_seqlen 2048 \
  --paper_scenario_ratios 0.5,0.75,1.0 \
  --paper_scenario_crops prefix,center,suffix \
  --paper_lcb_repeats 20 \
  --paper_budget_metrics params,flops,memory,kv_cache \
  --paper_band_coverage_ratio 0.90 \
  --paper_post_prune_validate \
  --eval_before \
  --eval_after \
  --output_dir outputs/full_run
```

`structured_zero` 删除完整结构单元的所有耦合张量但保持模型形状，便于直接评测和剪前/剪后对齐。`structured_surgery` 会物理缩小矩阵，能够反映参数和计算形状变化，但需要定制导出或推理运行时。

## 样本—场景双源不确定性

要真正研究任务、语言、提示模板、上下文构造等场景变化，应提供配对场景清单。相同原始样本在不同场景下必须使用相同 `base_sample_id`：

```json
{"text":"场景A构造后的文本", "base_sample_id":"sample-001", "scenario_id":"prompt-A", "task":"qa", "language":"zh"}
{"text":"场景B构造后的文本", "base_sample_id":"sample-001", "scenario_id":"prompt-B", "task":"qa", "language":"zh"}
```

运行时加入：

```bash
--paper_scenario_manifest paired_scenarios.jsonl
```

代码先对基础样本重采样，再对其场景实现重采样，保留同一基础样本多个场景之间的依赖关系。

## 多维部署预算

默认同时约束参数量、FLOPs 和显存：

```bash
--paper_budget_metrics params,flops,memory
```

可为每种资源指定不同保留比例：

```bash
--paper_budget_keep_ratios 0.85,0.80,0.82
```

若使用时延约束，必须提供实测配置，代码不会猜测时延：

```csv
unit_type,layer,unit,latency
mlp,,,0.000012
attention,,,0.000180
```

然后使用：

```bash
--paper_budget_metrics params,flops,memory,latency \
--paper_latency_profile measured_latency.csv
```

## 最大稳定剪枝率与临界区间

```bash
python run_paper_validation.py \
  --model /path/to/model \
  --c4_path /path/to/c4 \
  --wikitext2_path /path/to/wikitext-2-raw \
  --prune_ratios 0.05,0.10,0.15,0.20,0.25 \
  --seqlens 512,1024,2048 \
  --seeds 0,1,2 \
  --relative_ppl_limit 0.05
```

输出：

- `validation_curve.csv`：每个种子、上下文长度和剪枝率的性能；
- `validation_summary.json`：最大稳定剪枝率和相邻性能突降区间。

## 公式对齐与一个必须公开的口径问题

详细对应关系见 `FORMULA_TRACEABILITY.md`，版本差异与补充项见 `ALIGNMENT_AUDIT.md`。

其中一份方案先定义位置级 NLL 事件 `Y'_t`，但 DCT 后的频带能量是每个样本—场景一条向量，原文没有规定如何把位置级事件与样本级频带能量一一配对。当前实现采用可执行且不产生伪重复样本的口径：先保留位置级 NLL 与位置级分位事件用于诊断，再用该样本—场景的平均位置 NLL 做全局分位离散，得到进入互信息估计的样本级事件 `Y`。这一处不是代码遗漏，而是两份文字方案未给出唯一数学对齐方式；代码和元数据中均明确记录该选择。

## 测试

```bash
pytest -q
```

测试覆盖 DCT 公式、平方能量、熵分解近邻互信息、任务相关相邻合并、原始频带响应空间构球、三阶段严格隔离、重复估计方差自适应融合、双源重复估计、LCB、精确懒惰贪心预算、完整结构单元删除、物理缩形和场景清单解析。
