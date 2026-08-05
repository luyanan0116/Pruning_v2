# MI、MI+粒球、MI+粒球+LCB 三阶段 PPL 消融

## 1. 三组方法的唯一差异

| 方法 | 频域互信息 | 粒球局部化 | 多粒度融合 | 双源重复估计 | LCB |
|---|---:|---:|---:|---:|---:|
| `paper_mi` | 是 | 否 | 否 | 否 | 否 |
| `paper_mi_gb` | 是 | 是 | 是 | 否 | 否 |
| `paper_mi_gb_lcb` | 是 | 是 | 是 | 是 | 是 |

三组实验共同使用同一个模型、校准样本、场景构造、梯度响应缓存、DCT、频带合并、结构化剪枝预算和 PPL 评测设置。

## 2. 为什么严格消融固定粒度融合权重

研究方案允许粒度融合权重由估计方差或排序稳定性自适应确定。若在第二组中利用重复估计方差计算权重，第二组已经调用了重复估计，无法再把第三组的变化单独解释为 LCB 的作用。

因此严格消融默认采用等权：

\[
\rho_m=\frac{1}{M}.
\]

也可通过 `--paper_granularity_weight_mode manual` 和 `--paper_granularity_weights` 固定一组手工权重。第二组和第三组必须使用完全相同的权重。

完整方法复现实验可单独采用：

```bash
--paper_granularity_weight_mode repeat_variance
```

## 3. 一键运行

```bash
python run_paper_ablation.py \
  --model /path/to/model \
  --c4_path /path/to/c4 \
  --wikitext2_path /path/to/wikitext-2-raw \
  --prune_ratio 0.15 \
  --paper_granularity_weight_mode equal \
  --paper_score_nsamples 128 \
  --paper_calib_seqlen 2048 \
  --paper_lcb_repeats 20 \
  --seqlen 2048 \
  --seed 0 \
  --output_dir outputs/strict_ablation
```

## 4. 输出字段

`ablation_summary.csv` 中：

- `baseline_ppl`：未剪枝模型 PPL；
- `pruned_ppl`：当前方法剪枝后的 PPL；
- `absolute_ppl_increase`：相对未剪枝模型的绝对 PPL 增量；
- `relative_ppl_increase`：相对未剪枝模型的比例增量；
- `ppl_delta_vs_mi`：相对只用 MI 的 PPL 差；
- `ppl_delta_vs_previous_stage`：加入当前模块后相对上一阶段的 PPL 变化。

PPL 不保证随模块增加单调下降。粒球或 LCB 是否有效，应由同一剪枝率、多个随机种子下的均值和方差判断，而不是只看一次结果。

## 5. 消融污染检查

每个方法的 `score_report/executed_stage_manifest.json` 会逐层记录实际执行模块。严格模式下脚本会自动检查：

- MI 组不能出现粒球、重复估计或 LCB；
- MI+粒球组不能出现重复估计或 LCB；
- 完整组必须出现重复估计和 LCB。

三次新模型实例的 `baseline_ppl` 还必须在 `--baseline_tolerance` 范围内一致，否则脚本停止并拒绝输出可比较结论。
