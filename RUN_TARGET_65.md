# 稀疏度 0.5 与低 PPL 运行方案

## 先验证普通 Wanda 基线

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --model /root/dw2/Lya/models/Llama-2-7b \
  --c4_path /root/dw2/Lya/dataset/dataset_c4 \
  --wikitext2_path /root/dw2/Lya/dataset/dataset_wikitext-raw \
  --cache_dir /root/dw2/Lya/models/cache \
  --prune_method wanda \
  --sparsity_ratio 0.50 \
  --sparsity_type unstructured \
  --nsamples 128 \
  --seqlen 2048 \
  --save results/wanda_baseline_s050
```

只有该基线已经接近 6.5，论文引导版本才有现实机会保持在附近。

## 推荐正式消融命令

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u run_paper_ablation.py \
  --model /root/dw2/Lya/models/Llama-2-7b \
  --c4_path /root/dw2/Lya/dataset/dataset_c4 \
  --wikitext2_path /root/dw2/Lya/dataset/dataset_wikitext-raw \
  --cache_dir /root/dw2/Lya/models/cache \
  --output_dir results/paper_aligned_v5_s050 \
  --seed 0 \
  --sparsity_ratio 0.50 \
  --paper_mask_style wanda_weight \
  --paper_prune_targets mlp,attention \
  --mlp_sparsity_ratio 0.50 \
  --attention_sparsity_ratio 0.50 \
  --prune_per_layer 0 \
  --attention_prune_per_layer 0 \
  --seqlen 2048 \
  --paper_calib_dataset c4 \
  --paper_score_nsamples 64 \
  --paper_calib_seqlen 512 \
  --paper_response_length 32 \
  --paper_scenario_ratios 0.5,0.75,1.0 \
  --paper_event_bins 3 \
  --paper_num_bins 16 \
  --paper_num_bands 4 \
  --paper_probe_units 64 \
  --paper_mi_neighbors 3 \
  --paper_kde_bandwidth_scale 1.0 \
  --paper_purity_thresholds 0.65,0.75,0.85 \
  --paper_min_ball_size 8 \
  --paper_max_balls 64 \
  --paper_max_ball_depth 5 \
  --paper_gb_localization unit_local \
  --paper_gb_workers 8 \
  --paper_min_event_classes 2 \
  --paper_min_purity_gain 0.0 \
  --paper_min_radius_reduction 0.0 \
  --paper_compactness_ratio 0.55 \    --lcb_lambda 1.0 \
  --paper_band_coverage_ratio 0.90 \
  --paper_coverage_alpha 0.25 \
  --paper_greedy_batches 64 \
  --paper_wanda_sequential \
  --paper_wanda_nsamples 128 \
  --paper_wanda_seqlen 2048 \
  --paper_wanda_row_spread 0.01 \
  --paper_wanda_guidance_strength 0.01 \
  --paper_wanda_temperature 1.0 \
  --paper_wanda_chunk_rows 256 \
  --paper_prune_step 10 \
  --paper_plot_layers first,middle,last \
  --overwrite
```

## PPL 仍偏高时的诊断阶梯

先仅将以下两项置零，其余完全不动：

```bash
--paper_wanda_row_spread 0.0 \
--paper_wanda_guidance_strength 0.0
```

此时 paper mask 应退化为标准顺序 Wanda。若 PPL 回到约 6.5，说明 MI/GB/LCB 评分本身没有计算错误，问题来自引导强度。按顺序测试：

```text
0.000 -> 0.002 -> 0.005 -> 0.010
```

`row_spread` 建议比 `guidance_strength` 更保守，优先固定为 `0.0`，只让分数影响 `down_proj/o_proj` 的列级权重选择。

## 严格结构单元模式

```bash
--paper_mask_style structured_unit
```

该模式会删除完整头与完整 FFN 通道，更贴近青基方法语义，但 50% 通常会显著抬升 PPL。青基文件本身的预期结构单元削减范围是 10%-20%。
