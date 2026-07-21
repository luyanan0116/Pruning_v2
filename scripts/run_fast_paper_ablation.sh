#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NCPU=$(nproc)
GB_WORKERS=${GB_WORKERS:-$(( NCPU < 32 ? NCPU : 32 ))}
LCB_WORKERS=${LCB_WORKERS:-4}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -u run_paper_ablation.py \
  --model /root/dw2/Lya/models/Llama-2-7b \
  --c4_path /root/dw2/Lya/dataset/dataset_c4 \
  --wikitext2_path /root/dw2/Lya/dataset/dataset_wikitext-raw \
  --cache_dir /root/dw2/Lya/models/cache \
  --output_dir results/paper_aligned_v6_fast_s050 \
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
  --paper_fast_small_mi \
  --paper_kde_scope probe \
  --paper_kde_bandwidth_scale 1.0 \
  --paper_purity_thresholds 0.65,0.75,0.85 \
  --paper_min_ball_size 8 \
  --paper_max_balls 64 \
  --paper_max_ball_depth 5 \
  --paper_gb_localization unit_local \
  --paper_gb_workers "$GB_WORKERS" \
  --paper_gb_chunk_size 64 \
  --paper_lcb_workers "$LCB_WORKERS" \
  --paper_min_event_classes 2 \
  --paper_min_purity_gain 0.0 \
  --paper_min_radius_reduction 0.0 \
  --paper_compactness_ratio 0.55 \
  --paper_sample_fraction 0.8 \
  --paper_scenario_fraction 1.0 \
  --n_samples_lcb 10 \
  --lcb_lambda 1.0 \
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
