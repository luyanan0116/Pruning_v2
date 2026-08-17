#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_PATH=${MODEL_PATH:-/root/dw2/Lya/models/Llama-2-7b}
C4_PATH=${C4_PATH:-/root/dw2/Lya/dataset/dataset_c4}
WIKITEXT2_PATH=${WIKITEXT2_PATH:-/root/dw2/Lya/dataset/dataset_wikitext-raw}
CACHE_DIR=${CACHE_DIR:-/root/dw2/Lya/models/cache}
OUTPUT_DIR=${OUTPUT_DIR:-results/paper_v7_clean_s050}
LCB_REPEATS=${LCB_REPEATS:-10}
GB_WORKERS=${GB_WORKERS:-16}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

python -u run_paper_ablation.py \
  --model "$MODEL_PATH" \
  --c4_path "$C4_PATH" \
  --wikitext2_path "$WIKITEXT2_PATH" \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --seed 0 \
  --sparsity_ratio 0.50 \
  --paper_mask_style wanda_weight \
  --paper_prune_targets mlp,attention \
  --mlp_sparsity_ratio 0.50 \
  --attention_sparsity_ratio 0.50 \
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
  --paper_purity_thresholds 0.65,0.75,0.85 \
  --paper_min_ball_size 8 \
  --paper_max_balls 64 \
  --paper_max_ball_depth 5 \
  --paper_gb_localization unit_local \
  --paper_gb_workers "$GB_WORKERS" \
  --paper_gb_chunk_size 64 \
  --paper_min_event_classes 2 \
  --paper_min_event_count_per_ball 2 \
  --paper_gb_fusion_mode inverse_sqrt_dispersion \
  --paper_gb_fusion_max_ratio 5.0 \
  --paper_lcb_repeats "$LCB_REPEATS" \
  --paper_lcb_sample_fraction 0.80 \
  --paper_lcb_scenario_fraction 0.67 \
  --lcb_lambda 0.5 \
  --paper_global_budget \
  --paper_global_layer_spread 0.10 \
  --paper_final_keep_strategy paper_hybrid \
  --paper_band_coverage_ratio 0.90 \
  --paper_coverage_alpha 0.10 \
  --paper_greedy_batches 64 \
  --paper_wanda_sequential \
  --paper_wanda_nsamples 128 \
  --paper_wanda_seqlen 2048 \
  --paper_wanda_row_spread 0.02 \
  --paper_wanda_guidance_strength 0.04 \
  --paper_wanda_temperature 1.0 \
  --paper_wanda_chunk_rows 256 \
  --paper_prune_step 10 \
  --paper_plot_layers first,middle,last \
  "$@"
