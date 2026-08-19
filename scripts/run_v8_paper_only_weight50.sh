#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-2-7b-hf}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-llm_weights}"
C4_PATH="${C4_PATH:-/root/dw2/Lya/dataset/dataset_c4}"
WIKITEXT2_PATH="${WIKITEXT2_PATH:-/root/dw2/Lya/dataset/dataset_wikitext-raw}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper_v82_weight50}"
RESPONSE_CACHE_DIR="${RESPONSE_CACHE_DIR:-${OUTPUT_DIR}/response_cache}"

python run_paper_ablation.py \
  --model "${MODEL_PATH}" \
  --cache_dir "${MODEL_CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --response_cache_dir "${RESPONSE_CACHE_DIR}" \
  --c4_path "${C4_PATH}" \
  --wikitext2_path "${WIKITEXT2_PATH}" \
  --eval_wikitext_split validation \
  --sparsity_ratio 0.50 \
  --seqlen 4096 \
  --wanda_nsamples 128 \
  --wanda_calib_seqlen 4096 \
  --paper_prune_targets mlp,attention \
  --paper_weight_allocation paper_nonuniform \
  --paper_weight_min_unit_sparsity 0.45 \
  --paper_weight_max_unit_sparsity 0.55 \
  --paper_budget_temperature 1.0 \
  --paper_score_nsamples 128 \
  --paper_calib_seqlen 1024 \
  --paper_response_length 128 \
  --paper_scenario_ratios 0.5,0.75,1.0 \
  --paper_event_bins 3 \
  --paper_num_bins 16 \
  --paper_num_bands 4 \
  --paper_mi_neighbors 3 \
  --paper_purity_thresholds 0.65,0.75,0.85 \
  --paper_min_ball_size 8 \
  --paper_max_balls 64 \
  --paper_max_ball_depth 5 \
  --paper_gb_localization unit_local \
  --paper_gb_workers 16 \
  --paper_gb_chunk_size 64 \
  --paper_kde_scope probe \
  --paper_gb_fusion_mode inverse_sqrt_dispersion \
  --paper_gb_fusion_max_ratio 5.0 \
  --paper_lcb_repeats 20 \
  --paper_lcb_sample_fraction 0.80 \
  --paper_lcb_scenario_fraction 0.6666666666666666 \
  --lcb_lambda 0.5 \
  --paper_band_coverage_ratio 0.90 \
  --paper_coverage_alpha 0.10 \
  --paper_greedy_batches 8
