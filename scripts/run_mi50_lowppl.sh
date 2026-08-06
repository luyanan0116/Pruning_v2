#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/dw2/Lya/models/Llama-2-7b}"
C4_PATH="${C4_PATH:-/root/dw2/Lya/dataset/dataset_c4}"
WIKI_PATH="${WIKI_PATH:-/root/dw2/Lya/dataset/dataset_wikitext-raw}"
OUT="${OUT:-results/paper_mi50_s050_g0001}"
GUIDANCE="${GUIDANCE:-0.001}"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u main.py \
  --model "$MODEL_PATH" \
  --c4_path "$C4_PATH" \
  --wikitext2_path "$WIKI_PATH" \
  --cache_dir /root/dw2/Lya/models/cache \
  --prune_method paper_mi50 \
  --sparsity_ratio 0.50 \
  --mlp_sparsity_ratio 0.50 \
  --attention_sparsity_ratio 0.50 \
  --sparsity_type unstructured \
  --paper_mask_style wanda_weight \
  --paper_prune_targets mlp,attention \
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
  --paper_band_coverage_ratio 0.90 \
  --paper_coverage_alpha 0.25 \
  --paper_greedy_batches 64 \
  --paper_wanda_sequential \
  --paper_wanda_nsamples 128 \
  --paper_wanda_seqlen 2048 \
  --paper_mi50_guidance_strength "$GUIDANCE" \
  --paper_wanda_chunk_rows 256 \
  --paper_cache_dir paper_response_cache_mi50 \
  --paper_report_dir results/paper_mi50_shared_report \
  --save "$OUT"
