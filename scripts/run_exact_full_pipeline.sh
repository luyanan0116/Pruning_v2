#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:?Set MODEL to a local model directory or model identifier}
C4_PATH=${C4_PATH:?Set C4_PATH}
WIKITEXT2_PATH=${WIKITEXT2_PATH:?Set WIKITEXT2_PATH}
OUTPUT=${OUTPUT:-outputs/exact_full}

python main.py \
  --model "$MODEL" \
  --c4_path "$C4_PATH" \
  --wikitext2_path "$WIKITEXT2_PATH" \
  --prune_method paper_mi_gb_lcb \
  --prune_ratio 0.15 \
  --paper_mask_style structured_zero \
  --paper_score_nsamples 128 \
  --paper_calib_seqlen 2048 \
  --paper_scenario_ratios 0.5,0.75,1.0 \
  --paper_scenario_crops prefix,center,suffix \
  --paper_num_bins 16 \
  --paper_num_bands 4 \
  --paper_lcb_repeats 20 \
  --paper_budget_metrics params,flops,memory,kv_cache \
  --paper_band_coverage_ratio 0.90 \
  --paper_post_prune_validate \
  --eval_before \
  --eval_after \
  --paper_cache_dir "$OUTPUT/response_cache" \
  --paper_report_dir "$OUTPUT/report_before" \
  --paper_post_cache_dir "$OUTPUT/response_cache_after" \
  --paper_post_report_dir "$OUTPUT/report_after" \
  --output_dir "$OUTPUT/run"
