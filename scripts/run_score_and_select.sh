#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:?Set MODEL}
OUTPUT=${OUTPUT:-outputs/score_select}
RATIO=${RATIO:-0.15}

python main.py \
  --model "$MODEL" \
  --prune_ratio 0 \
  --paper_score_only \
  --paper_cache_dir "$OUTPUT/cache" \
  --paper_report_dir "$OUTPUT/scores" \
  --output_dir "$OUTPUT/score_run" \
  --no-eval_after

python generate_global_config.py \
  --report_dir "$OUTPUT/scores" \
  --model "$MODEL" \
  --method paper_mi_gb_lcb \
  --prune_ratio "$RATIO" \
  --output "$OUTPUT/selection.json"

python main.py \
  --model "$MODEL" \
  --prune_method paper_mi_gb_lcb \
  --prune_ratio "$RATIO" \
  --paper_selection_file "$OUTPUT/selection.json" \
  --paper_mask_style structured_zero \
  --paper_report_dir "$OUTPUT/application" \
  --eval_before \
  --eval_after \
  --no-paper_post_prune_validate \
  --output_dir "$OUTPUT/evaluation"
