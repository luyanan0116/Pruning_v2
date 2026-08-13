#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
CACHE_DIR="${CACHE_DIR:-paper_response_cache_exp}"
RESULT_ROOT="${RESULT_ROOT:-paper_compare_results/exp1_granularity}"
mkdir -p "$RESULT_ROOT"
COMMON=(
  --model "$MODEL_PATH"
  --prune_method paper_mi_gb_lcb
  --sparsity_ratio 0.5
  --paper_prune_targets mlp,attention
  --paper_num_bands 3
  --paper_band_coverage_ratios 0.85,0.70,0.55
  --paper_band_selection_weights 1.0,0.7,0.4
  --paper_final_keep_strategy paper_hybrid
  --paper_cache_dir "$CACHE_DIR"
)

# Legacy v6 baseline: 50% weight elements (uses the existing sequential Wanda path).
python main.py "${COMMON[@]}" \
  --paper_mask_style wanda_weight \
  --paper_report_dir "$RESULT_ROOT/shared_scores" \
  --save "$RESULT_ROOT/weight50" 2>&1 | tee "$RESULT_ROOT/weight50.log"

# Paper-aligned: 50% complete attention heads and FFN channels.
python main.py "${COMMON[@]}" \
  --paper_mask_style structured_unit \
  --paper_report_dir "$RESULT_ROOT/shared_scores" \
  --save "$RESULT_ROOT/unit50" 2>&1 | tee "$RESULT_ROOT/unit50.log"

printf "case\tppl_line\n" > "$RESULT_ROOT/summary.tsv"
printf "weight50\t%s\n" "$(grep -E 'wikitext perplexity|ppl_test' "$RESULT_ROOT/weight50.log" | tail -1)" >> "$RESULT_ROOT/summary.tsv"
printf "unit50\t%s\n" "$(grep -E 'wikitext perplexity|ppl_test' "$RESULT_ROOT/unit50.log" | tail -1)" >> "$RESULT_ROOT/summary.tsv"
cat "$RESULT_ROOT/summary.tsv"
