#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
CACHE_DIR="${CACHE_DIR:-paper_response_cache_exp}"
RESULT_ROOT="${RESULT_ROOT:-paper_compare_results/exp3_keep_strategy}"
mkdir -p "$RESULT_ROOT"
BASE=(
  --model "$MODEL_PATH" --prune_method paper_mi_gb_lcb --sparsity_ratio 0.5
  --paper_prune_targets mlp,attention --paper_mask_style structured_unit
  --paper_num_bands 3 --paper_band_coverage_ratios 0.85,0.70,0.55
  --paper_band_selection_weights 1.0,0.7,0.4 --paper_cache_dir "$CACHE_DIR"
)

for STRATEGY in lcb_only band_only paper_hybrid; do
  python main.py "${BASE[@]}" \
    --paper_final_keep_strategy "$STRATEGY" \
    --paper_report_dir "$RESULT_ROOT/${STRATEGY}_scores" \
    --save "$RESULT_ROOT/$STRATEGY" 2>&1 | tee "$RESULT_ROOT/$STRATEGY.log"
done

printf "strategy\tppl_line\n" > "$RESULT_ROOT/summary.tsv"
for STRATEGY in lcb_only band_only paper_hybrid; do
  printf "%s\t%s\n" "$STRATEGY" "$(grep -E 'wikitext perplexity|ppl_test' "$RESULT_ROOT/$STRATEGY.log" | tail -1)" >> "$RESULT_ROOT/summary.tsv"
done
cat "$RESULT_ROOT/summary.tsv"
