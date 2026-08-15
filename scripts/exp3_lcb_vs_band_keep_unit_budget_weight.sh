#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="/root/dw2/Lya/models/Llama-2-7b"
C4_PATH="/root/dw2/Lya/dataset/dataset_c4"
WIKITEXT2_PATH="/root/dw2/Lya/dataset/dataset_wikitext-raw"
CACHE_DIR="${CACHE_DIR:-paper_response_cache_exp3_unit_budget}"
RESULT_ROOT="${RESULT_ROOT:-paper_compare_results/exp3_keep_strategy_unit_budget}"
mkdir -p "$RESULT_ROOT"

BASE=(
  --model "$MODEL_PATH"
  --c4_path "$C4_PATH"
  --wikitext2_path "$WIKITEXT2_PATH"
  --paper_calib_dataset c4
  --prune_method paper_mi_gb_lcb
  --sparsity_ratio 0.5
  --paper_prune_targets mlp,attention
  --paper_mask_style unit_budget_weight
  --paper_unit_min_sparsity 0.30
  --paper_unit_max_sparsity 0.70
  --paper_num_bands 3
  --paper_band_coverage_ratios 0.85,0.70,0.55
  --paper_band_selection_weights 1.0,0.7,0.4
  --paper_cache_dir "$CACHE_DIR"
  --seqlen 2048
)

STRATEGIES=(lcb_only band_only paper_hybrid)
for i in "${!STRATEGIES[@]}"; do
  S="${STRATEGIES[$i]}"
  EXTRA=(--paper_overwrite_scores)
  if [[ "$i" -eq 0 ]]; then EXTRA+=(--paper_overwrite_cache); fi
  python main.py "${BASE[@]}"     --paper_final_keep_strategy "$S"     --paper_report_dir "$RESULT_ROOT/${S}_scores"     "${EXTRA[@]}"     --save "$RESULT_ROOT/$S" 2>&1 | tee "$RESULT_ROOT/$S.log"
done

printf "strategy\tppl_line\n" > "$RESULT_ROOT/summary.tsv"
for S in "${STRATEGIES[@]}"; do
  printf "%s\t%s\n" "$S" "$(grep -E 'wikitext perplexity|ppl_test' "$RESULT_ROOT/$S.log" | tail -1)" >> "$RESULT_ROOT/summary.tsv"
done
cat "$RESULT_ROOT/summary.tsv"
