#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/path/to/Llama-or-Mistral}"
C4_PATH="${C4_PATH:-/path/to/c4}"
WIKITEXT2_PATH="${WIKITEXT2_PATH:-/path/to/wikitext2}"

python main.py \
  --model "$MODEL" \
  --prune_method paper_mi_gb_lcb \
  --sparsity_ratio 0.15 \
  --paper_prune_targets mlp,attention \
  --paper_apply_mode shrink \
  --paper_budget_metric params \
  --paper_score_nsamples 32 \
  --paper_calib_seqlen 512 \
  --paper_response_length 32 \
  --paper_scenario_ratios 0.5,0.75,1.0 \
  --paper_num_bins 16 \
  --paper_num_bands 4 \
  --paper_gb_localization unit_local \
  --n_samples_lcb 20 \
  --paper_sample_fraction 0.8 \
  --paper_scenario_fraction 1.0 \
  --paper_band_coverage_ratio 0.90 \
  --paper_coverage_alpha 0.25 \
  --paper_cache_dir paper_response_cache_strict \
  --paper_report_dir paper_structured_report \
  --c4_path "$C4_PATH" \
  --wikitext2_path "$WIKITEXT2_PATH" \
  "$@"
