#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:?Set MODEL}
OUTPUT=${OUTPUT:-outputs/validation}

python run_paper_validation.py \
  --model "$MODEL" \
  --output_dir "$OUTPUT" \
  --prune_ratios 0.05,0.10,0.15,0.20,0.25 \
  --seqlens 512,1024,2048 \
  --seeds 0,1,2 \
  --relative_ppl_limit 0.05
