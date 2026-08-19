#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/dw2/Lya/models/Llama-2-7b}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/root/dw2/Lya/models/cache}"
C4_PATH="${C4_PATH:-/root/dw2/Lya/dataset/dataset_c4}"
WIKITEXT2_PATH="${WIKITEXT2_PATH:-/root/dw2/Lya/dataset/dataset_wikitext-raw}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper_v8_uniform_weight50}"
RESPONSE_CACHE_DIR="${RESPONSE_CACHE_DIR:-results/paper_v7_clean_s050/response_cache}"

python run_paper_ablation.py \
  --model "${MODEL_PATH}" --cache_dir "${MODEL_CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" --response_cache_dir "${RESPONSE_CACHE_DIR}" \
  --c4_path "${C4_PATH}" --wikitext2_path "${WIKITEXT2_PATH}" \
  --sparsity_ratio 0.50 --paper_weight_allocation uniform
