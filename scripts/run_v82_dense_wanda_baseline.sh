#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-2-7b-hf}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-llm_weights}"
C4_PATH="${C4_PATH:-/root/dw2/Lya/dataset/dataset_c4}"
WIKITEXT2_PATH="${WIKITEXT2_PATH:-/root/dw2/Lya/dataset/dataset_wikitext-raw}"
OUTPUT_DIR="${OUTPUT_DIR:-results/v82_baseline}"
mkdir -p "${OUTPUT_DIR}"

python main.py --model "${MODEL_PATH}" --cache_dir "${MODEL_CACHE_DIR}" \
  --prune_method dense --seqlen 4096 \
  --wikitext2_path "${WIKITEXT2_PATH}" --eval_wikitext_split both --save "${OUTPUT_DIR}"

python main.py --model "${MODEL_PATH}" --cache_dir "${MODEL_CACHE_DIR}" \
  --prune_method wanda --sparsity_ratio 0.50 --seqlen 4096 \
  --wanda_nsamples 128 --wanda_calib_seqlen 4096 \
  --c4_path "${C4_PATH}" --wikitext2_path "${WIKITEXT2_PATH}" \
  --eval_wikitext_split both --save "${OUTPUT_DIR}"
