#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/root/dw2/Lya/models/Llama-2-7b}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/root/dw2/Lya/models}"
C4_PATH="${C4_PATH:-/root/dw2/Lya/dataset/dataset_c4}"
WIKITEXT2_PATH="${WIKITEXT2_PATH:-/root/dw2/Lya/dataset/dataset_wikitext-raw}"
OUTPUT_DIR="${OUTPUT_DIR:-results/v82_baseline}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${OUTPUT_DIR}"

python main.py --model "${MODEL_PATH}" --cache_dir "${MODEL_CACHE_DIR}" \
  --prune_method dense --seqlen 4096 \
  --wikitext2_path "${WIKITEXT2_PATH}" --eval_wikitext_split both --save "${OUTPUT_DIR}"

python main.py --model "${MODEL_PATH}" --cache_dir "${MODEL_CACHE_DIR}" \
  --prune_method wanda --sparsity_ratio 0.50 --seqlen 4096 \
  --wanda_nsamples 128 --wanda_calib_seqlen 4096 \
  --c4_path "${C4_PATH}" --wikitext2_path "${WIKITEXT2_PATH}" \
  --eval_wikitext_split both --save "${OUTPUT_DIR}"
