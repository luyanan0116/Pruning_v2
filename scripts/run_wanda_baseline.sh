#!/usr/bin/env bash
set -euo pipefail

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u main.py \
  --model /root/dw2/Lya/models/Llama-2-7b \
  --c4_path /root/dw2/Lya/dataset/dataset_c4 \
  --wikitext2_path /root/dw2/Lya/dataset/dataset_wikitext-raw \
  --cache_dir /root/dw2/Lya/models/cache \
  --prune_method wanda --sparsity_ratio 0.50 \
  --sparsity_type unstructured --nsamples 128 --seqlen 2048 \
  --save results/wanda_baseline_s050
