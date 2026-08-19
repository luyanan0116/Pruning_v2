#!/usr/bin/env bash
set -euo pipefail

# Always run from the project root, even if this script is launched elsewhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# -----------------------------------------------------------------------------
# Local defaults for this server.
# You can still override any of them from the command line, e.g.
# MODEL_PATH=/other/model OUTPUT_DIR=results/test bash scripts/run_v8_paper_only_weight50.sh
# -----------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/root/dw2/Lya/models/Llama-2-7b}"
# This is only the Hugging Face cache directory. Because MODEL_PATH is a local
# model directory, it does NOT need to point to the model itself.
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/root/dw2/Lya/models}"  # compatibility only; model is loaded directly from MODEL_PATH
C4_PATH="${C4_PATH:-/root/dw2/Lya/dataset/dataset_c4}"
WIKITEXT2_PATH="${WIKITEXT2_PATH:-/root/dw2/Lya/dataset/dataset_wikitext-raw}"
OUTPUT_DIR="${OUTPUT_DIR:-results/v82_main}"
RESPONSE_CACHE_DIR="${RESPONSE_CACHE_DIR:-${OUTPUT_DIR}/response_cache}"

# Hard offline mode: never contact Hugging Face Hub.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Fail early on path mistakes instead of spending GPU time before discovering them.
for p in "${MODEL_PATH}" "${C4_PATH}" "${WIKITEXT2_PATH}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Required path does not exist: ${p}" >&2
    exit 1
  fi
done
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] ${MODEL_PATH}/config.json not found." >&2
  echo "        MODEL_PATH must be a local Hugging Face/Transformers-format Llama directory." >&2
  if [[ -f "${MODEL_PATH}/params.json" ]]; then
    echo "        This looks like an original Meta checkpoint; convert it to Transformers format first." >&2
  fi
  exit 1
fi
if ! compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null && \
   ! compgen -G "${MODEL_PATH}/pytorch_model*.bin" >/dev/null && \
   ! compgen -G "${MODEL_PATH}/model*.bin" >/dev/null; then
  echo "[ERROR] No local .safetensors/.bin model weights found under ${MODEL_PATH}." >&2
  exit 1
fi
mkdir -p "${MODEL_CACHE_DIR}" "${OUTPUT_DIR}" "${RESPONSE_CACHE_DIR}"

echo "[V8.2] MODEL_PATH=${MODEL_PATH}"
echo "[V8.2] MODEL_CACHE_DIR=${MODEL_CACHE_DIR}"
echo "[V8.2] C4_PATH=${C4_PATH}"
echo "[V8.2] WIKITEXT2_PATH=${WIKITEXT2_PATH}"
echo "[V8.2] OUTPUT_DIR=${OUTPUT_DIR}"

python run_paper_ablation.py \
  --model "${MODEL_PATH}" \
  --cache_dir "${MODEL_CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --response_cache_dir "${RESPONSE_CACHE_DIR}" \
  --c4_path "${C4_PATH}" \
  --wikitext2_path "${WIKITEXT2_PATH}" \
  --eval_wikitext_split validation \
  --sparsity_ratio 0.50 \
  --seqlen 4096 \
  --wanda_nsamples 128 \
  --wanda_calib_seqlen 4096 \
  --paper_prune_targets mlp,attention \
  --paper_weight_allocation paper_nonuniform \
  --paper_weight_min_unit_sparsity 0.45 \
  --paper_weight_max_unit_sparsity 0.55 \
  --paper_budget_temperature 1.0 \
  --paper_score_nsamples 128 \
  --paper_calib_seqlen 1024 \
  --paper_response_length 128 \
  --paper_scenario_ratios 0.5,0.75,1.0 \
  --paper_event_bins 3 \
  --paper_num_bins 16 \
  --paper_num_bands 4 \
  --paper_mi_neighbors 3 \
  --paper_purity_thresholds 0.65,0.75,0.85 \
  --paper_min_ball_size 8 \
  --paper_max_balls 64 \
  --paper_max_ball_depth 5 \
  --paper_gb_localization unit_local \
  --paper_gb_workers 16 \
  --paper_gb_chunk_size 64 \
  --paper_kde_scope probe \
  --paper_gb_fusion_mode inverse_sqrt_dispersion \
  --paper_gb_fusion_max_ratio 5.0 \
  --paper_lcb_repeats 20 \
  --paper_lcb_sample_fraction 0.80 \
  --paper_lcb_scenario_fraction 0.6666666666666666 \
  --lcb_lambda 0.5 \
  --paper_band_coverage_ratio 0.90 \
  --paper_coverage_alpha 0.10 \
  --paper_greedy_batches 8
