#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PRUNE_ORDER="${PRUNE_ORDER:-forward}"
BAND_GRADIENT="${BAND_GRADIENT:-on}"

case "${PRUNE_ORDER}" in
  forward|reverse|joint) ;;
  *) echo "[ERROR] PRUNE_ORDER must be forward, reverse, or joint" >&2; exit 2 ;;
esac
case "${BAND_GRADIENT}" in
  on|off) ;;
  *) echo "[ERROR] BAND_GRADIENT must be on or off" >&2; exit 2 ;;
esac

MODEL_PATH="${MODEL_PATH:-/root/dw2/Lya/models/Llama-2-7b}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/root/dw2/Lya/models}"
C4_PATH="${C4_PATH:-/root/dw2/Lya/dataset/dataset_c4}"
WIKITEXT2_PATH="${WIKITEXT2_PATH:-/root/dw2/Lya/dataset/dataset_wikitext-raw}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${PRUNE_ORDER}_band_${BAND_GRADIENT}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/v83_order/${EXPERIMENT_NAME}}"
# Share paper gradient-response cache by default. It is independent of mask order.
RESPONSE_CACHE_DIR="${RESPONSE_CACHE_DIR:-results/v83_order/shared_response_cache}"
REPORT_DIR="${REPORT_DIR:-${OUTPUT_DIR}/paper_report}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for p in "${MODEL_PATH}" "${C4_PATH}" "${WIKITEXT2_PATH}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Required path does not exist: ${p}" >&2
    exit 1
  fi
done
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] ${MODEL_PATH}/config.json not found; MODEL_PATH must be a local Transformers checkpoint." >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}/logs" "${REPORT_DIR}" "${RESPONSE_CACHE_DIR}"

BAND_FLAG="--paper_use_band_gradient"
if [[ "${BAND_GRADIENT}" == "off" ]]; then
  BAND_FLAG="--no-paper_use_band_gradient"
fi

echo "[V8.3-order] order=${PRUNE_ORDER} band_gradient=${BAND_GRADIENT}"
echo "[V8.3-order] output=${OUTPUT_DIR}"

python main.py \
  --model "${MODEL_PATH}" \
  --cache_dir "${MODEL_CACHE_DIR}" \
  --prune_method paper_full \
  --prune_order "${PRUNE_ORDER}" \
  "${BAND_FLAG}" \
  --c4_path "${C4_PATH}" \
  --wikitext2_path "${WIKITEXT2_PATH}" \
  --eval_wikitext_split validation \
  --sparsity_ratio 0.50 \
  --seqlen 4096 \
  --wanda_nsamples 128 \
  --wanda_calib_seqlen 4096 \
  --wanda_activation_storage auto \
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
  --paper_greedy_batches 8 \
  --paper_cache_dir "${RESPONSE_CACHE_DIR}" \
  --paper_report_dir "${REPORT_DIR}" \
  --save "${OUTPUT_DIR}/logs"
