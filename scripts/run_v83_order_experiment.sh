#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PRUNE_ORDER="${PRUNE_ORDER:-forward}"
BAND_GRADIENT="${BAND_GRADIENT:-on}"
FAST_PROFILE="${FAST_PROFILE:-balanced}"

case "${PRUNE_ORDER}" in
  forward|reverse|joint) ;;
  *) echo "[ERROR] PRUNE_ORDER must be forward, reverse, or joint" >&2; exit 2 ;;
esac
case "${BAND_GRADIENT}" in
  on|off) ;;
  *) echo "[ERROR] BAND_GRADIENT must be on or off" >&2; exit 2 ;;
esac
case "${FAST_PROFILE}" in
  balanced|full_cache) ;;
  *) echo "[ERROR] FAST_PROFILE must be balanced or full_cache" >&2; exit 2 ;;
esac

MODEL_PATH="${MODEL_PATH:-/root/dw2/Lya/models/Llama-2-7b}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/root/dw2/Lya/models}"
C4_PATH="${C4_PATH:-/root/dw2/Lya/dataset/dataset_c4}"
WIKITEXT2_PATH="${WIKITEXT2_PATH:-/root/dw2/Lya/dataset/dataset_wikitext-raw}"

# ---------------------------------------------------------------------------
# Profiles
# balanced: substantially faster calibration for six-way exploratory comparison.
# full_cache: original V8.3 calibration sizes, but shares expensive score/stats
#             caches and disables diagnostic-only KDE/plots.
# Every value can still be overridden by environment variables.
# ---------------------------------------------------------------------------
if [[ "${FAST_PROFILE}" == "balanced" ]]; then
  DEFAULT_EVAL_SEQLEN=2048
  DEFAULT_WANDA_NSAMPLES=32
  DEFAULT_WANDA_CALIB_SEQLEN=1024
  DEFAULT_PAPER_SCORE_NSAMPLES=32
  DEFAULT_PAPER_CALIB_SEQLEN=512
  DEFAULT_PAPER_RESPONSE_LENGTH=64
  DEFAULT_PAPER_SCENARIO_RATIOS="0.5,1.0"
  DEFAULT_PAPER_LCB_REPEATS=8
  DEFAULT_PAPER_MAX_BALLS=32
  DEFAULT_PAPER_MAX_BALL_DEPTH=4
  DEFAULT_PAPER_PROBE_UNITS=32
  DEFAULT_PAPER_GREEDY_BATCHES=4
else
  DEFAULT_EVAL_SEQLEN=4096
  DEFAULT_WANDA_NSAMPLES=128
  DEFAULT_WANDA_CALIB_SEQLEN=4096
  DEFAULT_PAPER_SCORE_NSAMPLES=128
  DEFAULT_PAPER_CALIB_SEQLEN=1024
  DEFAULT_PAPER_RESPONSE_LENGTH=128
  DEFAULT_PAPER_SCENARIO_RATIOS="0.5,0.75,1.0"
  DEFAULT_PAPER_LCB_REPEATS=20
  DEFAULT_PAPER_MAX_BALLS=64
  DEFAULT_PAPER_MAX_BALL_DEPTH=5
  DEFAULT_PAPER_PROBE_UNITS=64
  DEFAULT_PAPER_GREEDY_BATCHES=8
fi

EVAL_SEQLEN="${EVAL_SEQLEN:-${DEFAULT_EVAL_SEQLEN}}"
WANDA_NSAMPLES="${WANDA_NSAMPLES:-${DEFAULT_WANDA_NSAMPLES}}"
WANDA_CALIB_SEQLEN="${WANDA_CALIB_SEQLEN:-${DEFAULT_WANDA_CALIB_SEQLEN}}"
PAPER_SCORE_NSAMPLES="${PAPER_SCORE_NSAMPLES:-${DEFAULT_PAPER_SCORE_NSAMPLES}}"
PAPER_CALIB_SEQLEN="${PAPER_CALIB_SEQLEN:-${DEFAULT_PAPER_CALIB_SEQLEN}}"
PAPER_RESPONSE_LENGTH="${PAPER_RESPONSE_LENGTH:-${DEFAULT_PAPER_RESPONSE_LENGTH}}"
PAPER_SCENARIO_RATIOS="${PAPER_SCENARIO_RATIOS:-${DEFAULT_PAPER_SCENARIO_RATIOS}}"
PAPER_LCB_REPEATS="${PAPER_LCB_REPEATS:-${DEFAULT_PAPER_LCB_REPEATS}}"
PAPER_MAX_BALLS="${PAPER_MAX_BALLS:-${DEFAULT_PAPER_MAX_BALLS}}"
PAPER_MAX_BALL_DEPTH="${PAPER_MAX_BALL_DEPTH:-${DEFAULT_PAPER_MAX_BALL_DEPTH}}"
PAPER_PROBE_UNITS="${PAPER_PROBE_UNITS:-${DEFAULT_PAPER_PROBE_UNITS}}"
PAPER_GREEDY_BATCHES="${PAPER_GREEDY_BATCHES:-${DEFAULT_PAPER_GREEDY_BATCHES}}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${PRUNE_ORDER}_band_${BAND_GRADIENT}}"
BASE_RESULT_DIR="${BASE_RESULT_DIR:-results/v83_fast/${FAST_PROFILE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_DIR}/${EXPERIMENT_NAME}}"

# These are intentionally shared across all six variants in the same profile.
RESPONSE_CACHE_DIR="${RESPONSE_CACHE_DIR:-${BASE_RESULT_DIR}/shared_response_cache}"
SCORE_CACHE_DIR="${SCORE_CACHE_DIR:-${BASE_RESULT_DIR}/shared_score_cache}"
WANDA_STATS_CACHE_DIR="${WANDA_STATS_CACHE_DIR:-${BASE_RESULT_DIR}/shared_dense_wanda_stats}"
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
mkdir -p "${OUTPUT_DIR}/logs" "${REPORT_DIR}" "${RESPONSE_CACHE_DIR}" "${SCORE_CACHE_DIR}" "${WANDA_STATS_CACHE_DIR}"

BAND_FLAG="--paper_use_band_gradient"
if [[ "${BAND_GRADIENT}" == "off" ]]; then
  BAND_FLAG="--no-paper_use_band_gradient"
fi

echo "[V8.3-fast] profile=${FAST_PROFILE} order=${PRUNE_ORDER} band_gradient=${BAND_GRADIENT}"
echo "[V8.3-fast] output=${OUTPUT_DIR}"
echo "[V8.3-fast] Wanda: nsamples=${WANDA_NSAMPLES}, calib_seqlen=${WANDA_CALIB_SEQLEN}"
echo "[V8.3-fast] Paper: nsamples=${PAPER_SCORE_NSAMPLES}, calib_seqlen=${PAPER_CALIB_SEQLEN}, response=${PAPER_RESPONSE_LENGTH}, scenarios=${PAPER_SCENARIO_RATIOS}, LCB=${PAPER_LCB_REPEATS}"
echo "[V8.3-fast] shared response cache=${RESPONSE_CACHE_DIR}"
echo "[V8.3-fast] shared score cache=${SCORE_CACHE_DIR}"
echo "[V8.3-fast] shared dense Wanda stats=${WANDA_STATS_CACHE_DIR} (reverse/joint only)"

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
  --seqlen "${EVAL_SEQLEN}" \
  --wanda_nsamples "${WANDA_NSAMPLES}" \
  --wanda_calib_seqlen "${WANDA_CALIB_SEQLEN}" \
  --wanda_activation_storage auto \
  --wanda_stats_cache_dir "${WANDA_STATS_CACHE_DIR}" \
  --paper_prune_targets mlp,attention \
  --paper_weight_allocation paper_nonuniform \
  --paper_weight_min_unit_sparsity 0.45 \
  --paper_weight_max_unit_sparsity 0.55 \
  --paper_budget_temperature 1.0 \
  --paper_score_nsamples "${PAPER_SCORE_NSAMPLES}" \
  --paper_calib_seqlen "${PAPER_CALIB_SEQLEN}" \
  --paper_response_length "${PAPER_RESPONSE_LENGTH}" \
  --paper_scenario_ratios "${PAPER_SCENARIO_RATIOS}" \
  --paper_event_bins 3 \
  --paper_num_bins 16 \
  --paper_num_bands 4 \
  --paper_mi_neighbors 3 \
  --paper_probe_units "${PAPER_PROBE_UNITS}" \
  --paper_purity_thresholds 0.65,0.75,0.85 \
  --paper_min_ball_size 8 \
  --paper_max_balls "${PAPER_MAX_BALLS}" \
  --paper_max_ball_depth "${PAPER_MAX_BALL_DEPTH}" \
  --paper_gb_localization unit_local \
  --paper_gb_workers 16 \
  --paper_gb_chunk_size 64 \
  --paper_kde_scope none \
  --paper_gb_fusion_mode inverse_sqrt_dispersion \
  --paper_gb_fusion_max_ratio 5.0 \
  --paper_lcb_repeats "${PAPER_LCB_REPEATS}" \
  --paper_lcb_sample_fraction 0.80 \
  --paper_lcb_scenario_fraction 0.6666666666666666 \
  --lcb_lambda 0.5 \
  --paper_band_coverage_ratio 0.90 \
  --paper_coverage_alpha 0.10 \
  --paper_greedy_batches "${PAPER_GREEDY_BATCHES}" \
  --paper_cache_dir "${RESPONSE_CACHE_DIR}" \
  --paper_score_cache_dir "${SCORE_CACHE_DIR}" \
  --paper_report_dir "${REPORT_DIR}" \
  --paper_plot_layers "" \
  --save "${OUTPUT_DIR}/logs" \
  "$@"
