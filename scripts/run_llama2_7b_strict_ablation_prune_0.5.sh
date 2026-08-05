#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Llama-2-7B 三阶段严格 PPL 消融
#
#   1. paper_mi         : 互信息
#   2. paper_mi_gb      : 互信息 + 粒球
#   3. paper_mi_gb_lcb  : 互信息 + 粒球 + 双源重复估计 + LCB
#
# 将本脚本放到 Pruning_paper_exact_v9_strict_ablation
# 项目根目录后运行。
# ============================================================

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="/root/dw2/Lya/models/Llama-2-7b"
C4_PATH="/root/dw2/Lya/dataset/dataset_c4"
WIKITEXT2_PATH="/root/dw2/Lya/dataset/dataset_wikitext-raw"

# 可通过环境变量覆盖：
# CUDA_VISIBLE_DEVICES=1 PRUNE_RATIO=0.5 SEED=1 bash run_llama2_7b_strict_ablation_prune_0.5.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRUNE_RATIO="${PRUNE_RATIO:-0.5}"
SEED="${SEED:-0}"
SEQLEN="${SEQLEN:-2048}"
SCORE_NSAMPLES="${SCORE_NSAMPLES:-128}"
LCB_REPEATS="${LCB_REPEATS:-20}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.8}"
SCENARIO_FRACTION="${SCENARIO_FRACTION:-1.0}"
MASK_STYLE="${MASK_STYLE:-structured_zero}"

RUN_TAG="llama2_7b_prune_${PRUNE_RATIO}_seed_${SEED}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${RUN_TAG}}"
LOG_DIR="${OUTPUT_DIR}/launcher_logs"
mkdir -p "${LOG_DIR}"

# OVERWRITE=1 时重新计算贡献分数和响应缓存。
OVERWRITE="${OVERWRITE:-0}"

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ -f "${PROJECT_DIR}/run_paper_ablation.py" ]] \
    || die "未找到 ${PROJECT_DIR}/run_paper_ablation.py。请把脚本放到项目根目录。"

[[ -d "${MODEL_PATH}" ]] \
    || die "模型目录不存在：${MODEL_PATH}"

[[ -e "${C4_PATH}" ]] \
    || die "C4 数据路径不存在：${C4_PATH}"

[[ -e "${WIKITEXT2_PATH}" ]] \
    || die "WikiText2 数据路径不存在：${WIKITEXT2_PATH}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
    || die "找不到 Python：${PYTHON_BIN}"

export C4_PATH
export WIKITEXT2_PATH
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# 减少显存碎片；旧版 PyTorch 不识别时可删除这一行。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/run_paper_ablation.py"
    --model "${MODEL_PATH}"
    --c4_path "${C4_PATH}"
    --wikitext2_path "${WIKITEXT2_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --prune_ratio "${PRUNE_RATIO}"
    --seed "${SEED}"
    --mask_style "${MASK_STYLE}"

    # 严格消融必须使用 equal：
    # MI+粒球阶段不提前调用重复估计，第三阶段才新增 LCB。
    --paper_granularity_weight_mode equal

    --paper_score_nsamples "${SCORE_NSAMPLES}"
    --paper_calib_seqlen "${SEQLEN}"
    --paper_lcb_repeats "${LCB_REPEATS}"
    --paper_sample_fraction "${SAMPLE_FRACTION}"
    --paper_scenario_fraction "${SCENARIO_FRACTION}"
    --seqlen "${SEQLEN}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
    CMD+=(--overwrite)
fi

echo "============================================================"
echo "三阶段严格消融启动"
echo "PROJECT_DIR        : ${PROJECT_DIR}"
echo "MODEL_PATH         : ${MODEL_PATH}"
echo "C4_PATH            : ${C4_PATH}"
echo "WIKITEXT2_PATH     : ${WIKITEXT2_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "PRUNE_RATIO        : ${PRUNE_RATIO}"
echo "SEED               : ${SEED}"
echo "SEQLEN             : ${SEQLEN}"
echo "SCORE_NSAMPLES     : ${SCORE_NSAMPLES}"
echo "LCB_REPEATS        : ${LCB_REPEATS}"
echo "MASK_STYLE         : ${MASK_STYLE}"
echo "OUTPUT_DIR         : ${OUTPUT_DIR}"
echo "OVERWRITE          : ${OVERWRITE}"
echo "============================================================"

printf '执行命令：'
printf ' %q' "${CMD[@]}"
printf '\n\n'

START_TIME="$(date '+%F %T')"
echo "开始时间：${START_TIME}"

"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/strict_ablation_console.log"

echo
echo "============================================================"
echo "实验完成"
echo "结束时间：$(date '+%F %T')"
echo "结果汇总："
echo "  ${OUTPUT_DIR}/ablation_summary.csv"
echo "  ${OUTPUT_DIR}/ablation_summary.json"
echo "完整日志："
echo "  ${LOG_DIR}/strict_ablation_console.log"
echo "============================================================"
