#!/usr/bin/env bash
set +e

TOTAL=6
FAST_PROFILE="${FAST_PROFILE:-balanced}"
LOG_DIR="logs/v83_fast_${FAST_PROFILE}"
mkdir -p "${LOG_DIR}"

ALL_START_TS=$(date +%s)
ALL_START_TIME=$(date "+%Y-%m-%d %H:%M:%S")

declare -a NAMES STATUS_LIST TIME_LIST EXIT_LIST

format_seconds() {
    local total_seconds=$1
    printf "%02d:%02d:%02d" $((total_seconds/3600)) $(((total_seconds%3600)/60)) $((total_seconds%60))
}

run_exp() {
    local INDEX=$1 NAME=$2 SCRIPT=$3
    local START_TS END_TS ELAPSED STATUS STATUS_TEXT
    START_TS=$(date +%s)
    echo ""
    echo "============================================================"
    echo "[$INDEX/$TOTAL] START ${NAME}"
    echo "Profile: ${FAST_PROFILE}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Log: ${LOG_DIR}/${NAME}.log"
    echo "============================================================"

    FAST_PROFILE="${FAST_PROFILE}" bash "${SCRIPT}" > "${LOG_DIR}/${NAME}.log" 2>&1
    STATUS=$?
    END_TS=$(date +%s)
    ELAPSED=$((END_TS-START_TS))

    if [[ ${STATUS} -eq 0 ]]; then STATUS_TEXT="SUCCESS"; else STATUS_TEXT="FAILED"; fi
    echo "[$INDEX/$TOTAL] ${STATUS_TEXT} ${NAME} | elapsed=$(format_seconds ${ELAPSED}) | exit=${STATUS}"

    NAMES[$INDEX]="${NAME}"
    STATUS_LIST[$INDEX]="${STATUS_TEXT}"
    TIME_LIST[$INDEX]="$(format_seconds ${ELAPSED})"
    EXIT_LIST[$INDEX]="${STATUS}"
    return 0
}

echo "============================================================"
echo "V8.3 FAST six-experiment batch"
echo "Profile: ${FAST_PROFILE}"
echo "Start: ${ALL_START_TIME}"
echo "============================================================"

# First run creates shared paper response + MI/GB/LCB score caches.
run_exp 1 forward_band_on  scripts/run_v83_forward_band_on.sh
# Reuses paper caches; forward activations remain independently recomputed because
# band on/off changes the pruning budget and therefore downstream activations.
run_exp 2 forward_band_off scripts/run_v83_forward_band_off.sh
# First frozen-stat run creates the shared dense Wanda stats cache.
run_exp 3 reverse_band_on  scripts/run_v83_reverse_band_on.sh
# The remaining reverse/joint variants reuse frozen dense Wanda statistics.
run_exp 4 reverse_band_off scripts/run_v83_reverse_band_off.sh
run_exp 5 joint_band_on    scripts/run_v83_joint_band_on.sh
run_exp 6 joint_band_off   scripts/run_v83_joint_band_off.sh

ALL_END_TS=$(date +%s)
ALL_END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
ALL_ELAPSED=$((ALL_END_TS-ALL_START_TS))
SUCCESS_COUNT=0
FAILED_COUNT=0
for i in $(seq 1 ${TOTAL}); do
    if [[ "${STATUS_LIST[$i]}" == "SUCCESS" ]]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT+1))
    else
        FAILED_COUNT=$((FAILED_COUNT+1))
    fi
done

echo ""
echo "================================================================================"
echo "V8.3 FAST SUMMARY (${FAST_PROFILE})"
echo "================================================================================"
printf "%-4s %-28s %-10s %-12s %-10s\n" "No." "Experiment" "Status" "Elapsed" "ExitCode"
echo "--------------------------------------------------------------------------------"
for i in $(seq 1 ${TOTAL}); do
    printf "%-4s %-28s %-10s %-12s %-10s\n" "$i" "${NAMES[$i]}" "${STATUS_LIST[$i]}" "${TIME_LIST[$i]}" "${EXIT_LIST[$i]}"
done
echo "--------------------------------------------------------------------------------"
echo "Success: ${SUCCESS_COUNT}/${TOTAL}"
echo "Failed : ${FAILED_COUNT}/${TOTAL}"
echo "Start  : ${ALL_START_TIME}"
echo "End    : ${ALL_END_TIME}"
echo "Total  : $(format_seconds ${ALL_ELAPSED})"
echo "================================================================================"

# Keep shell exit 0 so a single failed experiment does not make nohup supervision
# look like the batch script itself crashed. Failure details remain in the table.
exit 0
