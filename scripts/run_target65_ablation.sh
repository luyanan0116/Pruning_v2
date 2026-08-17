#!/usr/bin/env bash
set -euo pipefail
# v7 full four-stage ablation.  This script does not promise a fixed PPL;
# it preserves the historical entry point while using the corrected pipeline.
export LCB_REPEATS=${LCB_REPEATS:-10}
export OUTPUT_DIR=${OUTPUT_DIR:-results/paper_v7_target_s050}
exec bash "$(dirname "$0")/run_v7_clean_ablation.sh" "$@"
