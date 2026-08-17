#!/usr/bin/env bash
set -euo pipefail
# v7 fast functional run: keeps real repeated LCB but uses 3 repeats by default.
export LCB_REPEATS=${LCB_REPEATS:-3}
export OUTPUT_DIR=${OUTPUT_DIR:-results/paper_v7_fast_s050}
exec bash "$(dirname "$0")/run_v7_clean_ablation.sh" "$@"
