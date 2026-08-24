#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRUNE_ORDER="reverse" BAND_GRADIENT="on"   bash "${SCRIPT_DIR}/run_v83_order_experiment.sh" "$@"
