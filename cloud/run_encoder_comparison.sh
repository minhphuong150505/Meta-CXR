#!/usr/bin/env bash
# Backward-compatible entry point. The production run trains the all-encoder
# full-data model once; encoder comparisons are evaluation-time ablations.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_stage1.sh"
