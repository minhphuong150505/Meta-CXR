#!/usr/bin/env bash
# Backward-compatible name for the private GCP Stage-2 pipeline.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_stage2.sh" "$@"
