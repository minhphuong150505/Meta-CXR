#!/usr/bin/env bash
# Backward-compatible Q-Former-only entry point. The production Stage-2 runner
# owns train/validation/held-out-test evaluation and private-GCS upload.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STAGE2_IMAGE_MODE="${STAGE2_IMAGE_MODE:-qformer}"
exec "$SCRIPT_DIR/run_stage2.sh" "$@"
