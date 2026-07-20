#!/usr/bin/env bash
# Generate paper assets on GCP and store them only in private GCS. Assets can
# contain image identifiers or generated clinical text and are not publishable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/lib/common.sh"
require_gcp_config
require_private_bucket "$GCS_BUCKET"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="${RUN_NAME:-$STAGE1_RUN}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$HOME/meta-cxr-checkpoints}"
CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_ROOT/$RUN_NAME/checkpoint_best.pth}"
OUT_DIR="${PAPER_OUTPUT_DIR:-$HOME/meta-cxr-output/paper-assets}"
GCS_DST="gs://$GCS_BUCKET/paper-assets/$RUN_NAME"

test -f "$CHECKPOINT"
mkdir -p "$OUT_DIR"
cd "$REPO_DIR"
"$PYTHON_BIN" paper_assets.py \
  --run-name "$RUN_NAME" \
  --cfg-path "$STAGE1_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split "${SPLIT:-test}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --figure-samples "${FIGURE_SAMPLES:-2}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-220}" \
  --outputs "$OUT_DIR" \
  "$@"

upload_gcs "$OUT_DIR" "$GCS_DST"
log "Paper assets uploaded to private destination: $GCS_DST"
