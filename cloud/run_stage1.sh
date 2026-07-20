#!/usr/bin/env bash
# Train Stage-1 directly on one GCP GPU and upload only model/log artifacts to
# the configured private GCS bucket.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/lib/common.sh"
require_gcp_config
require_private_bucket "$GCS_BUCKET"
require_private_bucket "$GCS_DATA_BUCKET"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_BASE="${OUTPUT_BASE:-$HOME/meta-cxr-output/stage1/$RUN_ID}"
RUN_DIR="$OUTPUT_BASE/$STAGE1_RUN"
GCS_DST="gs://$GCS_BUCKET/stage1/$STAGE1_RUN/$RUN_ID"
mkdir -p "$OUTPUT_BASE"

cd "$REPO_DIR"
log "Stage-1 config: $STAGE1_CONFIG"
log "Local output: $RUN_DIR"
"$PYTHON_BIN" -m pretraining.train \
  --cfg-path "$STAGE1_CONFIG" \
  --options run.output_dir="$OUTPUT_BASE" run.run_name="$STAGE1_RUN"

test -f "$RUN_DIR/checkpoint_best.pth"
upload_gcs "$RUN_DIR" "$GCS_DST"
log "Stage-1 complete: $GCS_DST"
