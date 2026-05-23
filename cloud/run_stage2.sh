#!/usr/bin/env bash
# Stage 2: push evaluation notebook lên Kaggle, poll, pull output, upload GCS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/lib/common.sh"
require_kaggle_cli

NOTEBOOK_PATH="$SCRIPT_DIR/../$STAGE2_NOTEBOOK"
if [ ! -f "$NOTEBOOK_PATH" ]; then
  log "ERROR: $NOTEBOOK_PATH không tồn tại."
  exit 1
fi

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
WORK_DIR=$(mktemp -d)
OUTPUT_DIR="$SCRIPT_DIR/outputs/stage2/$RUN_ID"
mkdir -p "$OUTPUT_DIR"

log "=== Stage 2 run $RUN_ID ==="
log "Notebook: $NOTEBOOK_PATH"
log "Output dir: $OUTPUT_DIR"

cp "$NOTEBOOK_PATH" "$WORK_DIR/"
cp "$SCRIPT_DIR/kernels/eval/kernel-metadata.json" "$WORK_DIR/"

log "Pushing kernel..."
kaggle kernels push -p "$WORK_DIR"

if ! poll_kernel "$STAGE2_KERNEL_SLUG" "$MAX_POLL_HOURS"; then
  rc=$?
  log "Kernel polling kết thúc với rc=$rc — vẫn cố pull output để debug."
fi

log "Downloading kernel output..."
kaggle kernels output "$STAGE2_KERNEL_SLUG" -p "$OUTPUT_DIR" \
  || log "kernel output download failed."

GCS_DST="gs://$GCS_BUCKET/stage2/$RUN_ID/"
upload_gcs "$OUTPUT_DIR" "$GCS_DST"

MANIFEST=$(printf '{"run_id":"%s","kernel":"%s","gcs":"%s","timestamp":"%s"}' \
  "$RUN_ID" "$STAGE2_KERNEL_SLUG" "$GCS_DST" "$(date -u +%Y-%m-%dT%H:%M:%SZ)")
echo "$MANIFEST" | gsutil cp - "${GCS_DST}manifest.json"
log "Manifest: $MANIFEST"

log "=== Stage 2 run $RUN_ID complete ==="
log "Output: $GCS_DST"
log "Local copy: $OUTPUT_DIR"
