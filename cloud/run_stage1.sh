#!/usr/bin/env bash
# Stage 1: push training notebook lên Kaggle, poll, pull output, upload GCS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/lib/common.sh"
require_kaggle_cli

NOTEBOOK_PATH="$SCRIPT_DIR/../$STAGE1_NOTEBOOK"
if [ ! -f "$NOTEBOOK_PATH" ]; then
  log "ERROR: $NOTEBOOK_PATH không tồn tại."
  exit 1
fi

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
WORK_DIR=$(mktemp -d)
OUTPUT_DIR="$SCRIPT_DIR/outputs/stage1/$RUN_ID"
mkdir -p "$OUTPUT_DIR"

log "=== Stage 1 run $RUN_ID ==="
log "Notebook: $NOTEBOOK_PATH"
log "Work dir: $WORK_DIR"
log "Output dir: $OUTPUT_DIR"

# 1. Stage notebook + metadata
cp "$NOTEBOOK_PATH" "$WORK_DIR/"
cp "$SCRIPT_DIR/kernels/train/kernel-metadata.json" "$WORK_DIR/"

# 2. Push lên Kaggle
log "Pushing kernel..."
kaggle kernels push -p "$WORK_DIR"

# 3. Poll
if ! poll_kernel "$STAGE1_KERNEL_SLUG" "$MAX_POLL_HOURS"; then
  rc=$?
  log "Kernel polling kết thúc với rc=$rc — vẫn cố pull output để debug."
fi

# 4. Pull kernel output
log "Downloading kernel output..."
kaggle kernels output "$STAGE1_KERNEL_SLUG" -p "$OUTPUT_DIR" \
  || log "kernel output download failed (có thể kernel chưa có file output)."

# 5. Checkpoints are now uploaded directly from the Kaggle notebook to GCS.
log "Skipping Kaggle checkpoint dataset download; notebook uploads checkpoints to gs://$GCS_BUCKET/<run_name>/."

# 6. Upload kernel output lên GCS
GCS_DST="gs://$GCS_BUCKET/stage1/$RUN_ID/"
upload_gcs "$OUTPUT_DIR" "$GCS_DST"

# 7. Manifest
MANIFEST=$(printf '{"run_id":"%s","kernel":"%s","gcs":"%s","timestamp":"%s"}' \
  "$RUN_ID" "$STAGE1_KERNEL_SLUG" "$GCS_DST" "$(date -u +%Y-%m-%dT%H:%M:%SZ)")
echo "$MANIFEST" | gsutil cp - "${GCS_DST}manifest.json"
log "Manifest: $MANIFEST"

log "=== Stage 1 run $RUN_ID complete ==="
log "Output: $GCS_DST"
log "Local copy: $OUTPUT_DIR"
