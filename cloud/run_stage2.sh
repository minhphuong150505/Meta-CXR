#!/usr/bin/env bash
# Fine-tune/evaluate the Q-Former soft-token model and native MedGemma-image
# baseline on one GCP GPU. Uploads only to the configured private GCS bucket.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/lib/common.sh"
require_gcp_config
require_private_bucket "$GCS_BUCKET"
require_private_bucket "$GCS_DATA_BUCKET"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$HOME/meta-cxr-checkpoints}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-$HOME/meta-cxr-output/stage2/$RUN_ID}"
GCS_DST="gs://$GCS_BUCKET/stage2/$STAGE1_RUN/$RUN_ID"

case "$STAGE2_IMAGE_MODE" in
  qformer|native|both) ;;
  *) log "ERROR: STAGE2_IMAGE_MODE must be qformer, native, or both."; exit 1 ;;
esac

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"
log "Stage-2 image mode: $STAGE2_IMAGE_MODE"
log "Local output: $OUTPUT_DIR"
"$PYTHON_BIN" training/run_medgemma_qlora.py \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --stage1-run "$STAGE1_RUN" \
  --stage1-config "$STAGE1_CONFIG" \
  --image-mode "$STAGE2_IMAGE_MODE" \
  --output-dir "$OUTPUT_DIR" \
  --gcs-output "$GCS_DST" \
  "$@"

log "Stage-2 complete: $GCS_DST"
