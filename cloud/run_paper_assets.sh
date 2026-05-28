#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/META-CXR}"
RUN_NAME="${RUN_NAME:-07_all_three}"
GCS_BUCKET="${GCS_BUCKET:-gs://meta-cxr-checkpoint}"
GCS_OUT="${GCS_OUT:-$GCS_BUCKET/eval/paper_assets}"
LOG_DIR="$HOME/logs"
OUT_DIR="$PROJECT_DIR/paper_outputs"
MOUNT_CKPT="/mnt/meta-cxr-checkpoint/${RUN_NAME}/checkpoint_best.pth"
LOCAL_CKPT="$HOME/checkpoints/${RUN_NAME}/checkpoint_best.pth"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$(dirname "$LOCAL_CKPT")"
if [[ -f "$MOUNT_CKPT" ]]; then
  CKPT="$MOUNT_CKPT"
else
  CKPT="$LOCAL_CKPT"
  if [[ ! -f "$CKPT" ]]; then
    gcloud storage cp "$GCS_BUCKET/${RUN_NAME}/checkpoint_best.pth" "$CKPT" --quiet
  fi
fi

cd "$PROJECT_DIR"
python3 paper_assets.py \
  --run-name "$RUN_NAME" \
  --cfg-path "$PROJECT_DIR/pretraining/configs/encoder_comparison/${RUN_NAME}.yaml" \
  --checkpoint "$CKPT" \
  --split "${SPLIT:-test}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --figure-samples "${FIGURE_SAMPLES:-2}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-220}" \
  --outputs "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$LOG_DIR/paper_assets_${RUN_NAME}.log"

gcloud storage cp -r "$OUT_DIR/${RUN_NAME}_${SPLIT:-test}" "$GCS_OUT/" --quiet
gcloud storage cp "$LOG_DIR/paper_assets_${RUN_NAME}.log" "$GCS_OUT/${RUN_NAME}_${SPLIT:-test}/paper_assets_${RUN_NAME}.log" --quiet
echo "[uploaded] $GCS_OUT/${RUN_NAME}_${SPLIT:-test}/"
