#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/META-CXR}"
RUN_NAME="${RUN_NAME:-07_all_three}"
GCS_BUCKET="${GCS_BUCKET:-gs://meta-cxr-checkpoint}"
GCS_EVAL_DIR="${GCS_EVAL_DIR:-$GCS_BUCKET/eval/table_7_loss_ablation}"

CFG_PATH="$PROJECT_DIR/pretraining/configs/encoder_comparison/${RUN_NAME}.yaml"
MOUNT_CKPT="/mnt/meta-cxr-checkpoint/${RUN_NAME}/checkpoint_best.pth"
LOCAL_CKPT="$HOME/checkpoints/${RUN_NAME}/checkpoint_best.pth"
LOG_DIR="$HOME/logs"
OUT_DIR="$PROJECT_DIR/tables/table_7_loss_ablation"
OUT_JSON="$OUT_DIR/${RUN_NAME}_mean_f1.json"
LOG_FILE="$LOG_DIR/table_7_loss_f1_${RUN_NAME}.log"

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
python3 eval_table7_loss_f1.py \
  --run-name "$RUN_NAME" \
  --cfg-path "$CFG_PATH" \
  --checkpoint "$CKPT" \
  --split test \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --f1-mode "${F1_MODE:-weighted_multiclass}" \
  --out-json "$OUT_JSON" \
  2>&1 | tee "$LOG_FILE"

gcloud storage cp "$OUT_JSON" "$GCS_EVAL_DIR/${RUN_NAME}_mean_f1.json" --quiet
gcloud storage cp "$LOG_FILE" "$GCS_EVAL_DIR/table_7_loss_f1_${RUN_NAME}.log" --quiet

echo "[uploaded] $GCS_EVAL_DIR/${RUN_NAME}_mean_f1.json"
