#!/usr/bin/env bash
# Trains a single model with all three encoders (biovil + pubmedclip + swin).
# Per-encoder comparison is done by toggling encoders at eval on this one
# checkpoint (see notebooks/07_table5_f1_encoder_toggle_single_ckpt.ipynb),
# not by retraining per combination.
# Uploads checkpoints + final output_dir to gs://meta-cxr-checkpoint/{run_name}/.
set -e

CFG_DIR="pretraining/configs/encoder_comparison"
CKPT_BUCKET="gs://meta-cxr-checkpoint"
LOG_DIR="$HOME/logs"
mkdir -p "$LOG_DIR"

RUNS=(
  "07_all_three"
)

cd "$HOME/META-CXR"

for RUN in "${RUNS[@]}"; do
  echo "================================================================"
  echo "[$(date -Is)] Starting $RUN"
  echo "================================================================"

  LOG="$LOG_DIR/${RUN}.log"
  OUT_DIR="$HOME/output/${RUN}"
  mkdir -p "$OUT_DIR"

  python -m torch.distributed.run --standalone --nproc_per_node=1 \
    -m pretraining.train --cfg-path "${CFG_DIR}/${RUN}.yaml" \
    2>&1 | tee "$LOG" || {
      echo "[$(date -Is)] $RUN FAILED — uploading partial output then continuing"
    }

  echo "[$(date -Is)] Uploading $RUN to $CKPT_BUCKET/${RUN}/"
  gcloud storage cp -r "$OUT_DIR" "$CKPT_BUCKET/${RUN}/" --quiet || true
  gcloud storage cp "$LOG" "$CKPT_BUCKET/${RUN}/train.log" --quiet || true

  echo "[$(date -Is)] $RUN done"
done

echo "================================================================"
echo "[$(date -Is)] TRAINING COMPLETE (07_all_three)"
echo "================================================================"
