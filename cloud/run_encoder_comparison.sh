#!/usr/bin/env bash
# Sequentially runs 7 encoder comparison configs.
# Uploads checkpoints + final output_dir to gs://meta-cxr-checkpoint/{run_name}/.
set -e

CFG_DIR="pretraining/configs/encoder_comparison"
CKPT_BUCKET="gs://meta-cxr-checkpoint"
LOG_DIR="$HOME/logs"
mkdir -p "$LOG_DIR"

RUNS=(
  "01_biovil_only"
  "02_pubmedclip_only"
  "03_swin_only"
  "04_biovil_pubmedclip"
  "05_biovil_swin"
  "06_pubmedclip_swin"
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
echo "[$(date -Is)] ALL 7 RUNS COMPLETE"
echo "================================================================"
