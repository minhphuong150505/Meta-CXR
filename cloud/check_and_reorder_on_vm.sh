#!/usr/bin/env bash
# RUNS ON THE VM ITSELF (not from local).
# Hourly: if run 01_biovil_only finished (master_run.log contains
# "Starting 02_pubmedclip_only"), kill tmux 'train' and relaunch
# with reversed order: 07 -> 06 -> 05 -> 04 -> 03 -> 02.
# Self-disables via ~/logs/.reordered_done marker.
#
# Install on VM:
#   crontab -e
#   17 * * * * /home/phuong/check_and_reorder.sh >> /home/phuong/logs/check_and_reorder.log 2>&1

set -u
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:$PATH"

LOG_PREFIX="[$(date -Is)]"
echo "$LOG_PREFIX --- check_and_reorder run ---"

MASTER_LOG="$HOME/logs/master_run.log"
MARKER="$HOME/logs/.reordered_done"

# ---- Step 1: check trigger ----
if [ -f "$MARKER" ]; then
  echo "$LOG_PREFIX ALREADY_REORDERED — nothing to do."
  exit 0
fi

if [ ! -f "$MASTER_LOG" ]; then
  echo "$LOG_PREFIX WAITING — master_run.log chưa tồn tại."
  exit 0
fi

if ! grep -q "Starting 02_pubmedclip_only" "$MASTER_LOG"; then
  echo "$LOG_PREFIX WAITING. Last 2 lines:"
  tail -2 "$MASTER_LOG"
  exit 0
fi

echo "$LOG_PREFIX TRIGGER detected. Proceeding with kill + relaunch."

# ---- Step 2: kill current training ----
tmux kill-session -t train 2>/dev/null || true
sleep 2
pkill -9 -f pretraining.train 2>/dev/null || true
sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true

# ---- Step 3: write reversed script ----
cat > "$HOME/run_reversed.sh" <<'EOF'
#!/usr/bin/env bash
set -e
CFG_DIR="pretraining/configs/encoder_comparison"
CKPT_BUCKET="gs://meta-cxr-checkpoint"
LOG_DIR="$HOME/logs"
mkdir -p "$LOG_DIR"
RUNS=(
  "07_all_three"
  "06_pubmedclip_swin"
  "05_biovil_swin"
  "04_biovil_pubmedclip"
  "03_swin_only"
  "02_pubmedclip_only"
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
    2>&1 | tee "$LOG" || echo "[$(date -Is)] $RUN FAILED — continuing"
  echo "[$(date -Is)] Uploading $RUN"
  gcloud storage cp -r "$OUT_DIR" "$CKPT_BUCKET/${RUN}/" --quiet || true
  gcloud storage cp "$LOG" "$CKPT_BUCKET/${RUN}/train.log" --quiet || true
done
echo "[$(date -Is)] REVERSED ORDER ALL DONE"
EOF
chmod +x "$HOME/run_reversed.sh"

# ---- Step 4: launch + marker ----
touch "$MARKER"
tmux new -d -s train "bash $HOME/run_reversed.sh 2>&1 | tee $HOME/logs/master_run_reversed.log"
sleep 3
tmux ls
echo "$LOG_PREFIX REORDER COMPLETE. Marker $MARKER set."
