#!/usr/bin/env bash
# Chạy trên máy LOCAL (không phải VM). Đẩy thư mục cloud/ + 2 notebook lên VM
# qua gcloud compute scp.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

REMOTE_DIR="meta-cxr-cloud"

echo "Đẩy lên VM $VM_INSTANCE (zone=$GCP_ZONE)..."

gcloud compute ssh "$VM_INSTANCE" \
  --zone="$GCP_ZONE" --project="$GCP_PROJECT" \
  --command="mkdir -p ~/$REMOTE_DIR"

gcloud compute scp --recurse \
  --zone="$GCP_ZONE" --project="$GCP_PROJECT" \
  "$SCRIPT_DIR" \
  "$SCRIPT_DIR/../$STAGE1_NOTEBOOK" \
  "$SCRIPT_DIR/../$STAGE2_NOTEBOOK" \
  "$VM_INSTANCE:~/$REMOTE_DIR/"

cat <<EOF

== Đã push xong ==

Layout trên VM:
  ~/$REMOTE_DIR/cloud/           # scripts
  ~/$REMOTE_DIR/$STAGE1_NOTEBOOK
  ~/$REMOTE_DIR/$STAGE2_NOTEBOOK

Bước tiếp theo:
1. Lấy kaggle.json từ https://www.kaggle.com/settings và copy lên VM:
     gcloud compute ssh $VM_INSTANCE --zone=$GCP_ZONE --project=$GCP_PROJECT \\
       --command="mkdir -p ~/.kaggle"
     gcloud compute scp ~/Downloads/kaggle.json \\
       $VM_INSTANCE:~/.kaggle/kaggle.json \\
       --zone=$GCP_ZONE --project=$GCP_PROJECT

2. SSH vào VM và setup:
     gcloud compute ssh $VM_INSTANCE --zone=$GCP_ZONE --project=$GCP_PROJECT
     cd ~/$REMOTE_DIR/cloud
     bash setup_vm.sh

3. Chạy training stage 1 (background, ~12h/session):
     nohup bash run_stage1.sh > stage1_\$(date +%s).log 2>&1 &
     tail -f stage1_*.log

4. Khi stage 1 xong → chạy stage 2:
     nohup bash run_stage2.sh > stage2_\$(date +%s).log 2>&1 &
EOF
