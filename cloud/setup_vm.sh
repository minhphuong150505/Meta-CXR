#!/usr/bin/env bash
# Setup script chạy 1 lần trên VM Google Cloud.
# Cài kaggle CLI, verify gcloud, tạo GCS bucket cho stage1/stage2 checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/lib/common.sh"

log "=== META-CXR VM setup ==="
log "Project=$GCP_PROJECT  Bucket=$GCS_BUCKET  Zone=$GCP_ZONE"

# 1. APT packages
log "Installing apt packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv jq curl

# 2. Kaggle CLI
log "Installing kaggle CLI in a virtualenv..."
KAGGLE_VENV="$HOME/.venvs/kaggle-cli"
python3 -m venv "$KAGGLE_VENV"
"$KAGGLE_VENV/bin/python" -m pip install --upgrade pip
"$KAGGLE_VENV/bin/python" -m pip install --upgrade kaggle

if ! grep -q 'HOME/.venvs/kaggle-cli/bin' "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/.venvs/kaggle-cli/bin:$PATH"' >> "$HOME/.bashrc"
  log "Added kaggle CLI virtualenv to PATH in ~/.bashrc"
fi
export PATH="$KAGGLE_VENV/bin:$PATH"

# 3. Kaggle credentials
KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
mkdir -p "$HOME/.kaggle"
if [ ! -f "$KAGGLE_JSON" ]; then
  log "WARNING: $KAGGLE_JSON không tồn tại."
  log "  Lấy file từ https://www.kaggle.com/settings → Create New Token"
  log "  Rồi copy lên VM (chạy trên máy local):"
  log "    gcloud compute scp ~/Downloads/kaggle.json $VM_INSTANCE:~/.kaggle/kaggle.json \\"
  log "      --zone=$GCP_ZONE --project=$GCP_PROJECT"
  log "  Sau đó chạy lại setup_vm.sh."
else
  chmod 600 "$KAGGLE_JSON"
  log "kaggle.json OK (user=$(jq -r .username < "$KAGGLE_JSON"))"
  if kaggle kernels list -m --page-size 1 >/dev/null 2>&1; then
    log "Kaggle CLI authenticated."
  else
    log "WARNING: Kaggle CLI auth failed — kiểm tra lại kaggle.json."
  fi
fi

# 4. gcloud + service account scope
log "gcloud auth list:"
gcloud auth list

SCOPE_URL="http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/scopes"
SCOPES=$(curl -s -H "Metadata-Flavor: Google" "$SCOPE_URL" 2>/dev/null || echo "")
log "VM service account scopes:"
echo "$SCOPES"
if ! echo "$SCOPES" | grep -qE 'devstorage.(full_control|read_write)|cloud-platform'; then
  log "WARNING: VM service account thiếu storage scope."
  log "  Stop VM → Edit → Cloud API access scopes → 'Allow full access to all Cloud APIs' → Save → Start."
fi

# 5. GCS bucket
gcloud config set project "$GCP_PROJECT" --quiet
if gsutil ls -b "gs://$GCS_BUCKET" >/dev/null 2>&1; then
  log "Bucket gs://$GCS_BUCKET đã tồn tại."
else
  log "Tạo bucket gs://$GCS_BUCKET tại $GCP_REGION..."
  gsutil mb -p "$GCP_PROJECT" -l "$GCP_REGION" -b on "gs://$GCS_BUCKET"
fi

# 6. Subfolder placeholders
for stage in stage1 stage2; do
  if ! gsutil -q stat "gs://$GCS_BUCKET/$stage/.keep" 2>/dev/null; then
    echo "" | gsutil cp - "gs://$GCS_BUCKET/$stage/.keep"
    log "Created gs://$GCS_BUCKET/$stage/"
  fi
done

# 7. Final verify
log "Bucket layout:"
gsutil ls "gs://$GCS_BUCKET/"

log "=== Setup complete ==="
log "Next: bash run_stage1.sh   (hoặc run_stage2.sh)"
