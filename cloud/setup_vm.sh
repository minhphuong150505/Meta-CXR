#!/usr/bin/env bash
# One-time setup for a private GCP training VM. No MIMIC-CXR artifact is sent
# to Kaggle or another public data service.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/lib/common.sh"
require_gcp_config

log "=== META-CXR private GCP setup ==="
log "Project=$GCP_PROJECT  Bucket=$GCS_BUCKET  Zone=$GCP_ZONE"

log "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv jq curl libgl1 libglib2.0-0

gcloud config set project "$GCP_PROJECT" --quiet
log "Active gcloud identity:"
gcloud auth list --filter=status:ACTIVE --format='value(account)'

if gcloud storage buckets describe "gs://$GCS_BUCKET" >/dev/null 2>&1; then
  log "Bucket gs://$GCS_BUCKET already exists."
else
  log "Creating private bucket gs://$GCS_BUCKET in $GCP_REGION..."
  gcloud storage buckets create "gs://$GCS_BUCKET" \
    --project="$GCP_PROJECT" \
    --location="$GCP_REGION" \
    --uniform-bucket-level-access
fi

# Defence in depth: even a later accidental allUsers IAM grant is blocked.
enforce_private_bucket "$GCS_BUCKET"
enforce_private_bucket "$GCS_DATA_BUCKET"

for stage in stage1 stage2; do
  if ! gcloud storage objects describe \
      "gs://$GCS_BUCKET/$stage/.keep" >/dev/null 2>&1; then
    printf '' | gcloud storage cp - "gs://$GCS_BUCKET/$stage/.keep" --quiet
  fi
done

log "Private GCS checks passed."
log "Install Stage-1 and Stage-2 dependencies in separate virtual environments."
log "=== Setup complete ==="
