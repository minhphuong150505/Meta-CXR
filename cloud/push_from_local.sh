#!/usr/bin/env bash
# Update an existing private code checkout on the GCP VM. Patient data,
# reports, outputs, credentials, and checkpoints are never copied by this tool.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/lib/common.sh"
require_gcp_config

if [ -z "${VM_INSTANCE:-}" ]; then
  log "ERROR: export VM_INSTANCE before running this script."
  exit 1
fi

REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-META-CXR}"
case "$REMOTE_REPO_DIR" in
  *[!A-Za-z0-9._/-]*|/*|*..*)
    log "ERROR: REMOTE_REPO_DIR must be a relative, shell-safe path."
    exit 1
    ;;
esac

log "Updating the private Git checkout on $VM_INSTANCE..."
gcloud compute ssh "$VM_INSTANCE" \
  --zone="$GCP_ZONE" \
  --project="$GCP_PROJECT" \
  --command="git -C ~/$REMOTE_REPO_DIR pull --ff-only"

log "Code updated. Data and artifacts remain in private GCS/local VM storage."
