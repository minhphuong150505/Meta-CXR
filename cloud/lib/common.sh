#!/usr/bin/env bash
# Helpers for private-GCS/GCP workflows.

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

require_gcp_config() {
  if [ -z "${GCP_PROJECT:-}" ] || [ -z "${GCS_BUCKET:-}" ] \
      || [ -z "${GCS_DATA_BUCKET:-}" ]; then
    log "ERROR: export GCP_PROJECT, GCS_BUCKET, and GCS_DATA_BUCKET first."
    return 1
  fi
  local bucket
  for bucket in "$GCS_BUCKET" "$GCS_DATA_BUCKET"; do
    case "$bucket" in
      gs://*|*/*|*[$'\t\r\n ']*|'')
        log "ERROR: bucket variables must be bare GCS bucket names."
        return 1
        ;;
    esac
  done
}

require_private_bucket() {
  local bucket="${1:?bucket name is required}"
  if ! gcloud storage buckets describe "gs://$bucket" >/dev/null 2>&1; then
    log "ERROR: gs://$bucket does not exist or is not accessible."
    return 1
  fi

  local policy
  policy=$(gcloud storage buckets get-iam-policy "gs://$bucket" --format=json)
  if printf '%s' "$policy" | grep -Eq 'allUsers|allAuthenticatedUsers'; then
    log "ERROR: gs://$bucket has a public IAM principal; refusing to use it."
    return 1
  fi
  local prevention
  prevention=$(gcloud storage buckets describe "gs://$bucket" \
    --format='value(iamConfiguration.publicAccessPrevention)')
  if [ "$prevention" != "enforced" ]; then
    log "ERROR: gs://$bucket does not enforce public-access prevention."
    return 1
  fi

  local uniform_access
  uniform_access=$(gcloud storage buckets describe "gs://$bucket" \
    --format='value(iamConfiguration.uniformBucketLevelAccess.enabled)')
  if [ "$uniform_access" != "True" ] && [ "$uniform_access" != "true" ]; then
    log "ERROR: gs://$bucket does not use uniform bucket-level access."
    return 1
  fi
}

enforce_private_bucket() {
  local bucket="${1:?bucket name is required}"
  gcloud storage buckets update "gs://$bucket" \
    --uniform-bucket-level-access \
    --public-access-prevention=enforced --quiet
  require_private_bucket "$bucket"
}

# upload_gcs SRC DST_PREFIX -- retry three times after privacy validation.
upload_gcs() {
  local src="${1:?source path is required}"
  local dst="${2:?GCS destination is required}"
  if [ ! -e "$src" ]; then
    log "ERROR: upload source does not exist: $src"
    return 1
  fi
  if [ "$src" = "/" ] || [ "$src" = "." ] \
      || { [ -n "${HOME:-}" ] && [ "$src" = "$HOME" ]; }; then
    log "ERROR: refusing to upload a broad filesystem root: $src"
    return 1
  fi
  case "$dst" in
    "gs://$GCS_BUCKET"/*) ;;
    *) log "ERROR: destination must be inside private gs://$GCS_BUCKET/"; return 1 ;;
  esac
  require_private_bucket "$GCS_BUCKET"
  local attempt
  for attempt in 1 2 3; do
    log "gcloud storage cp --recursive $src $dst (attempt $attempt)"
    if gcloud storage cp --recursive "$src" "$dst" --quiet; then
      return 0
    fi
    sleep 30
  done
  log "GCS upload failed after three attempts."
  return 1
}
