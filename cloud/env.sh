#!/usr/bin/env bash
# Shared GCP defaults. Export project/bucket values in the shell or in a local,
# untracked wrapper before sourcing this file; do not commit machine identities.

export GCP_PROJECT="${GCP_PROJECT:-}"
export GCP_ZONE="${GCP_ZONE:-us-central1-a}"
export GCP_REGION="${GCP_REGION:-us-central1}"
export VM_INSTANCE="${VM_INSTANCE:-}"

# Bucket names do not include gs://. Both buckets must have uniform access and
# public-access prevention enabled.
export GCS_DATA_BUCKET="${GCS_DATA_BUCKET:-mimic-cxr-jpg-dataset-phuongnm}"
export GCS_BUCKET="${GCS_BUCKET:-}"

export STAGE1_CONFIG="${STAGE1_CONFIG:-pretraining/configs/mimic_cxr_full_l4.yaml}"
export STAGE1_RUN="${STAGE1_RUN:-mimic_cxr_full_l4_blip2}"
export STAGE2_IMAGE_MODE="${STAGE2_IMAGE_MODE:-both}"
