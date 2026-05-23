#!/usr/bin/env bash
# Tham số tập trung. Source file này trước khi chạy bất kỳ script nào khác.

export GCP_PROJECT="mimic-cxr-jpg-491409"
export GCP_ZONE="us-central1-a"
export GCP_REGION="us-central1"
export VM_INSTANCE="instance-20260521-072851"

export GCS_BUCKET="meta-cxr-checkpoint"

export KAGGLE_USERNAME="phuong20052"
export STAGE1_KERNEL_SLUG="phuong20052/meta-cxr-stage1-train"
export STAGE2_KERNEL_SLUG="phuong20052/meta-cxr-stage2-eval"
export STAGE1_NOTEBOOK="meta-cxt-kaggle-train.ipynb"
export STAGE2_NOTEBOOK="META_CXR_eval_kaggle.ipynb"

export POLL_INTERVAL_SECS=300
export MAX_POLL_HOURS=13
