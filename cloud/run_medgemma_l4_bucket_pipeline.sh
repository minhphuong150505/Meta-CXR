#!/usr/bin/env bash
# Run on the GCP L4 VM. Pulls data from gs://mimic-cxr-lite-data, fine-tunes
# MedGemma with QLoRA, and writes NLP metrics for exactly 200 test samples.
set -Eeuo pipefail

export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export DISABLE_TORCH_COMPILE=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_ID="${PROJECT_ID:-mimic-cxr-jpg-491409}"
GCS_BUCKET="${GCS_BUCKET:-gs://mimic-cxr-lite-data}"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$HOME/checkpoints}"
REPO_DIR="${REPO_DIR:-$HOME/META-CXR}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/output/medgemma_qlora}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$REPO_DIR/output/medgemma_qlora_smoke_eager}"
GCS_OUT="${GCS_OUT:-$GCS_BUCKET/outputs/MedGemma_QLoRA}"
GCS_SMOKE_OUT="${GCS_SMOKE_OUT:-$GCS_BUCKET/outputs/MedGemma_QLoRA_smoke_eager}"
TRAIN_LIMIT="${TRAIN_LIMIT:-100000}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"

mkdir -p "$HOME/logs"
LOG="$HOME/logs/medgemma_l4_pipeline.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

sync_log() {
  gcloud storage cp "$LOG" "$GCS_OUT/pipeline.log" --quiet 2>/dev/null || true
}

note() {
  echo
  echo "=== $* === $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sync_log
}

fail() {
  echo "PIPELINE_FAILED: $*"
  sync_log
  exit 1
}

( while true; do sleep 60; sync_log; done ) &
SYNCER=$!
trap 'kill "$SYNCER" 2>/dev/null || true; sync_log' EXIT

export HF_TOKEN="$(cat "$HOME/.hf_token" 2>/dev/null || true)"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || fail "Missing Hugging Face token at ~/.hf_token"

note "STEP0 configure gcloud"
gcloud config set project "$PROJECT_ID" --quiet
gcloud storage ls "$GCS_BUCKET/" --quiet >/dev/null || fail "Cannot access $GCS_BUCKET"

note "STEP1 install system and Python dependencies"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv openjdk-11-jdk libgl1 libglib2.0-0 tmux jq
"$PYTHON_BIN" -m pip install --user --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install --user --upgrade \
  "numpy<2" "pandas>=1.5,<3" "scikit-image>=0.22" "pillow>=10" \
  "omegaconf==2.3.0" "iopath>=0.1.10" "timm>=0.9" "matplotlib>=3.7" \
  "scikit-learn>=1.3" "scipy>=1.10" "tqdm>=4.65" "pyyaml>=6" \
  "opencv-python-headless>=4.8" "transformers==4.53.0" \
  "tokenizers>=0.21,<0.22" "huggingface_hub>=0.30,<1.0" \
  "accelerate>=0.30" "bitsandbytes>=0.46" "peft>=0.13" \
  "bert-score>=0.3.13" "pycocoevalcap>=1.2" "nltk>=3.8" \
  "sentencepiece>=0.1.99" "protobuf>=4" "einops>=0.7" "fairscale>=0.4.13" \
  "webdataset>=0.2" "decord>=0.6" "ftfy>=6.1" "regex>=2023" \
  "iterative-stratification>=0.1.7" "hi-ml-multimodal>=0.2.1" \
  "wandb>=0.16"
"$PYTHON_BIN" -m pip install --user --upgrade "radgraph>=0.1" || \
  echo "[warn] radgraph install failed; eval_final will record RadGraph/RadCliQ as unavailable"

note "STEP2 sync dataset, metadata, processed CSVs, and checkpoint"
mkdir -p "$DATA_ROOT/p10" "$DATA_ROOT/processed" "$DATA_ROOT/csv" "$CHECKPOINT_ROOT/07_all_three"
gcloud storage rsync -r "$GCS_BUCKET/p10" "$DATA_ROOT/p10" --quiet
gcloud storage rsync -r "$GCS_BUCKET/processed" "$DATA_ROOT/processed" --quiet
gcloud storage rsync -r "$GCS_BUCKET/csv" "$DATA_ROOT/csv" --quiet
gcloud storage cp "$GCS_BUCKET/checkpoints/07_all_three/checkpoint_best.pth" \
  "$CHECKPOINT_ROOT/07_all_three/checkpoint_best.pth" --quiet

test -f "$DATA_ROOT/processed/train.csv" || fail "Missing $DATA_ROOT/processed/train.csv"
test -f "$DATA_ROOT/processed/test.csv" || fail "Missing $DATA_ROOT/processed/test.csv"
test -f "$DATA_ROOT/csv/mimic-cxr-2.0.0-chexpert.csv" || fail "Missing CheXpert CSV"
test -f "$CHECKPOINT_ROOT/07_all_three/checkpoint_best.pth" || fail "Missing 07_all_three checkpoint"
echo "p10 files: $(find "$DATA_ROOT/p10" -type f | wc -l)"

note "STEP3 patch META-CXR env_config"
cat > "$REPO_DIR/configs/env_config.yaml" <<EOF
paths:
  data_root: "$DATA_ROOT"
  mimic_cxr_jpg_root: "$DATA_ROOT"
  split_csv: "$DATA_ROOT/csv/mimic-cxr-2.0.0-split.csv"
  reports_csv: "$DATA_ROOT/csv/mimic_cxr_cleaned.csv"
  chexpert_csv: "$DATA_ROOT/csv/mimic-cxr-2.0.0-chexpert.csv"
  metadata_csv: "$DATA_ROOT/csv/mimic-cxr-2.0.0-metadata.csv"
  processed_dir: "$DATA_ROOT/processed"
  processed_train_csv: "$DATA_ROOT/processed/train.csv"
  processed_val_csv: "$DATA_ROOT/processed/val.csv"
  processed_test_csv: "$DATA_ROOT/processed/test.csv"
  output_dir: "$HOME/output"
  checkpoint_dir: "$CHECKPOINT_ROOT"
  gcs_bucket: "$GCS_BUCKET"
  gcs_project: "$PROJECT_ID"

wandb:
  entity: "phuongnm150505-uit"
  project: "meta-cxr-encoder-comparison"

java:
  home: "/usr/lib/jvm/java-11-openjdk-amd64"
  path: "/usr/lib/jvm/java-11-openjdk-amd64/bin:"
EOF

note "STEP4 import and model-access smoke test"
cd "$REPO_DIR"
"$PYTHON_BIN" - <<'PY'
import sys
import torch
sys.path.insert(0, ".")
sys.path.insert(0, "model")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
from transformers import AutoProcessor
AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset
from model.lavis.models.blip2_models.blip2_qformer import Blip2Qformer
print("SMOKE_IMPORT_OK")
PY

note "STEP5 smoke fine-tune/eval"
"$PYTHON_BIN" training/run_medgemma_qlora.py \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --train-limit 5 \
  --eval-limit 3 \
  --train-epochs 1 \
  --grad-accum 1 \
  --num-workers "$NUM_WORKERS" \
  --output-dir "$SMOKE_OUTPUT_DIR" \
  --gcs-output "$GCS_SMOKE_OUT" || fail "Smoke run failed"

note "STEP6 full MedGemma QLoRA fine-tune/eval on 200 test samples"
"$PYTHON_BIN" training/run_medgemma_qlora.py \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --train-limit "$TRAIN_LIMIT" \
  --eval-limit "$EVAL_LIMIT" \
  --train-epochs "$TRAIN_EPOCHS" \
  --grad-accum "$GRAD_ACCUM" \
  --num-workers "$NUM_WORKERS" \
  --output-dir "$OUTPUT_DIR" \
  --gcs-output "$GCS_OUT" || fail "Full run failed"

note "STEP7 export Table 6-style MedGemma NLP metrics"
METRICS_JSON="$OUTPUT_DIR/eval/metrics_medgemma_qlora_fine_07_all_three.json"
TABLE_CSV="$OUTPUT_DIR/eval/table6_medgemma_qlora_200_metrics.csv"
test -f "$METRICS_JSON" || fail "Missing metrics JSON: $METRICS_JSON"
jq -r '
  ["Method","Vision Encoders","LLM","Alignment Module","Medical Knowledge Integration","Run","N","BLEU-1","BLEU-2","BLEU-3","BLEU-4","METEOR","ROUGE-L","CIDEr","BERTScore"],
  ["META-CXR","RN50 + ViT + Swin","MedGemma","META-Former","IT + MHCAC",.Run,(.N|tostring),(.["BLEU-1"]|tostring),(.["BLEU-2"]|tostring),(.["BLEU-3"]|tostring),(.["BLEU-4"]|tostring),(.METEOR|tostring),(.["ROUGE-L"]|tostring),(.CIDEr|tostring),(.BERTScore|tostring)]
  | @csv
' "$METRICS_JSON" > "$TABLE_CSV"
gcloud storage cp "$TABLE_CSV" "$GCS_OUT/eval/$(basename "$TABLE_CSV")" --quiet
gcloud storage cp "$LOG" "$GCS_OUT/pipeline.log" --quiet

note "STEP8 eval_final tables and figures for finetuned Vicuna + MedGemma on 200 samples"
"$PYTHON_BIN" evaluation/eval_final_200.py \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --sample-limit 200 \
  --num-workers "$NUM_WORKERS" \
  --medgemma-adapter "$OUTPUT_DIR/adapters/medgemma_qlora_fine" \
  --medgemma-eval-dir "$OUTPUT_DIR/eval" \
  --output-dir "$REPO_DIR/eval_final" \
  --gcs-output "$GCS_BUCKET/outputs/eval_final" || fail "eval_final failed"
gcloud storage cp "$LOG" "$GCS_OUT/pipeline.log" --quiet

note "PIPELINE_COMPLETE"
