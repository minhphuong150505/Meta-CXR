#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/META-CXR}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/medgemma-qformer}"
GCS_OUT="${GCS_OUT:-gs://meta-cxr-checkpoint/eval/MedGemma_QFormer}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/meta-cxr-checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output/medgemma_qformer}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-300}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MEDGEMMA_MODEL_ID="${MEDGEMMA_MODEL_ID:-google/medgemma-1.5-4b-it}"
MEDGEMMA_LORA_ID="${MEDGEMMA_LORA_ID:-DeepRadiology/medgemma1.5-CXR}"
DATA_GCS="${DATA_GCS:-gs://mimic-cxr-jpg-data}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

cd "$PROJECT_DIR"

if ! gcloud storage ls "$DATA_GCS" >/dev/null 2>&1; then
  echo "[warn] DATA_GCS=$DATA_GCS is not accessible from this VM. Using env_config.yaml data paths instead."
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  rm -rf "$VENV_DIR"
  if ! python3 -m venv --system-site-packages "$VENV_DIR"; then
    echo "[warn] python3 -m venv failed; falling back to virtualenv."
    python3 -m pip install --user -q --upgrade virtualenv
    python3 -m virtualenv --system-site-packages "$VENV_DIR"
  fi
fi

source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q --upgrade \
  "transformers>=4.53.0,<4.58" \
  "peft>=0.18.0" \
  "accelerate>=1.8.0" \
  "huggingface_hub>=0.30.0" \
  "safetensors>=0.4.5" \
  "sentencepiece>=0.2.0" \
  "protobuf>=4.25.0" \
  "bert-score>=0.3.13" \
  "pycocoevalcap>=1.2" \
  "nltk>=3.8" \
  "pandas>=2.0,<3" \
  "numpy>=1.26,<2.3" \
  "scikit-learn>=1.4" \
  "radgraph>=0.1.18"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/model:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export DISABLE_TORCH_COMPILE=1

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" && ! -f "$HOME/.cache/huggingface/token" ]]; then
  echo "[warn] HF_TOKEN is not set. google/medgemma-1.5-4b-it is gated and will fail unless this VM is logged in to Hugging Face."
fi

GEN_LOG="$LOG_DIR/medgemma_qformer_generation.log"
METRIC_LOG="$LOG_DIR/medgemma_qformer_metrics.log"

python eval_bertscore_medgemma_qformer.py \
  --sample-limit "$SAMPLE_LIMIT" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --gcs-output "$GCS_OUT" \
  --medgemma-model-id "$MEDGEMMA_MODEL_ID" \
  --medgemma-lora-id "$MEDGEMMA_LORA_ID" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-workers "$NUM_WORKERS" \
  --reuse-stage1-cache \
  "$@" \
  2>&1 | tee "$GEN_LOG"

python eval_metrics_medgemma_qformer.py \
  --input-dir "$OUTPUT_DIR" \
  --output-base "$OUTPUT_DIR/metrics" \
  --gcs-output "$GCS_OUT/metrics" \
  2>&1 | tee "$METRIC_LOG"

gcloud storage cp "$GEN_LOG" "$GCS_OUT/logs/medgemma_qformer_generation.log" --quiet
gcloud storage cp "$METRIC_LOG" "$GCS_OUT/logs/medgemma_qformer_metrics.log" --quiet
gcloud storage cp eval_bertscore_medgemma_qformer.py "$GCS_OUT/scripts/eval_bertscore_medgemma_qformer.py" --quiet
gcloud storage cp eval_metrics_medgemma_qformer.py "$GCS_OUT/scripts/eval_metrics_medgemma_qformer.py" --quiet
gcloud storage cp cloud/run_medgemma_qformer_eval.sh "$GCS_OUT/scripts/run_medgemma_qformer_eval.sh" --quiet

echo "[done] outputs: $GCS_OUT/"
