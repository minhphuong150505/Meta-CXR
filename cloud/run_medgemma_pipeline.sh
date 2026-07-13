#!/bin/bash
# Autonomous MedGemma QLoRA pipeline. Streams log to GCS for cheap remote
# monitoring. Aborts (and uploads failure log) on any hard error.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || true)
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
cd ~/META-CXR

LOG=~/pipeline.log
GCSLOG=gs://meta-cxr-checkpoint/eval/MedGemma_QLoRA/pipeline.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
note(){ echo "=== $* === $(date -u +%Y-%m-%dT%H:%M:%SZ)"; gcloud storage cp "$LOG" "$GCSLOG" 2>/dev/null || true; }
fail(){ echo "PIPELINE_FAILED: $*"; gcloud storage cp "$LOG" "$GCSLOG" 2>/dev/null || true; exit 1; }

# periodic log sync every 60s
( while true; do sleep 60; gcloud storage cp "$LOG" "$GCSLOG" 2>/dev/null || true; done ) &
SYNCER=$!
trap 'kill $SYNCER 2>/dev/null; gcloud storage cp "$LOG" "$GCSLOG" 2>/dev/null || true' EXIT

[ -n "$HF_TOKEN" ] || fail "no HF token at ~/.hf_token"

note "STEP0 wait for image download to finish"
while pgrep -f "kaggle datasets download" >/dev/null 2>&1; do sleep 30; done
tail -3 ~/kaggle_dl.log 2>/dev/null || true

note "STEP1 locate p10 image root + patch env_config"
P10DIR=$(find ~/data -maxdepth 4 -type d -name p10 2>/dev/null | head -1)
[ -n "$P10DIR" ] || fail "no p10/ dir found under ~/data (download incomplete?)"
VISROOT=$(dirname "$P10DIR")
echo "p10 dir: $P10DIR ; vis_root: $VISROOT"
python3 - "$VISROOT" <<'PY'
import sys, re, pathlib, glob
vis=sys.argv[1]
p=pathlib.Path.home()/ "META-CXR/configs/env_config.yaml"
s=p.read_text()
s=re.sub(r'(\s*mimic_cxr_jpg_root:\s*).*', r'\g<1>"%s"'%vis, s)
s=re.sub(r'(\s*data_root:\s*).*', r'\g<1>"%s"'%vis, s)
ch=glob.glob(str(pathlib.Path(vis)/ "**/mimic-cxr-2.0.0-chexpert.csv"), recursive=True)
if ch:
    s=re.sub(r'(\s*chexpert_csv:\s*).*', r'\g<1>"%s"'%ch[0], s)
p.write_text(s)
print("patched env_config: vis_root=%s chexpert=%s"%(vis, ch[0] if ch else "UNCHANGED"))
PY

note "STEP2 verify checkpoint 07"
ls -lh /mnt/meta-cxr-checkpoint/07_all_three/checkpoint_best.pth || fail "checkpoint 07 missing"

note "STEP3 upgrade transformers stack for MedGemma (Gemma3)"
pip install -q "transformers==4.53.0" "tokenizers>=0.21,<0.22" "huggingface_hub>=0.23" \
    "accelerate>=0.30" "bitsandbytes>=0.43" "peft>=0.11" 2>&1 | tail -3 || true
python3 -c "import transformers,peft,bitsandbytes,accelerate; print('transformers',transformers.__version__,'peft',peft.__version__)" || fail "dep import failed"

note "STEP4 env compat: LAVIS + MedGemma in one process"
python3 - <<'PY'
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'model')
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers import AutoModelForImageTextToText
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset
from model.lavis.models.blip2_models.blip2_qformer import Blip2Qformer
print("ENV_COMPAT_OK: LAVIS + MedGemma import coexist")
PY
[ $? -eq 0 ] || fail "ENV_COMPAT failed (transformers 4.53 vs LAVIS) - needs version retune"

note "STEP5 SMOKE run (train 5 / eval 3)"
python3 training/run_medgemma_qlora.py \
    --train-limit 5 --eval-limit 3 --train-epochs 1 --grad-accum 1 \
    --output-dir ~/META-CXR/output/medgemma_qlora_smoke \
    --gcs-output gs://meta-cxr-checkpoint/eval/MedGemma_QLoRA_smoke || fail "SMOKE failed"
note "SMOKE_OK"

note "STEP6 FULL run (train full ~18k / eval 300)"
python3 training/run_medgemma_qlora.py \
    --train-limit 100000 --eval-limit 300 --train-epochs 1 --grad-accum 8 || fail "FULL run failed"

note "PIPELINE_COMPLETE"
