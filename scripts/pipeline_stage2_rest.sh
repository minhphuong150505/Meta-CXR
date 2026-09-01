#!/usr/bin/env bash
# Everything left to run after the Stage-2 arm A fine-tune, in order, on one GPU.
#
# ⚠ ARM B (native_anchor_guided) IS NOT RUN HERE. Native manifest records carry
# no `pred_groups`, so that arm trained on prompts IDENTICAL to arm A. It now
# RAISES instead of degrading silently (stage2/prompts/records.py), which is the
# fix; wiring the cues onto the native route was superseded by arm C below.
#
# ARM C (meta_cxr_native_qformer_guided) is the originally-designed architecture:
# MedGemma keeps its own vision tower AND additionally receives the 32 Q-Former
# soft tokens, with MHCAC cues in the text. It requires a Stage-1 pass, which is
# why the smoke gate exists -- see step 1.
#
# Rules this encodes, each of which costs real time if broken:
#   * ONE GPU job at a time -- never overlap;
#   * check the ARTIFACT between steps, never the exit code, and never assume
#     a previous step succeeded;
#   * a failed step stops the chain and says why. Silently carrying on would
#     produce a metric with nothing valid behind it.
#
# Progress and every abort reason go to ~/pipeline_rest.status.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
cd "$HOME/Meta-CXR" || exit 1
PY=~/.venvs/meta-cxr-stage1-311/bin/python
MAN=/mnt/drive1tb/mimic-cxr-jpg-full/processed/full_allviews_v2/test.csv
IMG=/mnt/drive1tb/mimic-cxr-jpg-full
NOTE=$HOME/pipeline_rest.status
A=$HOME/ft_only_full
ADIR=$A/adapters/medgemma_qlora_medgemma_direct

log() { echo "[pipeline $(date '+%F %T')] $*" | tee -a "$NOTE"; }
die() { log "ABORT: $*"; exit 1; }
gpu_free() { while pgrep -f "run_medgemma_qlora|generate_stage2_reports|explain_stage2|evaluate_stage2" >/dev/null; do sleep 60; done; sleep 20; }
complete() {  # $1 = adapter dir, $2 = label
  [ -f "$1/adapter_model.safetensors" ] || { log "  $2: no adapter"; return 1; }
  s=$($PY -c "import json;print(json.load(open('$1/manifest.json')).get('status','?'))" 2>/dev/null)
  [ "$s" = "complete" ] || { log "  $2: status=$s (not complete)"; return 1; }
  return 0
}
watch_env() { printf 'EXPECT_RUNNING=%s\nRUN_DIR=%s\nLOG=%s\n' "$1" "$2" "$3" > ~/.train_watchdog.env; }

: > "$NOTE"; log "pipeline armed (arm A only; arm B removed); waiting for arm A"
gpu_free
complete "$ADIR" "arm A" || die "arm A did not complete; nothing downstream started"
log "arm A complete: $(grep '^\[epoch' $A.log | tail -1)"
watch_env 0 "" ""

C=$HOME/ft_guided_full
CDIR=$C/adapters/medgemma_qlora_meta_cxr_native_qformer_guided
CFG=configs/experiment_native_qformer_guided.yaml

# ---- 1. SMOKE the combined mode before spending 70 h on it ----------------
# The Stage-1 record pass runs at batch 1 over ~220k studies and caches ~10 GiB
# of soft tokens; the prompt must carry 32 <qformer_soft_token> placeholders and
# a real image at the same time. None of that has run on a GPU. Prove it small
# first: a failure here costs 20 minutes, the same failure at hour 3 of the full
# run costs the Stage-1 pass as well.
SM=$HOME/ft_guided_smoke
log "STEP 1/5: arm C smoke (200 train studies)"
rm -rf "$SM"
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True $PY \
  training/run_medgemma_qlora.py --pipeline-mode meta_cxr_native_qformer_guided \
  --section-mode findings_only --prompt-config "$CFG" \
  --train-limit 200 --val-limit 10 --test-limit 10 \
  --train-epochs 1 --batch-size 2 --grad-accum 8 --num-workers 4 \
  --output-dir "$SM" --no-upload >> "$HOME/smoke_guided.log" 2>&1
complete "$SM/adapters/medgemma_qlora_meta_cxr_native_qformer_guided" "arm C smoke" \
  || die "arm C smoke failed -- read $HOME/smoke_guided.log; NOT starting the 70 h run"
log "  smoke ok; the combined mode trains end to end"

# ---- 2. arm C, full ------------------------------------------------------
log "STEP 2/5: arm C full (~70 h)"
rm -rf "$C" "$C.log"; watch_env 1 "$C/adapters" "$C.log"
setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True $PY \
  training/run_medgemma_qlora.py --pipeline-mode meta_cxr_native_qformer_guided \
  --section-mode findings_only --prompt-config "$CFG" \
  --train-epochs 1 --batch-size 2 --grad-accum 8 --num-workers 4 \
  --save-every-updates 250 --skip-test --output-dir "$C" --no-upload \
  > "$C.log" 2>&1 < /dev/null &
sleep 120
pgrep -f run_medgemma_qlora >/dev/null || die "arm C failed to start; see $C.log"
gpu_free
complete "$CDIR" "arm C" || die "arm C did not complete"
log "arm C complete: $(grep '^\[epoch' $C.log | tail -1)"
watch_env 0 "" ""

# ---- 3. generate on TEST --------------------------------------------------
OUT=$HOME/gen_test_only
log "STEP 3/5: generating test reports from the fine-tuned adapter"
rm -rf "$OUT"
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True $PY \
  scripts/generate_stage2_reports.py --manifest "$MAN" --image-root "$IMG" \
  --output-dir "$OUT" --split test --limit 0 --adapter "$ADIR" >> "$HOME/gen_only.log" 2>&1
[ -f "$OUT/generated_test.jsonl" ] || die "generation produced nothing"
log "  $(wc -l < $OUT/generated_test.jsonl) reports"
# The mode key is the guard against silently reporting a zero-shot run as ours.
m=$($PY -c "import json;print(json.load(open('$OUT/summary.json'))['mode'])" 2>/dev/null)
[ "$m" = "medgemma_direct_finetuned" ] || die "generation ran as '$m', not finetuned"

# ---- 2. NLG metrics -------------------------------------------------------
log "STEP 4/5: NLG metrics"
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate_stage2.py \
  --predictions "$OUT/generated_test.jsonl" \
  --metrics bleu,rouge,meteor,cider,bertscore --skip-clinical-metrics \
  --bootstrap-samples 1000 --split test \
  --output-dir "$HOME/eval_final/stage2_test_only" >> "$HOME/nlg_only.log" 2>&1
[ -f "$HOME/eval_final/stage2_test_only/metrics.json" ] || die "metrics failed"
log "  metrics written"

# ---- 3. Stage-2 XAI on the FINE-TUNED model -------------------------------
# --adapter is not optional here. Without it this explains the base model, which
# is a zero-shot baseline and not this project's Stage 2; it fails silently,
# producing a complete and entirely valid-looking summary.json about the wrong
# model. The mode check below is the backstop.
log "STEP 5/5: Stage-2 XAI (fine-tuned)"
rm -rf "$HOME/xai_test_only"
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True $PY \
  scripts/explain_stage2.py --manifest "$MAN" --image-root "$IMG" \
  --output-dir "$HOME/xai_test_only" --split test --limit 300 \
  --ablation-studies 100 --adapter "$ADIR" --verbose >> "$HOME/xai_only.log" 2>&1
if [ -f "$HOME/xai_test_only/summary.json" ]; then
  x=$($PY -c "import json;print(json.load(open('$HOME/xai_test_only/summary.json'))['mode'])" 2>/dev/null)
  log "  XAI done, mode=$x"
  [ "$x" = "medgemma_direct_finetuned" ] || log "  ⚠ XAI ran as '$x' -- do not report it as ours"
else
  log "  XAI did not finish (a gate may have aborted it -- read $HOME/xai_only.log)"
fi

log "PIPELINE DONE. Remaining by hand:"
log "  * MS-CXR grounding comparison -- no script yet; Stage-2 maps are in"
log "    MedGemma's 896 frame, MS-CXR boxes are in original-image coordinates."
log "  * Generation / NLG / XAI for ARM C -- the combined mode needs Stage-1"
log "    records at generation time, which scripts/generate_stage2_reports.py"
log "    does not build. Compare arm A vs arm C by hand until it does."
