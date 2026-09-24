#!/usr/bin/env bash
# Stage 1 as the META-CXR paper trains it: phase 1a -> 1b -> 1c (D-020).
#
#   ROOT=$HOME/run_<date>_3phase bash scripts/run_stage1_phases.sh
#
# Environment:
#   ROOT        run root; each phase writes <ROOT>/<phase>/, and every
#               checkpoint_<phase>.pth also lands in <ROOT>/ (run.phase_root)
#   CACHE       phase-1a feature cache (anchor-only, train+val); built if absent
#   PHASES      subset/order to run, default "phase1a phase1b phase1c"
#   EXTRA_OPTS  extra --options for every phase (smoke: truncation, epochs)
#   CACHE_OPTS  extra arguments for the cache build (smoke: --truncate N)
#   PY, CFG     interpreter and YAML
#
# Stops -- non-zero exit, nothing further launched -- when:
#   * /mnt/drive1tb is not ntfs3 (the FUSE driver stalls training),
#   * another training/generation process holds the GPU,
#   * a phase output directory already exists (one card, one run, one path),
#   * phase 1a's ITC gate fails after 2 epochs (PHASE_GATE_FAILED; exit 3),
#   * a phase ends without its checkpoint_<phase>.pth (exit 4).
set -euo pipefail

ROOT=${ROOT:?set ROOT to a new directory on /home}
CACHE=${CACHE:-$ROOT/feature_cache_anchor}
PHASES=${PHASES:-"phase1a phase1b phase1c"}
EXTRA_OPTS=${EXTRA_OPTS:-}
CACHE_OPTS=${CACHE_OPTS:-}
PY=${PY:-$HOME/.venvs/meta-cxr-stage1-311/bin/python}
CFG=${CFG:-pretraining/configs/mimic_cxr_full.yaml}

log() { echo "[$(date '+%F %T')] $*"; }

fstype=$(findmnt -no FSTYPE /mnt/drive1tb || true)
if [ -n "$fstype" ] && [ "$fstype" != ntfs3 ]; then
  log "ABORT: /mnt/drive1tb is $fstype, not ntfs3"; exit 1
fi
if ps -eo args --no-headers | awk '$0 ~ /pretraining\.train|run_medgemma_qlora|generate_stage2|precompute_features/ && $0 !~ /awk/ {f=1} END {exit !f}'; then
  log "ABORT: another training/generation process is running"; exit 1
fi
for ph in $PHASES; do
  if [ -e "$ROOT/$ph" ]; then log "ABORT: $ROOT/$ph already exists"; exit 1; fi
done
mkdir -p "$ROOT"

if [[ " $PHASES " == *" phase1a "* ]] && [ ! -e "$CACHE/swin/train_feats.npy" ]; then
  log "building anchor-only feature cache in $CACHE"
  # shellcheck disable=SC2086
  "$PY" -m pretraining.precompute_features --cfg-path "$CFG" --output-dir "$CACHE" \
      --splits train val --anchor-only $CACHE_OPTS \
      --options run.distributed=false run.world_size=1
fi

for ph in $PHASES; do
  opts=(run.phase="$ph" run.phase_root="$ROOT" run.output_dir="$ROOT/$ph")
  if [ "$ph" = phase1a ]; then opts+=(run.feature_cache_dir="$CACHE"); fi
  log "phase $ph: starting"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} WANDB_MODE=${WANDB_MODE:-disabled} \
    "$PY" -m pretraining.train --cfg-path "$CFG" --options "${opts[@]}" $EXTRA_OPTS
  if compgen -G "$ROOT/$ph/*/PHASE_GATE_FAILED" > /dev/null; then
    log "STOP: $ph ITC gate failed; see $(compgen -G "$ROOT/$ph/*/PHASE_GATE_FAILED")"
    exit 3
  fi
  if [ ! -f "$ROOT/checkpoint_$ph.pth" ]; then
    log "STOP: $ph finished without $ROOT/checkpoint_$ph.pth"; exit 4
  fi
  log "phase $ph: done -> $ROOT/checkpoint_$ph.pth"
done
log "all phases done"
