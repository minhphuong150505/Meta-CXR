# PLAN 2026-10-01 — Logit-adjusted classification loss (D-025)

Planner: Claude (dev box). User decision 2026-10-01: implement "option 1" from
the Uncertain-label literature review — replace the capped inverse-frequency
`class_weights` with the logit-adjusted loss (Menon et al., ICLR 2021, Eq. 10),
as PromptMRG (Jin et al., AAAI 2024) does on MIMIC-CXR.

## What changed

- `mhcac.loss.ClassificationLoss(logit_adjust_counts=..., logit_adjust_tau=...)`:
  CE over `logits + tau*log(prior)`; raw logits unchanged for argmax/metrics.
- `model.mhcac.logit_adjustment` in `mimic_cxr_full.yaml` (`tau: 1.0`, train
  counts `[neg, pos, unc]` per finding, each row 220,379). `class_weights`
  removed from the YAML (kept as a comment); the model raises if both are set.
- Selection on `macro_recall` (D-024) is unchanged; the loss now targets the
  same quantity (balanced error).

## Comparator

`run_20260930_3class` (same code path, `class_weights`, selected on `loss`).
NOTE two variables move: the loss AND the selection metric (D-024). To isolate
the loss, also score the comparator's 1b epoch picked by macro_recall from its
`phase_metrics.jsonl` — but only 1b epoch 3/4 checkpoints exist there.

## Executor steps (host)

```bash
cd ~/Meta-CXR && git pull --ff-only
findmnt -no FSTYPE /mnt/drive1tb          # ntfs3
CUDA_VISIBLE_DEVICES="" ~/.venvs/meta-cxr-stage1-311/bin/python -m pytest tests/ training/ -q
# smoke: 2,000 studies, phase 1b only
ROOT=$HOME/smoke_20261001_logitadj; mkdir -p $ROOT
cp ~/run_20260925b_3phase/checkpoint_phase1a.pth $ROOT/
ROOT=$ROOT PHASES=phase1b EXTRA_OPTS="run.truncate_train=2000 run.truncate_val=400" \
    bash scripts/run_stage1_phases.sh
# full: ROOT=$HOME/run_20261001_logitadj PHASES="phase1b phase1c"
```

Abort: test failures beyond baseline, NaN/inf loss, OOM, smoke val
`macro_recall` not computed.

## Execution report

_(executor appends here)_
