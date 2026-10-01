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

### 2026-10-01 — host tests, smoke, and a simulation that stops the full run

- `8cc083c` + `4a1c968` on the host: full suite 1,131 tests, 0 failed, 2 skipped.
- Smoke 1 failed at phase-1b init: `checkpoint_phase1a.pth` carried 14 CE
  class-weight buffers the logit-adjusted model lacks. Fixed in `4a1c968`
  (loss configuration is no longer stored in or read from checkpoints).
- Smoke 2 (`~/smoke_20261001_logitadj`, 2,000 train / 400 val, 1b only):
  mechanically clean — loads 765 tensors from 1a, no NaN/inf, 0.33 s/it,
  `max mem` 6,481 MiB, `Saving best model at epoch 1 (val macro_recall 0.329605)`.
  Val after ~150 updates: weighted F1 0.08-0.17, macro_recall 0.30-0.33.
  Too small to judge, but consistent with the risk below.

**Post-hoc simulation (Menon Eq. 9) on run_20260930_3class predictions.** The
weighted-CE scores are un-weighted (log p - log w) and then shifted by
-tau*log(prior). An approximation of what training with the loss would give;
Menon notes the trained solution can differ for non-convex models.

| test, n=3,269 | wF1 | F1_5 | macro_recall | predicted Uncertain (1,516 true) |
|---|---:|---:|---:|---:|
| as trained (weighted CE, capped) | 0.7658 | 0.6438 | 0.4560 | 411 |
| tau 0 (= plain CE, no weights) | 0.8045 | 0.7301 | 0.4192 | 0 |
| tau 0.5, all classes | 0.7972 | 0.7049 | 0.4407 | 30 |
| tau 0.75, all classes | 0.6293 | 0.6700 | 0.4700 | 9,041 |
| **tau 1.0, all classes (committed config)** | **0.3202** | **0.4708** | **0.4340** | **24,902** |
| tau 1.0, Positive only (PromptMRG form) | 0.6104 | 0.6154 | 0.4659 | 0 |

Reading: tau 1 makes the model call Uncertain everywhere because Uncertain has
no image signal (AUROC 0.57) and a tiny prior, and it is worse than the current
model on every metric. **Do not launch the full run with tau 1.0.** No variant
improves the paper metrics and macro_recall together; the user must choose.
