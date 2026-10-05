# PLAN 2026-10-05 — Stage 1 retrain with the paper's MHCAC values (D-026)

## What changes

One run, compared against `run_20260930_3class`. Two settings differ, both
set to the paper's values (code `206fb0e`):

| | `run_20260930_3class` | this run | paper |
|---|---|---|---|
| `model.mhcac.num_common_tokens` | 14 | **8** | 8 (Fig. 10) |
| `model.mhcac.label_smoothing` | 0.05 | **0.0** | none (Eq. 10) |
| MHCAC dropout | 0.2 | 0.2 | 0.2 |

Everything else is identical, including the phase-1a checkpoint (same file,
with `mhcac.expert_tokens` removed: phase 1a does not train MHCAC, and the
`[14, 768]` tensor would otherwise raise `size mismatch` in
`prepare_phase_model`). Both settings move together, so their effects are not
separated.

## Commands (host)

```bash
cd ~/Meta-CXR && git pull                     # 206fb0e
findmnt -no FSTYPE /mnt/drive1tb              # ntfs3
ROOT=$HOME/run_20261005_paper
mkdir -p $ROOT
# copy of ~/run_20260930_3class/checkpoint_phase1a.pth minus mhcac.expert_tokens
ROOT=$ROOT PHASES="phase1b phase1c" setsid nohup bash scripts/run_stage1_phases.sh \
    > $ROOT.log 2>&1 < /dev/null &
setsid nohup bash ~/eval_after_paper.sh > ~/eval_after_paper.log 2>&1 &
```

`~/eval_after_paper.sh` waits for the schedule to exit, then scores test and
val with `scripts/evaluate_stage1.py` into `~/eval_20261005_paper/` and fits
Eq. 22 thresholds on val. Status in `~/eval_20261005_paper/STATUS`.

## Expected

~7h50m (1b) + ~4h45m (1c), as on the previous run. Then a paired bootstrap on
the same 3,269 test studies against `run_20260930_3class`: weighted P/R/F1,
`mean_weighted_f1_5`, `macro_recall`, AUROC per class.

## Abort conditions

OOM, NaN loss, `size mismatch`, or `checkpoint_phase1c.pth` missing at the end.

## Execution report

2026-10-05 14:38 launched (runner pid 20258). Log confirms
`num_common_tokens: 8`, `label_smoothing: 0.0`, phase 1b initialised from the
stripped 1a checkpoint (764 tensors). At iteration 850 of 13,922: 7,003 MiB,
GPU 100%, epoch ETA ~1h20m. Host tests for the touched files: 43 passed.
