# PLAN 2026-09-30 — Stage-1 retrain under D-023 (three classes only)

Planner: Claude (dev box). Requested by the user 2026-09-30: every earlier
Stage-1 number was produced or scored under a binary framing, so Stage 1 is
retrained on the D-023 code and scored with the paper's protocol.

## Decision: reuse phase 1a, retrain 1b + 1c

Phase 1a trains only the META-Former (Q-Former + shared projection) with
ITC/ITM/LM; `lambda_cls` is 0 and MHCAC is not run, so no CheXpert label
touches it. D-023 changed labels, losses and metrics only. The 1a checkpoint of
`run_20260925b_3phase` (D-022 code, gate passed at epoch 2: `delta_nats` +2.93,
R@5 0.664 / 0.641) is therefore copied into the new root and only 1b + 1c run.
Saves ~15.5 h plus the feature-cache build. `Blip2Qformer.load_state_dict`
passes through `drop_retired_state`, so any retired head in the old file is
dropped on load.

Checkpoint selection in 1b and 1c is `selection_metric: loss` on val, so no
binary metric picks a checkpoint.

## Commands (host)

```bash
cd ~/Meta-CXR && git fetch && git checkout feat/stage2-finding-tokens && git pull
findmnt -no FSTYPE /mnt/drive1tb                 # must be ntfs3
CUDA_VISIBLE_DEVICES="" ~/.venvs/meta-cxr-stage1-311/bin/python -m pytest tests/ -q
# expected: only the 4 test_native_independence failures (missing private config)

ROOT=$HOME/run_20260930_3class
mkdir -p $ROOT && cp ~/run_20260925b_3phase/checkpoint_phase1a.pth $ROOT/
ROOT=$ROOT PHASES="phase1b phase1c" setsid nohup bash scripts/run_stage1_phases.sh \
    > $ROOT.log 2>&1 < /dev/null &
```

Expected: ~7h53m (1b) + ~4h45m (1c), as measured on the D-022 run.

## Evaluation

```bash
python scripts/evaluate_stage1.py --predictions <phase1c test .npz> --output-dir ~/eval_20260930_3class
```

Paper references: weighted P/R/F1 over 14 findings 0.87 / 0.78 / 0.73;
`mean_weighted_f1_5` 0.701; one-vs-rest AUROC per class (Fig. 5).

## Abort conditions

- tests fail beyond the known 4;
- `/mnt/drive1tb` is not `ntfs3`;
- another GPU process is running;
- NaN/inf in any loss, OOM, or `checkpoint_phase1b.pth` missing after 1b.

## Execution report

_(executor appends here)_

### 2026-09-30 21:31 — launched

- Host on `f419e0c`; full suite `tests/ training/`: **1,106 tests, 0 failed,
  2 skipped**, rc 0 (after replacing the retired `evaluation:` block of the
  host's private `env_config.yaml`; backup `~/env_config.yaml.bak_20260930`).
- `/mnt/drive1tb` ntfs3, GPU idle (119 MiB) before launch.
- `checkpoint_phase1a.pth` copied from `run_20260925b_3phase`, sha256 prefix
  `e4683e642ad1bccd` identical on both.
- `ROOT=~/run_20260930_3class PHASES="phase1b phase1c"`, log `~/run_20260930_3class.log`.
- Phase 1b init: 765 tensors loaded from 1a, the rest are the frozen encoders.
- Iter 850/13,922: `loss_cls` 1.04, `loss_mpc` 1.66, ITC/ITM/LM 0 as intended,
  0.36 s/it, `max mem` 6,476 MiB, GPU 98%. Epoch ETA ~1h25m.

### 2026-09-30 22:22 — automatic scoring armed

`~/eval_after_3class.sh` (pid 14780, setsid) waits for pid 13476 to exit, checks
`checkpoint_phase1c.pth`, then runs `evaluate_stage1.py` on the phase-1c
`test`/`val_predictions_epoch_best.npz` into `~/eval_20260930_3class/{test,val}`
and `calibrate_thresholds.py` on val (Eq. 22). State in
`~/eval_20260930_3class/STATUS` (WAITING / SCORING / DONE / FAILED).

Evaluator dry run on the D-022 run (`run_20260925b_3phase`, which already
trained three-class; only its old scoring was binary), test n=3,269:
weighted P/R/F1 **0.8387 / 0.7653 / 0.7658** (paper 0.87 / 0.78 / 0.73),
`mean_weighted_f1_5` **0.6434** (paper 0.701), AUROC Pos/Neg/Unc
**0.7774 / 0.7708 / 0.5904**. Output `~/eval_20260925b_rescored/test`.

### 2026-10-01 — selection metric changed for FUTURE runs (D-024)

At the user's request, `mimic_cxr_full.yaml` phases 1b/1c now select on
`macro_recall` (three-class balanced recall, new aggregate). **This run is not
affected**: it was launched on `f419e0c` and selected 1b epoch 3 on `loss`.
`auroc_mean` would have chosen the same epoch; `weighted_f1` and
`mean_weighted_f1_5` would have chosen epoch 2 (+0.002 / +0.017 on val).
Do not `git pull` on the host until this run and its scorer have finished.

### 2026-10-01 09:23 — finished and scored

Phase 1c done 09:13 (`rc` clean, `checkpoint_phase1c.pth` present); scorer DONE
09:23. Test n=3,269, paired bootstrap (1,000 resamples, seed 16) against the
D-022 run `run_20260925b_3phase`, same studies, labels verified identical:

| metric | D-022 | this run | delta, 95% CI |
|---|---:|---:|:---|
| weighted_precision | 0.8387 | 0.8379 | -0.0008 [-0.0030, +0.0052] |
| weighted_recall (= accuracy) | 0.7653 | 0.7685 | +0.0032 [+0.0015, +0.0051] |
| weighted_f1 | 0.7658 | 0.7658 | +0.0000 [-0.0016, +0.0017] |
| mean_weighted_f1_5 | 0.6434 | 0.6438 | +0.0004 [-0.0030, +0.0041] |
| macro_recall | 0.4600 | 0.4560 | -0.0039 [-0.0070, -0.0007] |
| auroc_positive_mean | 0.7774 | 0.7755 | -0.0018 [-0.0049, +0.0014] |
| auroc_negative_mean | 0.7708 | 0.7691 | -0.0017 [-0.0049, +0.0014] |
| auroc_uncertain_mean | 0.5904 | 0.5737 | -0.0167 [-0.0524, +0.0308] |

Paper: 0.87 / 0.78 / 0.73, five-finding F1 0.701.

Reading: the two runs are the same model to within noise. Only weighted_recall
(+0.003) and macro_recall (-0.004) clear zero, in opposite directions and by
amounts too small to act on: the usual operating-point shift. Expected, since
D-022 already trained three-class and D-023 changed scoring, not training.
macro_recall 0.456 against a 0.333 chance level says the model rarely predicts
Uncertain and misses many Positives; the weighted numbers ride on the majority
Negative class. Artifacts: `~/eval_20260930_3class/{test,val,test_v2}`,
`thresholds_eq22_val.json`, `~/paired_3class.log`.
