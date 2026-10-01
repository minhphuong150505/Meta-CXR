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

### 2026-10-01 — supplementary: test scored with val Eq. 22 thresholds

`evaluate_stage1.py --thresholds ~/eval_20260930_3class/thresholds_eq22_val.json`
(code `30a101c`; host tests for the touched files pass, sklearn pins included).
NOT the paper's protocol. Test n=3,269, 95% bootstrap CI (1,000):

| metric | argmax (paper) | Eq. 22 thresholds |
|---|---:|---:|
| weighted_precision | 0.8379 [0.8344, 0.8452] | 0.8278 [0.8236, 0.8325] |
| weighted_recall | 0.7685 [0.7641, 0.7728] | 0.6383 [0.6320, 0.6442] |
| weighted_f1 | 0.7658 [0.7612, 0.7705] | 0.6966 [0.6910, 0.7023] |
| mean_weighted_f1_5 | 0.6438 [0.6352, 0.6525] | 0.6547 [0.6457, 0.6637] |
| macro_recall | 0.4560 [0.4513, 0.4688] | 0.5027 [0.4914, 0.5206] |

Pooled over 14 findings, recall N/P/U 0.802/0.747/0.046 -> 0.637/0.728/0.254;
precision 0.924/0.409/0.170 -> 0.929/0.338/0.080; Uncertain predicted 411 ->
4,810 times for 1,516 true cells. Rare positives recovered (Fracture 0.01 ->
0.53, Pleural Other 0.05 -> 0.86, Enlarged Cardiomediastinum 0.04 -> 0.36),
Cardiomegaly negative recall 0.21 -> 0.50. Weighted F1 falls because Negative
recall falls. Uncertain precision 0.08 confirms AUROC_unc 0.57: thresholds can
buy Uncertain recall only with mostly-wrong Uncertain calls.

### 2026-10-01 — is Uncertain visible in the image? (analysis, no training)

Score s = p_pos/(p_pos+p_neg) from run_20260930_3class; AUROC between label
groups, 95% CI from 1,000 bootstrap resamples per group; findings with >= 30
Uncertain and Positive cells. Test split:

| finding | nN / nU / nP | s: U>N | s: P>U | s: P>N | p_unc: U>N | p_unc: U>P | median s N / U / P |
|---|---|---|---|---|---|---|---|
| Enl. Cardiomediastinum | 2893 / 204 / 151 | 0.59 [0.54,0.63] | 0.56 [0.50,0.63] | 0.65 | 0.60 | 0.51 | 0.27 / 0.30 / 0.31 |
| Cardiomegaly | 2267 / 115 / 866 | 0.60 [0.56,0.65] | 0.67 [0.61,0.73] | 0.75 | 0.44 | 0.68 | 0.77 / 0.83 / 0.92 |
| Lung Opacity | 2131 / 79 / 1038 | 0.63 [0.57,0.69] | 0.61 [0.55,0.67] | 0.73 | 0.46 | 0.60 | 0.46 / 0.60 / 0.68 |
| Edema | 2276 / 289 / 683 | 0.77 [0.74,0.79] | 0.69 [0.65,0.72] | 0.87 | 0.68 | 0.65 | 0.27 / 0.64 / 0.87 |
| Consolidation | 2931 / 97 / 220 | 0.64 [0.59,0.69] | 0.66 [0.60,0.72] | 0.76 | 0.44 | 0.61 | 0.29 / 0.40 / 0.52 |
| Pneumonia | 2563 / 350 / 335 | 0.73 [0.70,0.76] | 0.53 [0.48,0.57] | 0.74 | 0.70 | 0.57 | 0.32 / 0.50 / 0.52 |
| Atelectasis | 2326 / 191 / 731 | 0.73 [0.70,0.77] | 0.58 [0.53,0.62] | 0.78 | 0.60 | 0.61 | 0.39 / 0.70 / 0.76 |
| Pleural Effusion | 2044 / 130 / 1074 | 0.78 [0.75,0.82] | 0.74 [0.70,0.78] | 0.90 | 0.45 | 0.74 | 0.25 / 0.76 / 0.92 |

Val shows the same pattern. Reading: on every finding the median s orders
N < U < P and U>N excludes 0.5, so Uncertain cases ARE visibly different from
negatives — as an intermediate on the positive/negative axis, not a distinct
appearance. Pneumonia U vs P is indistinguishable (0.53 [0.48,0.57]): uncertain
pneumonia looks like positive pneumonia, consistent with the report hedging on
cause rather than appearance. The dedicated Uncertain output does not capture
this: p_unc ranks Uncertain BELOW negatives for Cardiomegaly, Lung Opacity,
Consolidation and Pleural Effusion (0.44-0.46). A middle class is structurally
hard for argmax and for one-vs-rest AUROC. Caveats: labels are labeler
output (noisy), and part of the signal may come from co-occurring findings.

### 2026-10-01 — ordinal decision rule: two cutpoints on s (analysis, no training)

Per finding, cutpoints t1 <= t2 on s = p_pos/(p_pos+p_neg) fitted on val
(121 quantile grid): N below t1, U in [t1, t2), P at or above t2. Scored on test
(n=3,269), paired bootstrap vs argmax (1,000, seed 16). NOT the paper's protocol.

| rule | wF1 | F1_5 | macro_recall | U predicted (1,516 true) | U recall / precision |
|---|---:|---:|---:|---:|---|
| argmax (paper) | 0.7658 | 0.6438 | 0.4560 | 411 | 0.046 / 0.170 |
| 1 cutpoint (no U), fit wF1 | 0.8143 | 0.7440 | 0.4320 | 0 | 0 / - |
| 2 cutpoints, fit wF1 | 0.8151 | 0.7425 | 0.4372 | 579 | 0.062 / 0.162 |
| 2 cutpoints, fit macro_recall | 0.6275 | 0.5342 | 0.4826 | 13,296 | 0.538 / 0.061 |

2 cutpoints (wF1) - argmax: wF1 +0.0493 [+0.0459, +0.0524], F1_5 +0.0987
[+0.0916, +0.1055], macro_recall -0.0188 [-0.0237, -0.0143].
1 cutpoint - argmax: wF1 +0.0485 [+0.0451, +0.0519], F1_5 +0.1003 [+0.0934, +0.1074].

Reading: the Uncertain band adds almost nothing (+0.0008 wF1 over one
cutpoint, U precision 0.16). The whole gain is moving the Positive/Negative
cutpoint, i.e. undoing the over-calling that the class weights put into argmax.
The intermediate image signal for Uncertain is real but overlaps both
neighbours too much to be called as a class. Consistent with the earlier
simulation that plain (unweighted) CE gives wF1 0.805 / F1_5 0.730 at argmax.

### 2026-10-01 — user chose B: no retrain; argmax headline, cutpoints supplementary

Reproduced with the committed script (`8809995`,
`scripts/evaluate_stage1_cutpoints.py`, host tests pass): point estimates
identical to the ad-hoc run above, CIs within 0.0004 (different bootstrap
stream). Report: `~/eval_20260930_3class/cutpoints/cutpoints_report.json`.
