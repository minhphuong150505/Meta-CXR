# PLAN 2026-10-25 — train with fusion in the Swin stream only (val pilot)

Status: **DONE 2026-10-25 23:36 -- FAILED the pre-declared rule; SW is worse than P15 and OFF (report below). No full run.** User go-ahead 2026-10-25 ("ok") for the Swin-only arm. The optional
"Swin + BioViL" arm was offered and is NOT part of this go-ahead. No repo code or
YAML change; test is not scored. ~50 min GPU.

## Why

The per-stream ablation (`PLAN-2026-10-25-fusion-per-stream.md`) found that on
two-view val studies the whole operating-point gain of fusion comes from the Swin
stream. Turning Swin fusion off alone drops wF1 -0.0045* and F1-5 -0.0070*. BioViL
fusion adds AUROC only, and PubMedCLIP fusion adds nothing measurable. Hypothesis:
the BioViL / PubMedCLIP fusion blocks contribute little and still change the features
MHCAC is trained on. Training with fusion in the Swin stream only may keep the
two-view gain and lose less elsewhere.

## Arm

**SW**: the aux-drop pilot recipe, unchanged. That is phase 1b only, the first
60,000 train studies, 2 epochs, warm-up 150, from
`run_20261005_paper/checkpoint_phase1a.pth`, `test_splits=[]`, and the shipped
`p_view_drop` 0.15. The one difference: `Blip2Qformer._fuse` returns the anchor
tokens for biovil and pubmedclip (in-memory patch, `FUSION_SKIP=biovil,pubmedclip`)
in training and in val alike.

- The launch goes through `scripts/run_stage1_phases.sh` with `PY=~/fusion_py.sh`.
  That shim runs `~/pretraining.train_fusion_patch.py`, the same patch as the
  per-stream probe; the filename keeps `pretraining.train` visible to every GPU
  guard.
- The BioViL / PubMedCLIP fusion blocks then get no gradient
  (`zero_grad(set_to_none=True)`, so AdamW skips them).
- Their zero-init `w_o` and `ffn.3` stay exactly 0. Each block is therefore an exact
  identity, and the checkpoint is a Swin-only-fusion model even when loaded without
  the patch.
- MPC and view consistency stay on, as in P15.

**Controls are reused, not rerun**: P15 and OFF from `~/auxdrop_pilot_20261025/`.
Same recipe, and the code is identical: `git diff 761f30b HEAD` touches no `.py`,
`.yaml` or `.sh`. The chain checks this again before it launches.

## Guards (abort on any)

- GPU idle, `ntfs3`, no other job, `~/swinonly_pilot_20261025` absent, no code diff
  since `761f30b`.
- Smoke first (2,000 studies, 1 epoch, val 400). It must have rc=0, a
  `checkpoint_phase1b.pth`, and `FUSION_SKIP_FIRST` for biovil and pubmedclip. In its
  checkpoint, `w_o` and `ffn.3` of the biovil and pubmedclip fusion blocks must be
  present and all zero, and Swin's must be non-zero.
- The full arm repeats those checks, plus no Traceback / OOM / NaN, `p_view_drop`
  0.15 in the logged config, and a `val_predictions_epoch_best.npz` that carries
  `num_views`.

## Reading rule — fixed before any result (same as the aux-drop pilot)

Each arm's best epoch by `macro_recall`. Val n = 1,808. Cutpoints are cross-fitted
(rows sorted by `sample_key`, fold = `default_rng(16).permutation(n) % 2`, each arm
fits its own). Groups come from the ON arms' `num_views`. Paired bootstrap, 1,000
draws, seed 16. Metrics: weighted P / R / F1, F1-5, macro recall, and `auroc_mean`
(descriptive).

**SW advances to a full-run proposal** (separate confirmation, ~12 h) only if all
three hold, on point estimates:

1. all-val wF1: (SW - OFF) - (P15 - OFF) >= +0.002;
2. views_1 F1-5: SW - OFF > P15 - OFF;
3. views_2 F1-5: SW - OFF > 0.

Otherwise: stop, no full run, record it. SW - P15 with CIs on every group is reported
beside the rule, descriptive only. Val n = 1,808 cannot separate differences this
small from noise; the rule only screens out what clearly does not help.

## Data handling

`.npz`, checkpoints and logs stay on the host. Only aggregates come back here.

## Execution report

Launched once 22:43 (host code `c8417b6`). The CPU dry runs came first: the
checkpoint check FAILs on A's checkpoint, as it must since A's BioViL fusion was
trained, and the probe with SW = P15 gives "no candidate".

- Smoke passed 22:44: max mem 5,685 MiB; BioViL / PubMedCLIP `w_o` + `ffn.3` exactly 0,
  Swin's 0.0034.
- SW trained 22:44 -> 23:32 (47m35s): best epoch = index 1 (the last), max mem 5,682 MiB.
  The wrapper skipped BioViL and PubMedCLIP in all 7,818 batches that had an aux view,
  and fused Swin in all of them. Checkpoint check PASS: BioViL / PubMedCLIP exactly 0,
  Swin 0.041. Logged `p_view_drop` 0.15. Test not scored. Probe 23:36.
- (The chain's "max mem: 56822026" is the known log-grep artefact; the value is 5,682 MiB.)

### Result — val n=1,808 (988 one-view / 820 two-view), cross-fitted cutpoints

Values (wP / wR / wF1 / F1-5 / macro recall / AUROC mean):

| arm | all | views_1 | views_2 |
|---|---|---|---|
| P15 | 0.8503 / 0.8647 / 0.8535 / 0.7993 / 0.4555 / 0.7403 | 0.8128 / 0.8291 / 0.8148 / 0.7370 / 0.4370 / 0.6995 | 0.8979 / 0.9085 / 0.8992 / 0.8756 / 0.4535 / 0.7580 |
| SW | 0.8520 / 0.8655 / 0.8522 / 0.7922 / 0.4480 / 0.7253 | 0.8173 / 0.8279 / 0.8124 / 0.7282 / 0.4302 / 0.6862 | 0.8981 / 0.9116 / 0.8968 / 0.8699 / 0.4360 / 0.7503 |
| OFF | 0.8507 / 0.8650 / 0.8534 / 0.7994 / 0.4504 / 0.7394 | 0.8155 / 0.8293 / 0.8155 / 0.7358 / 0.4333 / 0.7117 | 0.8953 / 0.9088 / 0.8982 / 0.8763 / 0.4470 / 0.7542 |

Paired bootstrap (1,000, seed 16), * = CI excludes 0:

| | all wF1 | all F1-5 | all macro recall | all AUROC | views_1 F1-5 | views_2 F1-5 | views_2 macro recall |
|---|---|---|---|---|---|---|---|
| SW - P15 | -0.0013 | **-0.0071** * | **-0.0075** * | **-0.0149** * | **-0.0088** * | -0.0057 | **-0.0175** * |
| SW - OFF | -0.0011 | **-0.0072** * | -0.0024 | **-0.0140** * | -0.0077 | -0.0063 | -0.0109 |
| P15 - OFF | +0.0002 | -0.0000 | **+0.0052** * | +0.0009 | +0.0012 | -0.0007 | +0.0066 |

Rule: all three conditions fail (wF1 gain over P15 -0.0013, needs >= +0.002; views_1
F1-5 vs OFF -0.0077, worse than P15's +0.0012; views_2 F1-5 vs OFF -0.0063 < 0).
**VERDICT: no candidate passes -> no full run.**

**Reading.**
- Training with fusion in the Swin stream only is worse than all-stream fusion, and
  worse than no fusion, on F1-5 and AUROC. That holds even on single-view studies,
  where no fusion block runs at all (views_1 SW - P15 F1-5 -0.0088*, AUROC -0.0132*).
  So either removing the BioViL / PubMedCLIP fusion blocks changes how MHCAC trains
  (not only what the two-view input looks like), or run-to-run noise at this budget is
  this large (next bullet). This pilot cannot tell the two apart.
- The per-stream inference result (Swin carries the gain in a model trained with all
  three) does not become a better training recipe.
- **Seed noise at this budget is not measured.** P15 - OFF views_1 AUROC is
  -0.0123 [-0.0191, +0.0045] on inputs where fusion never runs, so run-to-run
  differences of ~0.01 AUROC are plausible. One seed per arm; the controls were
  trained earlier the same day with identical code. Do not read the SW deficit as
  more than "it does not help".
