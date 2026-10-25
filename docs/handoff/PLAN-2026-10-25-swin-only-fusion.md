# PLAN 2026-10-25 — train with fusion in the Swin stream only (val pilot)

Status: **user go-ahead 2026-10-25 ("ok")** for the Swin-only arm. The optional
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
Same recipe, and the code is identical: `git diff a43a88d HEAD` touches no `.py`,
`.yaml` or `.sh`. The chain checks this again before it launches.

## Guards (abort on any)

- GPU idle, `ntfs3`, no other job, `~/swinonly_pilot_20261025` absent, no code diff
  since `a43a88d`.
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
