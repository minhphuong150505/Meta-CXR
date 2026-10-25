# PLAN 2026-10-25 — raise the auxiliary-view drop rate (PROPOSED, NOT APPROVED, NOT RUN)

Status: **proposal. No YAML or code change, no GPU launch.** Every step below needs
the user's explicit go-ahead (`.claude/skills/multi-view-novelty/SKILL.md`, Rule 1).

## Why

Step 0c (`PLAN-2026-10-25-multi-view-step0.md`) showed that fusion ON
(`run_20261005_paper`) beats OFF (`run_20261024_nomv`) on two-view test studies
(F1-5 +0.0115 [+0.0051, +0.0175] under per-view-count cutpoints). It loses on
single-view studies (F1-5 -0.0050 [-0.0100, +0.0002]), and cutpoints fitted per
view count do not remove that loss. So the single-view weakness is in the model
trained with fusion, not in the decision rule.

Hypothesis: during training MHCAC sees too few anchor-only inputs. It learns an
operating regime tuned to fused features and is weaker when no aux view exists.
Raising the training-time aux drop makes anchor-only inputs common while fusion
still trains on the rest.

## What the knob is (read from the code, 2026-10-25)

- `model.view_fusion.p_view_drop`. The YAML ships **0.15**; 0.2 is only the code
  default (`blip2_qformer.py`, `view_fusion_cfg`). It is train-only:
  `ViewFusionModule.forward` masks each real aux view with probability `p` under
  `self.training` (`mhcac/view_fusion.py:126`). Eval is untouched, so test inputs do
  not change.
- Share of anchor-only training inputs: 44.7% of train studies have no aux view, so
  at p=0.15 about 53% of inputs are anchor-only. At p=0.3 it is ~61%, at p=0.5 ~72%.
  OFF is 100%.
- ⚠ **The drop is drawn independently in each encoder's module** (one
  `ViewFusionModule` per encoder, each with its own `torch.rand`). At p=0.5, 25% of
  two-view studies lose the aux view in both BioViL and PubMedCLIP, 25% keep it in
  both, and 50% keep it in only one stream. This plan does **not** change that,
  because it would be a code change. A study-level shared drop is a separate option
  for the user to decide on (see "Not in this plan").
- Phase blocks beat `--options`, so the override goes in the phase:
  `run.phases.phase1b.model.view_fusion.p_view_drop=<p>` (and `phase1c` for the
  full run). Nested phase overrides already worked for `model.loss.*` in the OFF run.

## Step P — pilot on val (~4 h GPU, test is NOT scored)

Same pilot recipe as 2026-10-11: phase 1b only, the first 60,000 train studies,
2 epochs, warm-up 150 updates, init from `run_20261005_paper/checkpoint_phase1a.pth`,
`run.phases.phase1b.run.test_splits=[]`. Strictly serial, one card, each arm in its
own new directory under `~/auxdrop_pilot_20261025/` (`test ! -e` before every arm).

| arm | change |
|---|---|
| P15 | none (shipped 0.15), the control at the pilot budget |
| P30 | `p_view_drop=0.3` |
| P50 | `p_view_drop=0.5` |
| OFF | `multi_view=false`, `lambda_mpc=0`, `lambda_view_consistency=0`, as in `run_20261024_nomv` |

Cost: ~48 min per arm (2026-10-11 pilot), OFF somewhat less, plus a 20-minute smoke
of P50 first (2,000 studies). **Total ~3.5–4 h.**

Guards (abort on any):
- the GPU guard used by `nomv_chain.sh` (ntfs3, no other training process, <1 GB VRAM);
- the resolved config in each arm's log shows the intended `p_view_drop`; if the
  value cannot be found in the log, stop before the full arms;
- OFF's log has no "multi-view fusion enabled" line;
- no Traceback / OOM / NaN in any log; each arm writes
  `val_predictions_epoch_best.npz` carrying `num_views`.

### Pilot reading rule — fixed now, before any result

Per arm, take the best epoch by `macro_recall` (the phase's own selection). On
val (1,808 studies), fit the headline two-cutpoint rule **cross-fitted**: split val
into 2 halves by `sample_key` (seed 16), fit on one half, score the other, then swap.
In-sample cutpoints on val would flatter every arm. Groups come from the ON arms'
`num_views` (one aux view vs none). Paired bootstrap, 1,000 draws, seed 16.

A candidate p ∈ {0.3, 0.5} **advances to the full run** only if both hold:
1. all-val weighted F1, candidate − OFF, beats P15 − OFF by at least +0.002
   (point estimate);
2. single-view F1-5, candidate − OFF, is higher than P15 − OFF (point estimate),
   **and** two-view F1-5, candidate − OFF, stays > 0.

If both pass, pick the higher all-val wF1. If neither passes: **stop, no full run**,
record the result. Val n=1,808 cannot separate differences of this size from noise.
The rule only screens out what clearly does not help; the decision is made on the
full run.

## Step F — full run of the chosen p (~12 h GPU, needs a second confirmation)

`run_20261025_auxdrop<p>`: same recipe as both existing arms, which is 1b 5 epochs
plus 1c from `run_20261005_paper/checkpoint_phase1a.pth`, with `p_view_drop=<p>` in
**both** phase1b and phase1c, and `run.phases.phase1b.run.test_splits=[]`. Expect
~12 h (A: 1b 7h08m + 1c 4h40m). Then fit cutpoints on its val with
`calibrate_thresholds.py --rule cutpoints` and score test **once**.

### Full-run reading rule — fixed now

Test n=3,269, the headline rule (one set of val cutpoints per arm), paired bootstrap
of 1,000 draws, seed 16, against OFF `run_20261024_nomv` (same budget). This is the
same rule as step 0c. **ON wins** iff the CI of F1-5 OR weighted F1 is above 0, AND
none of wP / wR / wF1 / F1-5 / macro recall has a CI below 0. Reported beside it,
descriptive only:
- the views_1 and views_2 subgroups;
- the comparison vs the current ON (`run_20261005_paper`);
- argmax numbers and AUROC.

A CI bound within 0.0005 of zero is reported as "at the zero boundary", not as a win.
One run per arm: even a win needs seed replicates (to ask separately) before it
becomes a paper claim. If C does not win, report it and keep `run_20261005_paper`.

## Not in this plan (each would be its own decision)

- Study-level shared drop across encoders (code change in `view_fusion` /
  `blip2_qformer`, plus tests).
- A separate LR for `view_fusion`, or per-finding gated fusion (directions 1 and 4).
- Changing the shipped YAML. Even if C wins, adoption is a separate confirmation.

## Data handling

Predictions, npz, logs and checkpoints stay on the host. Only aggregate metrics come
back into this file.

## Execution report

### Step P — user go-ahead 2026-10-25, launched once 15:52 (code `a43a88d`)

`~/auxdrop_pilot.sh` → `~/auxdrop_pilot_20261025/`, log `~/auxdrop_pilot.log`;
reading by `~/auxdrop_probe.py` (cross-fitted val cutpoints, rule above). The probe was
dry-run first on four copies of one val npz: every delta 0, verdict "stop", as it must be.
Smoke (P50, 2,000 studies, 1 epoch) passed 15:53: resolved `p_view_drop` 0.5 in the
logged config, max mem 6,388 MiB. P15 started 15:53.

All four arms rc=0, no error signature, every resolved `p_view_drop` as intended, OFF built
no fusion. P15 15:53 -> 16:40, P30 -> 17:43 (data loader slower, 0.50 vs 0.39 s/it; not the
FUSE stall: no D-state wait on ntfs, drive `ntfs3`), P50 -> 18:27, OFF -> 19:03, probe 19:08.
Max mem ~6.4 GB ON, 4.8 GB OFF. (The chain printed "max mem: 63832026": the date stuck to
the number in the log grep; the value is 6,383 MiB.)

### Step P result — val n=1,808 (988 one-view / 820 two-view), cross-fitted cutpoints

Values (wP / wR / wF1 / F1-5 / macro recall):

| arm | all | views_1 | views_2 |
|---|---|---|---|
| P15 | 0.8503 / 0.8647 / 0.8535 / 0.7993 / 0.4555 | 0.8128 / 0.8291 / 0.8148 / 0.7370 / 0.4370 | 0.8979 / 0.9085 / 0.8992 / 0.8756 / 0.4535 |
| P30 | 0.8458 / 0.8659 / 0.8534 / 0.8011 / 0.4541 | 0.8079 / 0.8274 / 0.8119 / 0.7347 / 0.4312 | 0.8976 / 0.9132 / 0.9009 / 0.8814 / 0.4527 |
| P50 | 0.8455 / 0.8623 / 0.8499 / 0.7931 / 0.4457 | 0.8072 / 0.8249 / 0.8092 / 0.7284 / 0.4252 | 0.8967 / 0.9082 / 0.8968 / 0.8724 / 0.4446 |
| OFF | 0.8507 / 0.8650 / 0.8534 / 0.7994 / 0.4504 | 0.8155 / 0.8293 / 0.8155 / 0.7358 / 0.4333 | 0.8953 / 0.9088 / 0.8982 / 0.8763 / 0.4470 |

Paired bootstrap (1,000, seed 16), * = CI excludes 0:

| | all wF1 | all F1-5 | views_1 wF1 | views_1 F1-5 | views_2 wF1 | views_2 F1-5 |
|---|---|---|---|---|---|---|
| P15 - OFF | +0.0002 | -0.0000 | -0.0007 | +0.0012 | +0.0010 | -0.0007 |
| P30 - OFF | +0.0000 | +0.0017 | -0.0037* | -0.0012 | +0.0027 | +0.0052 [+0.0000, +0.0108]* |
| P50 - OFF | -0.0034* | -0.0062* | -0.0063* | -0.0074 | -0.0013 | -0.0039 |
| P30 - P15 | -0.0002 | +0.0017 | -0.0029* | -0.0023 | +0.0017 | +0.0058 [+0.0009, +0.0107]* |
| P50 - P15 | -0.0036* | -0.0062* | -0.0056* | -0.0086* | -0.0023 | -0.0032 |

Rule: P30 fails conditions 1 (wF1 gain over P15 -0.0002, needs >= +0.002) and 2
(views_1 F1-5 -0.0012 vs OFF, worse than P15's +0.0012); passes 3. P50 fails all three.
**VERDICT: no candidate passes -> step F NOT launched** (`~/auxdrop_full.sh` read the probe
at 19:09 and exited with `NO_CANDIDATE`, as pre-declared). The user had chosen step F
("cách 2") conditional on this rule.

**Reading.** The hypothesis is refuted at this budget: more anchor-only training makes
single-view studies WORSE, not better (P50 - P15 views_1 wF1 -0.0056*, macro recall
-0.0118*; P30 -0.0029*), and P50 is worse everywhere. P30 helps two-view studies (F1-5
+0.0058* over P15) at the cost of single-view ones, so the total does not move. At the pilot
budget P15 and OFF are tied on single-view studies too (wF1 -0.0007), i.e. the full-run
single-view deficit of step 0c is not reproduced here. Why higher drop hurts is not
measured; one candidate is that the drop is drawn per encoder, so at p=0.5 half of the
two-view studies train with one stream fused and the other not -- an input that never
occurs at eval. Limits: 1b only, 60,000 studies, 2 epochs, one seed per arm, val n=1,808.
