# PLAN 2026-10-26 — seed noise at the pilot budget (direction 5)

Status: **DONE 2026-10-26 01:58 -- noise bar measured (report below).** User go-ahead 2026-10-26 ("ok chạy đo nhiễu seed đi"). No code or YAML
change; test is not scored. ~1.5 h GPU.

## Why

Four val pilots in a row were decided on differences of 0.005–0.01, each with one
run per arm: aux-drop P30 / P50, and Swin-only SW, each against P15 and OFF. The
paired bootstrap over val studies measures sampling noise only, not seed noise.
P15 − OFF on single-view studies was AUROC −0.0123, on inputs where fusion never runs.
That suggests run-to-run noise of that order. Without a noise bar, no pilot verdict can
be called real.

## Arms

Same recipe as the aux-drop pilot, rerun with `run.seed=17` instead of 42:

- phase 1b only, the first 60,000 train studies, 2 epochs, warm-up 150;
- from `run_20261005_paper/checkpoint_phase1a.pth`, `test_splits=[]`.

`run.seed` exists only in the run block (no phase sets it), so `--options` reaches it.
Truncation keeps the first N studies whatever the seed, so both seeds train on the
same 60,000 studies. What changes with the seed: MHCAC's random init (1a does not
train it), batch order, dropout, augmentation, and the aux-drop draws.

| arm | change vs the 42-seed run |
|---|---|
| P15_s17 | `run.seed=17` |
| OFF_s17 | `run.seed=17` + the OFF overrides of the aux-drop pilot |

The 42-seed runs are the existing `~/auxdrop_pilot_20261025/{P15,OFF}`.
`git diff a43a88d HEAD` touches no `.py`, `.yaml` or `.sh`; the chain checks this
again.

## Guards (abort on any)

- GPU idle, `ntfs3`, no other job, `~/seednoise_20261026` absent, no code diff since
  `a43a88d`.
- Smoke first: P15 with seed 17, 2,000 studies, 1 epoch. It must have rc=0 and
  exactly one `"seed": 17` in the logged config.
- Each full arm: rc=0, no Traceback / OOM / NaN, `"seed": 17` logged.
- P15_s17: `p_view_drop` 0.15. OFF_s17: no fusion built, plus the OFF dataset line.
- Each writes `val_predictions_epoch_best.npz`; test is never scored.
- The probe must reproduce the aux-drop probe's P15 and OFF values for seed 42
  exactly (same procedure, max diff < 1e-9).

## Reading rule — fixed before any result

Same procedure as every pilot:

- cross-fitted cutpoints (rows sorted by `sample_key`,
  fold = `default_rng(16).permutation(n) % 2`);
- groups from the ON arms' `num_views`;
- metrics weighted P / R / F1, F1-5, macro recall and `auroc_mean`.

For each metric and each group (all / views_1 / views_2):

- **seed deltas** d_ON = P15_s17 − P15_s42 and d_OFF = OFF_s17 − OFF_s42;
- **noise bar** NB = 2 × sqrt((d_ON² + d_OFF²) / 2). This is a rough 95% bar for the
  difference between two single-seed pilot arms. It is estimated from two draws only:
  a χ²(2) interval puts the true SD anywhere from ~0.5× to ~6× that estimate. State
  this beside every NB.
- **ON − OFF per seed** (P15 − OFF at seed 42 and at seed 17), with paired bootstrap
  CIs (1,000 draws, seed 16), and their mean.

Re-read, descriptive only, of the earlier pilot deltas (taken from their `probe.json`):
P30 − P15, P50 − P15, SW − P15 on wF1 / F1-5 / macro recall (and AUROC for SW).

- **beyond noise**: |Δ| > NB;
- **within noise**: otherwise.

No earlier verdict is overturned by this re-read, because every one of them was
"stop". Whether a future pilot must also beat NB to advance is a change to the
decision rule. It is proposed to the user after this result and is not adopted here.

## Data handling

`.npz`, checkpoints and logs stay on the host. Only aggregates come back here.

## Execution report

Launched once 00:24 (host code `eaf942a`); the probe was dry-run first on CPU (seed 42 fed in as "seed
17": NB 0, sanity diff 0.0).

- Smoke passed 00:25 (`"seed": 17` logged).
- P15_s17 trained 00:25 -> 01:17 (50m59s); OFF_s17 01:17 -> 01:53 (35m35s). Every guard passed, and
  test was not scored.
- Probe 01:58. **Sanity: the seed-42 values reproduce the aux-drop probe exactly (max diff 0.0).**

### Result — val n=1,808 (988 one-view / 820 two-view), cross-fitted cutpoints

Seed deltas (seed 17 − seed 42), paired bootstrap over val studies (1,000, seed 16), * = CI excludes 0:

| | all wF1 | all F1-5 | all macro recall | all AUROC | views_1 wF1 | views_2 AUROC |
|---|---|---|---|---|---|---|
| P15 (ON) | -0.0017 | -0.0012 | **-0.0073** * | **-0.0159** * | **-0.0032** * | **-0.0169** * |
| OFF | +0.0008 | +0.0018 | +0.0027 | **-0.0112** * | -0.0004 | **-0.0190** * |

**The same recipe at another seed moves AUROC by 0.011–0.019, and the val bootstrap calls that
"significant".** So the paired bootstrap over studies does not cover seed noise. A single-seed "*" on
AUROC or macro recall says nothing about the method.

Noise bar NB = 2 × sqrt((d_ON² + d_OFF²) / 2) (two draws only; true SD ~0.5x–6x this):

| NB | wP | wR | wF1 | F1-5 | macro recall | AUROC |
|---|---|---|---|---|---|---|
| all | 0.0041 | 0.0030 | 0.0027 | 0.0031 | 0.0110 | 0.0276 |
| views_1 | 0.0050 | 0.0047 | 0.0045 | 0.0068 | 0.0115 | 0.0230 |
| views_2 | 0.0082 | 0.0041 | 0.0029 | 0.0047 | 0.0071 | 0.0360 |

**ON − OFF at the pilot budget, per seed** (all val):

| | wF1 | F1-5 | macro recall | AUROC |
|---|---|---|---|---|
| seed 42 | +0.0002 | -0.0000 | +0.0052 * | +0.0009 |
| seed 17 | -0.0023 | -0.0031 | -0.0048 * | -0.0038 |
| mean of 2 | -0.0011 | -0.0015 | +0.0002 | -0.0015 |

On views_2 the two-seed mean is ~0 on every metric (wF1 +0.0001, F1-5 0.0000). **At this budget
fusion ON does not beat OFF, at either seed.**

Re-read of the earlier pilots, descriptive (|Δ| vs NB):
- **P50 − P15:** beyond noise, worse: all wF1 -0.0036 (NB 0.0027) and F1-5 -0.0062 (0.0031); views_1
  wF1 / F1-5 / macro recall.
- **SW − P15:** beyond noise, worse: F1-5 all -0.0071 (0.0031), views_1 -0.0088 (0.0068), views_2
  -0.0057 (0.0047); views_2 macro recall -0.0175 (0.0071). Its AUROC losses (-0.0149 / -0.0132) are
  within noise.
- **P30 − P15:** everything within noise except views_2 F1-5 +0.0058 (NB 0.0047), just beyond.

All three earlier "stop" verdicts stand. P50 and SW are worse beyond noise; P30 is noise apart from a
small two-view F1-5 gain.

**Consequences, to be decided by the user (not adopted here).**
1. A pilot should not advance on AUROC or macro recall unless the gain also exceeds NB. NB on AUROC
   (~0.03) is larger than any effect seen so far.
2. The full-run ON − OFF comparisons are one seed per arm too. Step 0c's single-cutpoint wF1 +0.0021
   (test) is below this pilot-budget wF1 NB (0.0027 on val). Full-budget noise is not measured.
