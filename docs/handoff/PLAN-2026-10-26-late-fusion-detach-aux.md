# PLAN 2026-10-26 — late fusion with the aux-view pass detached from MHCAC

Status: **DONE 2026-10-27 -- VERDICT STOP: LD does not beat OFF** (rule 1 fails, rules 2/3 pass; see
the execution report). Approved by the user 2026-10-26 ("ok" to the code change, step G + step P and
the reading rule). Follows the late-fusion pilot
(STOP, `PLAN-2026-10-26-late-view-fusion.md`).

Implementation note: CPU suite = the same 21 failures as the baseline (missing nltk / torchvision /
env config on the dev box), 3 new tests passing. The `rows` index computation moved above the aux pass
(pure indexing, no RNG); G1 checks that the default path is still bit-identical.

## Why

The late-fusion pilot (LF) ended at zero over all val because two small effects cancelled. Two-seed
mean, LF - OFF:
- single-view studies: wF1 -0.0024* (within NB/√2);
- two-view studies: wF1 +0.0021, weighted recall +0.0036*.

A single-view study's output in LF is exactly its anchor logits, so the single-view loss is not caused
by the aux view at inference. It comes from training. MHCAC's weights are shared, and in LF the
aux-view pass (mostly laterals) back-propagates into MHCAC, the shared projector and the text tower.
MHCAC is therefore partly fitted to laterals, which the anchor path (mostly frontal) then pays for.

Change: run the aux-view pass with no gradient. MHCAC, the projector and the text tower then learn only
from the anchor image. The aux view still enters the prediction through `late_view_pool`, and the pool
(112 params) still trains.

What this changes and what it does not:
- MHCAC no longer sees aux images in its backward pass. Its forward on the aux view is unchanged: same
  weights, dropout still on in training, and the same RNG draws as LF. So at the same seed, LD and LF
  differ only in the gradient.
- The pooled CE still sends gradient to the ANCHOR logits of two-view studies (factor = the anchor's
  pool weight, ~0.5). MHCAC's objective is therefore still not exactly OFF's. Removing that too
  (stop-grad on both inputs of the pool) is a second, separate change and is not in this plan.
- Single-view studies at inference: still exactly the anchor logits (G3 re-checks it).
- Risk: an MHCAC that never trains on laterals may classify them worse, so the two-view gain may
  shrink. The pilot measures that; nothing guarantees it.

## Code change (opt-in, absent key = bit-identical to LF)

New key `model.view_fusion.late_detach_aux: true | false`. Absent = `false` = the LF code path as run
at `9d39156`.
- `Blip2Qformer.__init__`: read and store it. Refuse it outside `late` mode, because it would silently
  do nothing there. Early fusion has its own `detach_aux`, which is a different thing.
- `Blip2Qformer._late_view_logits`: when true, run the aux part (ln_vision / shared projector /
  encoder ablation / MHCAC) under `torch.no_grad()`. Training also asserts that the aux logits do not
  require grad. The pool call stays outside, so `late_view_pool` keeps its gradient.
- `from_config`: pass the key through.
- At eval everything already runs without grad, so LF checkpoints score the same either way.

**No YAML edit.** The key is set from the command line in the phase block. Editing
`mimic_cxr_full.yaml` would change its mtime and so Stage-2 caches (CLAUDE.md, mtime trap).

Tests (`tests/test_late_view_fusion.py`, CPU):
- the key is false by default;
- outside `late` it is refused;
- source check: the aux MHCAC call sits under the no-grad context when the flag is set, and the pool
  call does not.

Docs in the same commit: struct (`blip2_qformer` pages, `late_view_pool.py.doc.md`), CLAUDE.md,
README (Vietnamese), the skill.

## Step G — equivalence and smoke (~6 min GPU), abort on failure

Same smoke size as the LF plan: 2,000 train / 400 val studies, 1 epoch, seed 42.
- **G1:** a late smoke with the key absent at the new commit vs the LF code (`9d39156`). Every
  checkpoint tensor must be equal and the val probabilities equal. This shows the LF arms stay valid
  comparators.
- **G2:** a late smoke with `late_detach_aux=true`. rc=0, no NaN, the key logged, and the no-grad
  assertion is never hit. `late_view_pool` must have moved from zero (it still trains), and the
  checkpoint must differ from G1's (the change is live).
- **G3:** G2's checkpoint scored with and without the aux view. The single-view studies must be
  bit-identical.

## Step P — pilot (~1.6 h GPU), test is NOT scored

The same recipe as every pilot since 2026-10-25: phase 1b only, the first 60,000 train studies,
2 epochs, warm-up 150, from `run_20261005_paper/checkpoint_phase1a.pth`, `test_splits=[]`. Strictly
serial, one card, one new directory `~/latefusion_detach_20261026/`.

| arm | overrides | seed |
|---|---|---|
| LD_s42 | LF's overrides + `phase1b.model.view_fusion.late_detach_aux=true` | 42 |
| LD_s17 | same | 17 |

Comparators already on disk, same recipe and studies:
- OFF_s42 `~/auxdrop_pilot_20261025/OFF`, OFF_s17 `~/seednoise_20261026/OFF_s17` (deciding);
- LF_s42 / LF_s17 `~/latefusion_pilot_20261026/` and P15 at both seeds (descriptive).

Expected ~40-50 min per arm (LF took 50-53 min; the aux pass without backward is cheaper), plus ~5 min
of evaluate-only scoring with and without the aux view.

Guards (abort on any):
- before launch: GPU idle, `ntfs3`, no other job, the output dir absent;
- each arm: rc=0, no Traceback / OOM / NaN, logged seed as intended, `"mode": "late"` and
  `"late_detach_aux": true` in the logged config, `val_predictions_epoch_best.npz` with `num_views`;
- no test predictions.

## Reading rule — fixed before any result (identical to the LF plan)

Cross-fitted cutpoints: rows sorted by `sample_key`, fold = `default_rng(16).permutation(n) % 2`.
Groups from `num_views` (988 / 820). Δ_s = LD_s - OFF_s, Δ̄ = mean over s in {42, 17}, paired bootstrap
CI of Δ̄ (1,000 draws, seed 16). Bars NB/√2 from the pilot seed-noise run:

| NB/√2 | wF1 | F1-5 | macro recall |
|---|---|---|---|
| all val | 0.0019 | 0.0022 | 0.0078 |
| views_1 | 0.0032 | 0.0048 | 0.0081 |

**LD advances to a full-run proposal only if all three hold:**
1. **Gain:** for all-val wF1 OR F1-5: Δ42 > 0, Δ17 > 0, the CI lower bound of Δ̄ > 0, AND Δ̄ > NB/√2.
2. **No loss elsewhere:** on all val, none of wF1 / F1-5 / macro recall has Δ̄ < -NB/√2.
3. **Single-view protected:** on views_1, Δ̄ of wF1 and of F1-5 is not below -NB/√2.

Otherwise **stop** and record.

Descriptive only, never deciding:
- views_2;
- AUROC;
- LD - LF (did the detach remove the single-view loss? did it cost the two-view gain?);
- LD - P15;
- LD scored without the aux view;
- the mean aux pool weight per finding.

This is the second late-fusion variant read against the same OFF arms. Say so beside any pass: two
tries at one bar slightly raise the chance of a lucky pass. NB rests on two draws per arm.

## Not in this plan (each its own decision)

- Stop-grad on the anchor input of the pool as well (MHCAC's objective = exactly OFF's).
- A separate, higher LR for `late_view_pool`.
- A full run (1b 5 epochs + 1c) and any test reading. Even a pilot pass comes back to the user first,
  with seed replicates in the proposal.
- Changing the shipped YAML.

## Data handling

`.npz`, checkpoints and logs stay on the host. Only aggregates come back here.

## Execution report

The probe (`~/ldetach_probe.py`) was dry-run first on CPU with LD := LF and LDn := LF without aux. The
sanity check reproduced both earlier probes exactly (max diff 0.0). "LD" - OFF came out at LF - OFF to
the last digit, CIs included, and LD - LF was 0.

Launched once 22:39 (host code `a0e1eb0`, `~/ldetach.sh` -> `~/latefusion_detach_20261026/`, log
`~/ldetach.log`; LF code for G1 in a fresh clone `~/Meta-CXR-9d39156`).

**Step G passed 22:44 (smokes: 2,000 train / 400 val studies, 1 epoch, seed 42).**
- G1: late mode with the key absent, new code vs LF code `9d39156`: 724 of 724 checkpoint tensors
  equal (max abs diff 0), val probabilities equal. So LF_s42 / LF_s17 stay valid comparators.
  Both 51-54 s, max mem 5,219 MiB.
- G2: `late_detach_aux=true` smoke rc=0, key logged, the no-grad assertion was never hit.
  - The pool trained (|w| max 3.9e-3).
  - 191 MHCAC tensors differ from G1, so the change is live; same key set.
  - Max mem 4,907 MiB, ~300 MiB below LF (no backward through the aux pass).
- G3: the 237 single-view studies are bit-identical with and without the aux view; all 163 two-view
  studies move (no argmax cell changed at smoke size). Evaluate-only reproduces the training-time npz.

**Step P ran 22:44 -> 00:36 (`LDETACH_DONE`), code `a0e1eb0`, one card, no other job.**
- Guards: both arms rc=0, no Traceback / OOM / NaN, logged seeds 42 / 17, `"mode": "late"` and
  `"late_detach_aux": true` in the logged config, no test `.npz`. LD_s42 47m52s, best epoch 1;
  LD_s17 47m32s, best epoch 0 (LF: 53m / 50m, same best epochs). Max mem 4,940 / 4,935 MiB (the log
  line reads "49402026" -- a block-buffered timestamp glued on, as in the LF log).
- Evaluate-only reproduces each arm's training-time val npz. G3 on both pilot models: the 988
  single-view val studies are bit-identical with and without the aux view; all 820 two-view studies
  move, argmax cells changed 4.02% / 4.09%.
- Probe sanity: the OFF / P15 / LF values reproduce the earlier probes exactly (max diff 0.0).

**All val, n=1,808, LD - OFF** (cross-fitted cutpoints, paired bootstrap 1,000 draws):

| | Δ42 | Δ17 | Δ̄ [CI] | NB/√2 |
|---|---|---|---|---|
| weighted F1 | -0.0006 | +0.0008 | +0.0001 [-0.0017, +0.0018] | 0.0019 |
| F1-5 | -0.0004 | +0.0017 | +0.0007 [-0.0024, +0.0037] | 0.0022 |
| macro recall | -0.0021 | +0.0023 | +0.0001 [-0.0032, +0.0034] | 0.0077 |
| weighted P | -0.0053* | -0.0005 | -0.0029 [-0.0053, +0.0001] | 0.0029 |
| AUROC mean | +0.0028 | +0.0064* | +0.0046* [+0.0003, +0.0093] | 0.0195 |

**Rule 1 fails** (Δ42 < 0 on both wF1 and F1-5; neither CI excludes 0; both Δ̄ below NB/√2).
Rule 2 passes, rule 3 passes. **VERDICT: STOP, no full run.** AUROC +0.0046* is a quarter of its noise
bar and not deciding.

By view count (descriptive):

| Δ̄, two-seed mean | views_1 (988) | views_2 (820) |
|---|---|---|
| LD - OFF wF1 | -0.0001 [-0.0020, +0.0018] | -0.0009 [-0.0038, +0.0022] |
| LD - OFF F1-5 | +0.0016 [-0.0023, +0.0054] | -0.0012 [-0.0062, +0.0037] |
| LD - OFF macro recall | +0.0038* | -0.0105* [-0.0201, -0.0021] |
| LD - OFF AUROC | -0.0013 | +0.0134* [+0.0040, +0.0224] |
| **LD - LF wF1** | **+0.0023*** [+0.0007, +0.0042] | **-0.0029*** [-0.0047, -0.0012] |
| **LD - LF F1-5** | **+0.0037*** [+0.0001, +0.0075] | -0.0027 [-0.0062, +0.0002] |
| LD - LF macro recall | +0.0027 | -0.0073* |
| LD - LD scored without aux, wF1 | (identical logits) | -0.0005 [-0.0032, +0.0023] |
| LD - LD without aux, macro recall | -- | -0.0114* |

Other descriptive items, all val: LD - LF wF1 +0.0001, F1-5 +0.0011, weighted P -0.0055*; LD - P15
wF1 +0.0012, F1-5 +0.0022, both crossing 0. LD seed delta (17 - 42) wF1 +0.0021*, macro recall
+0.0071* -- the same size as the effects being read, which is why NB exists. Mean aux pool weight per
finding 0.472-0.502 (s42) and 0.474-0.501 (s17): the pool again stayed at plain averaging.

**Reading.** The detach did what it was designed to do and the trade simply moved:
- It removed LF's single-view loss: on single-view studies LD - LF wF1 +0.0023*, F1-5 +0.0037*, and
  LD - OFF there is flat (wF1 -0.0001). So the hypothesis in "Why" holds: LF's single-view cost came
  from MHCAC also training on laterals.
- It also removed LF's two-view gain: on two-view studies LD - LF wF1 -0.0029*, macro recall -0.0073*,
  and the aux view now adds nothing to wF1 there and lowers macro recall (LD - LD without aux wF1
  -0.0005, macro recall -0.0114*). An MHCAC that never trains on laterals reads them worse, which is the risk
  the plan named.
- Net over all val: zero, as for LF. With one shared MHCAC the aux view helps two-view studies only
  when MHCAC trains on it, and that training costs the anchor path by about the same amount.

Limits: two seeds per arm, pilot budget (60,000 studies, 2 epochs, phase 1b only), val only; NB rests
on two draws per arm; this is the second late-fusion variant read against the same OFF arms. Test was
not scored. Nothing here says what a full run would do.
