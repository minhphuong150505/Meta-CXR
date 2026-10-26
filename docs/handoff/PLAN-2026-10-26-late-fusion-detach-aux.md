# PLAN 2026-10-26 — late fusion with the aux-view pass detached from MHCAC

Status: **APPROVED by the user 2026-10-26 ("ok" to all three: the code change, step G + step P
(~1h45m GPU), and the reading rule). Code implemented; GPU not yet run.** Follows the late-fusion pilot
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
