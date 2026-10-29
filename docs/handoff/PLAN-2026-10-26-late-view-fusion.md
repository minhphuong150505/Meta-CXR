# PLAN 2026-10-26 — late fusion: one MHCAC per view, logits pooled over views

Status: **DONE 2026-10-26 -- VERDICT STOP: late fusion does NOT beat OFF at the pilot budget (rule 1
fails; rules 2 and 3 pass). No full run.** Approved by the user 2026-10-26 ("ok" to all three: the
code change, step G + step P (~2.5 h GPU), and the reading rule below). Direction 3 of the
multi-view-novelty skill's candidate list. Results: "Execution report" at the end.

Implementation note: besides the planned refusals, `late` also refuses the explanation loss
(`lambda_explanation*` > 0, which Grad-CAMs a single token sequence; off in production).
`scripts/evaluate_explanation.py` and `scripts/diagnose_stream_scale.py` read MHCAC directly
and stop with `SystemExit` in `late`. CPU suite: 1,224 passed (baseline 1,213 + 11 new), the
same 21 failures as the baseline (missing nltk / torchvision / env config on the dev box).

## Why

Every early-fusion result so far points at one weakness. Fusion ON is weaker than OFF on
single-view studies (step 0b test F1-5 -0.0084; step 0c showed per-view-count cutpoints do not fix
it, so it is the model). Raising the aux drop made single-view studies worse, not better. Swin-only
fusion lost on single-view studies too, where no fusion block runs. At the pilot budget ON - OFF is
~0 over two seeds (F1-5 -0.0015).

In early fusion the aux view changes the anchor's tokens before MHCAC, and every study, single-view
included, goes through the StreamAdapters and the MPC objective. Late fusion keeps the anchor path
the same architecture as OFF. MHCAC classifies each view on its own (shared weights), and a small
per-finding attention pools the per-view logits.

What this guarantees and what it does not:
- **Guaranteed by construction:** for a study with one view, the output IS that view's MHCAC
  logits. Same modules as OFF: no adapter, no fusion block, no MPC.
- **Not guaranteed:** MHCAC's weights are trained on lateral views as well, so single-view
  accuracy can still move either way. The reading rule checks it (rule 3 below). An earlier message
  said "single-view studies cannot be worse than OFF". That was too strong and is corrected here.

## Code change (opt-in, absent key = bit-identical)

New key `model.view_fusion.mode: early | late`; absent = `early` = every recorded run.

`late` requires `model.multi_view: true`, so the dataset loads the aux view. It REFUSES
`lambda_mpc > 0` and `lambda_view_consistency > 0`, both of which read early-fusion pre-fusion
streams. It builds:
- no `view_fusion`, no `stream_adapters`, no `mpc_heads`;
- `late_view_pool = LateViewPooling(num_findings=14, num_classes=3, num_view_types=4)`, a new module
  in `mhcac/late_view_pool.py`.

Flow in `late` mode (`Blip2Qformer`):
1. Anchor: encoders -> `_mask_stream` -> shared projector -> MHCAC, exactly the OFF path. The
   Q-Former reads these anchor tokens, as in OFF.
2. Aux view, real rows only (`real_aux_rows`): frozen encoders under `no_grad` (the existing
   `_encode_aux_streams`), then `mask_aux_tokens`, `ln_vision` on BioViL, the shared projector, and
   MHCAC with the SAME weights. Training passes the same report text and row mask as the anchor
   pass. Only its logits are used; its contrastive / orth / sparsity terms are not added
   (`labels=None`), so those terms stay as in OFF.
3. `LateViewPooling`: per view v and finding f, score
   `s[v,f] = w_f . [log_softmax(z[v,f]) (3), onehot(view type of v) (4)] + b_f`. This is a softmax
   over the study's REAL views (padded slots masked out); pooled `z[f] = sum_v a[v,f] z[v,f]`.
   Zero-init (`w = b = 0`) means the mean of the real views at step 0. One real view gives
   `a = 1`, so the output is exactly the anchor logits. 14 x 8 = 112 parameters.
4. Loss: `L_cls = 0.5 CE(pooled) + 0.5 CE(anchor)` (`model.view_fusion.late_anchor_weight: 0.5`,
   fixed, not tuned). A single-view study then gets exactly OFF's loss, CE(anchor), and the anchor
   path gets full supervision on every study.
5. `classification_logits` = pooled, in `forward` (training, val/test `.npz`) and in
   `forward_image` (evaluate-only, Stage-2 records). Any other MHCAC call site found by grep in
   `late` mode raises instead of silently using anchor-only logits.

YAML: add `late_view_pool` to the phase1b / phase1c `trainable` prefix lists. A prefix the model
does not build matches nothing (as `mpc_heads` is today in OFF), so existing runs are unchanged. The
shipped `mode` stays absent (early).

Tests, `tests/test_late_view_fusion.py` (CPU, no GPU stack):
- a single-view study pools to its anchor logits bit for bit;
- junk in padded aux slots does not change the output;
- zero-init gives the mean of the real views;
- the pooling weights sum to 1 over real views;
- refusals: late with `multi_view: false`, `lambda_mpc > 0`, `lambda_view_consistency > 0`;
- the loss of a single-view study equals CE(anchor);
- `early` / absent builds no `late_view_pool`, and its state-dict keys are unchanged.

Docs in the same commit: `struct/` (blip2_qformer, mhcac/late_view_pool, view_fusion), CLAUDE.md,
README (Vietnamese), the skill.

## Step G — equivalence and smoke (~15 min GPU), abort on failure

- **G1:** `smoke_P15` at `761f30b` (the comparators' code; no Stage-1 file has changed since) and the
  same smoke at the new commit (early mode, absent key). 2,000 studies, 1 epoch, seed 42. Every
  tensor of the 1b checkpoint must be EQUAL. The comparators below then stay valid.
- **G2:** late smoke, same size. rc=0, no NaN. The log must show no fusion block, no adapter, and
  `late_view_pool` trainable. The val `.npz` carries `num_views`.
- **G3:** a CPU check on G2's checkpoint. Re-score val with `model.data.max_aux_views=0`: the
  single-view studies' logits must equal G2's bit for bit (the guarantee above, on the real model).

## Step P — pilot (~2 h GPU), test is NOT scored

The recipe of every pilot since 2026-10-25: phase 1b only, the first 60,000 train studies,
2 epochs, warm-up 150, from `run_20261005_paper/checkpoint_phase1a.pth`, `test_splits=[]`. Strictly
serial, one card, one new directory `~/latefusion_pilot_20261026/`.

| arm | overrides | seed |
|---|---|---|
| LF_s42 | `phase1b.model.view_fusion.mode=late`, `lambda_mpc=0`, `lambda_view_consistency=0` | 42 |
| LF_s17 | same | 17 |

Comparators (already on disk, same recipe, same 60,000 studies):
- OFF_s42 `~/auxdrop_pilot_20261025/OFF`, OFF_s17 `~/seednoise_20261026/OFF_s17`;
- P15 (early fusion) at both seeds, descriptive only.

Expected ~50-60 min per LF arm (P15 51 min, OFF 36 min; MHCAC runs again on the ~55% of studies
that have an aux view).

Guards (abort on any): GPU idle, `ntfs3`, no other job, the output dir absent; each arm rc=0, no
Traceback / OOM / NaN, logged seed as intended, `mode: late` in the logged config; each writes
`val_predictions_epoch_best.npz` with `num_views`; no test predictions.

## Reading rule — fixed before any result

The procedure of `PLAN-2026-10-26-pilot-seed-noise.md`:
- cross-fitted cutpoints (rows sorted by `sample_key`, fold = `default_rng(16).permutation(n) % 2`);
- groups from `num_views` (val n=1,808: 988 one-view / 820 two-view);
- metrics weighted P / R / F1, F1-5, macro recall, `auroc_mean`.

Δ_s = LF_s - OFF_s, s in {42, 17}. Δ̄ = (Δ42 + Δ17)/2, with a paired bootstrap CI of Δ̄ (1,000
draws, seed 16; both seeds' deltas on the same resampled studies). The bar for Δ̄ is NB/√2, NB from
the pilot seed-noise run:

| NB/√2 | wF1 | F1-5 | macro recall |
|---|---|---|---|
| all val | 0.0019 | 0.0022 | 0.0078 |
| views_1 | 0.0032 | 0.0048 | 0.0081 |

**LF advances to a full-run proposal only if all three hold:**
1. **Gain (CI rule + NB rule):** for all-val wF1 OR F1-5, Δ42 > 0, Δ17 > 0, the CI of Δ̄ has its
   lower bound > 0, AND Δ̄ > NB/√2.
2. **No loss elsewhere:** on all val, none of wF1 / F1-5 / macro recall has Δ̄ < -NB/√2.
3. **Single-view protected:** views_1 Δ̄ of wF1 and of F1-5 is not below -NB/√2.

Otherwise **stop**: no full run, record the result.

Descriptive only, never deciding:
- the views_2 subgroup;
- AUROC (its NB is 0.0276, larger than any effect seen so far);
- LF - P15;
- LF scored with `max_aux_views=0` (inference only, ~3 min per seed: how much the aux view moves
  two-view studies);
- the mean pooling weight of the aux view per finding, on val two-view studies.

NB rests on two draws per arm (true SD ~0.5x-6x the estimate). Say so beside every verdict.

## Not in this plan (each its own decision)

- **Full run** (1b 5 epochs + 1c, ~12 h) and its test reading. No NB exists at full budget, so even
  a pilot pass comes back to the user first, with seed replicates in the proposal.
- Stage 2 on a late-fusion Stage-1.
- A view-type embedding inside MHCAC (it would change the anchor path), an aux-view drop for late
  fusion, or MPC on top of late fusion.
- Changing the shipped YAML default.

## Data handling

`.npz`, checkpoints and logs stay on the host. Only aggregates come back here.

## Execution report

Launched once 16:52 (host code `b914a92`, `~/latefusion.sh` -> `~/latefusion_pilot_20261026/`, log
`~/latefusion.log`; old code for G1 in a fresh clone `~/Meta-CXR-a43a88d`). The probe
(`~/latefusion_probe.py`) was dry-run first on CPU with LF := P15 and LFn := OFF: the sanity check
reproduced the seed-noise probe exactly (max diff 0.0), "LF" - OFF came out at the seed-noise run's
ON - OFF means (wF1 -0.0011, F1-5 -0.0015), and the verdict was STOP, as it must be.

**Step G passed 16:57 (all smokes: 2,000 train / 400 val studies, 1 epoch, seed 42).**
- G1: old code (`761f30b`) vs new code in early mode: 809 of 809 checkpoint tensors equal (max abs diff
  0), val probabilities equal. So the comparators trained at `761f30b` stay valid. Both 50-57 s,
  max mem 6,388 MiB.
- G2: late smoke rc=0, late module built, no early fusion block, `"mode": "late"` logged; max mem
  5,219 MiB.
- G3: the G2 checkpoint scored with and without the aux view (evaluate-only, val 400): the 237
  single-view studies are bit-identical (max abs diff 0); all 163 two-view studies move (0.22% of
  argmax cells change). The evaluate-only pass reproduces the training-time val npz exactly.

**Step P finished 18:55 (`LATEFUSION_DONE`).**
- Guards: both arms rc=0, 0 Traceback / OOM, logged seeds 42 / 17, `"mode": "late"` logged,
  `late_view_pool` in the trainable list, no test predictions written. LF_s42 53m20s (best epoch 1),
  LF_s17 49m43s (best epoch 0); max mem ~5.3 GB both (the `5301`/`5283` in `latefusion.log` carry a
  log timestamp glued on by buffering). Evaluate-only reproduces each arm's training-time val npz.
- The G3 guarantee holds on the real pilot models: with and without the aux view, the 988 single-view
  val studies are bit-identical at both seeds; the 820 two-view studies all move (4.3% / 4.5% of
  argmax cells change).
- Probe sanity: OFF / P15 values equal the seed-noise probe's (max diff 0.0).

Val n=1,808, cross-fitted cutpoints, paired bootstrap of the two-seed mean (1,000 draws, seed 16).
`*` = CI excludes 0.

| all val | Δ42 | Δ17 | Δ̄ [CI] | NB/√2 |
|---|---|---|---|---|
| weighted F1 | +0.0007 | -0.0007 | **+0.0000** [-0.0017, +0.0017] | 0.0019 |
| F1-5 | +0.0000 | -0.0010 | **-0.0005** [-0.0034, +0.0024] | 0.0022 |
| macro recall | +0.0027 | -0.0006 | +0.0011 [-0.0022, +0.0044] | 0.0077 |
| weighted P | +0.0039* | +0.0013 | +0.0026 [-0.0005, +0.0049] | 0.0029 |
| AUROC mean | +0.0005 | -0.0006 | -0.0000 [-0.0055, +0.0061] | 0.0195 |

| views_1 (n=988) Δ̄ | wF1 | F1-5 | macro recall | AUROC |
|---|---|---|---|---|
| LF - OFF | -0.0024* [-0.0045, -0.0005] | -0.0022 [-0.0060, +0.0014] | +0.0011 | -0.0059* |
| NB/√2 | 0.0032 | 0.0048 | 0.0081 | 0.0163 |

| views_2 (n=820) Δ̄, descriptive | wF1 | F1-5 | macro recall | weighted R | AUROC |
|---|---|---|---|---|---|
| LF - OFF | +0.0021 [-0.0007, +0.0051] | +0.0015 [-0.0032, +0.0061] | -0.0033 | +0.0036* | +0.0065 |
| LF - LF without aux | +0.0021 [-0.0009, +0.0051] | +0.0004 | -0.0066 | +0.0039* | +0.0027 |

**Reading:**
- Rule 1 FAILS: all-val wF1 Δ̄ +0.0000 and F1-5 -0.0005, and the two seeds disagree in sign.
- Rule 2 passes: no all-val Δ̄ below -NB/√2.
- Rule 3 passes: views_1 wF1 -0.0024 and F1-5 -0.0022 are above -NB/√2 (0.0032 / 0.0048).
- **VERDICT: STOP, no full run.**

**What the descriptive items say:**
- On single-view studies LF is a little worse than OFF (wF1 -0.0024, CI excludes 0 but within
  NB/√2; seed 17 alone -0.0038*). Their logits are the anchor's exactly, so this is the shared MHCAC
  being trained on lateral views too, plus cutpoints shared with the two-view studies. That is the risk
  the plan named ("not guaranteed").
- On two-view studies the aux view helps a little: weighted recall +0.0036 to +0.0039, CI excludes 0.
  wF1 +0.0021 sits at the views_2 NB/√2 (0.0021) with a CI crossing 0, and macro recall goes the other
  way (-0.0033 / -0.0066).
- Net over all val: zero.
- LF vs P15 (early fusion): all-val wF1 +0.0011, F1-5 +0.0011 (CIs cross 0); weighted P +0.0047*.
  Late is at least as good as early here, and neither beats OFF.
- The pooling attention barely trained. The mean aux-view weight per finding on two-view val studies
  is 0.480-0.502 at both seeds, close to the zero-init mean of 0.5. So LF in practice averaged the two
  views' logits. With 112 parameters at the 1b LR over 2 epochs, the pool had no strong signal to move
  away from the mean.
- AUROC seed-to-seed for LF itself: -0.0123*, again showing the single-seed val bootstrap does not
  cover seed noise.

**Limits:**
- NB rests on two draws per arm, so the true SD could be ~0.5x-6x the estimate.
- Pilot budget only: 60,000 studies, 2 epochs, phase 1b only.
- Val only; test was never scored.

