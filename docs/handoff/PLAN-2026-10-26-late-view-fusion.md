# PLAN 2026-10-26 — late fusion: one MHCAC per view, logits pooled over views

Status: **PROPOSED, NOT APPROVED, NOT RUN.** The user said "ok" (2026-10-26) to writing this plan.
No code, YAML or GPU change is made until the user approves it (multi-view-novelty skill, Rule 1).
Direction 3 of the skill's candidate list.

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

- **G1:** `smoke_P15` at `a43a88d` (the comparators' code; no Stage-1 file has changed since) and the
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
