# PLAN 2026-10-27 — aux-view tokens fed straight into MHCAC (candidate 2, "TK")

Status: **APPROVED 2026-10-27** ("ok": code change, step G + step P, reading rule). User chose this direction ("hướng 1 đi") as the last
multi-view try before writing the negative result (direction 3). Code implemented the same day;
CPU tests `tests/test_aux_token_fusion.py` (9) pass, and removing the key padding mask makes two of them
fail (checked). CPU suite: the same 21 baseline failures. Not yet run on GPU.

## Why

Every multi-view variant so far has hit the same trade at the pilot budget (multi-view-novelty skill,
"Current evidence"): the aux view helps two-view studies a little, inside the seed-noise bar, and it
costs about as much somewhere else.

- Early fusion (P15) mixes the aux view into the anchor tokens before projection. Every finding then
  sees the same mixed tokens.
- The late variants (LF, LD, AH) let MHCAC see one view at a time and average the logits afterwards.
  The pool stayed at about 0.5 per finding in all three, so no finding learned to prefer one view.

Neither design lets each finding decide, inside MHCAC, which view's tokens to read. That is what TK
does. MHCAC already has one expert token per finding, and each expert token cross-attends over the
image tokens. In TK the key/value sequence becomes the anchor tokens **plus** the aux view's tokens,
with a view-type embedding added to the aux tokens. Each finding's expert token can then put its
attention mass on the frontal, the lateral, or both. The anchor tokens are not modified. The Q-Former
still reads the anchor only.

## Design

New value: `model.view_fusion.mode: tokens`. The key's absence still means `early`. `late` is
unchanged. `tokens`:
- needs `multi_view: true`;
- builds no `view_fusion`, `stream_adapters`, `mpc_heads` or `late_view_pool`;
- refuses MPC, view consistency and the explanation loss, the same set `late` refuses, through
  `resolve_view_fusion_mode`;
- is refused together with `late_detach_aux` and `late_aux_head`.

The aux view is encoded as in late mode: frozen encoders under `no_grad`, the training feature mask
(`feature_mask_ratio` 0.1), and no stream adapter. It then goes through `ln_vision` (BioViL) and
**the same shared projector as the anchor, with gradient**. This is ordinary joint training: there is
one loss, CE on the study's prediction. No aux-view dropout is added. LF, LD and AH had none either;
the earlier attempt to raise `p_view_drop` failed its pilot, and single-view studies train the no-aux
path on their own.

Inside MHCAC (`mhcac/mhcac_12.py`):
- The aux tokens pass through the same `embedding_alignment` projection to 768 as the anchor.
- Each aux stream gets **its own** positional encoding, `aux_pos_enc[name]`. It has the same shape and
  init as `pos_enc[name]` (std 0.02). A lateral grid position is different anatomy from the same
  frontal grid position, so the anchor's encoding is not reused.
- A view-type embedding is added to every aux token: `aux_view_emb = nn.Embedding(4, 768)`, std 0.02,
  indexed by the aux view's id (the same `aux_view_ids` that `ViewFusionModule` reads).
- `image_patches = cat([anchor tokens, aux tokens], dim=1)`. A key padding mask marks the aux keys of
  every study that has no real aux view, so those keys get exactly zero attention weight. The mask is
  threaded through `ExpertTokenCrossAttention.forward` into `expert_to_image_attention`, and through
  `_self_then_image`. Every other mode passes `None`, which is the current call.
- A batch in which no study has a real aux view skips the concatenation and runs exactly today's code.
- The sparsity term normalises its entropy by the number of real (unmasked) attention cells, not by
  `numel`. Otherwise padded keys would shrink the term for single-view studies in mixed batches. With
  no aux in the batch the count equals `numel`, so the term is bit-identical.
- New parameters, created at the end of MHCAC's `__init__` and only in `tokens` mode (so the RNG order
  of every other mode is untouched): `aux_pos_enc` 296 × 768 = 227,328 and `aux_view_emb` 4 × 768 =
  3,072, **230,400 in total**. They sit under the `mhcac` prefix, so the 1b trainable and fade-in lists
  already cover them and need no command-line override. `prepare_phase_model` loads with
  `strict=False`, so a 1a checkpoint without them loads, and they keep their init. MHCAC itself starts
  from its untrained init in 1b anyway: phase 1a never runs it.

What this guarantees, by construction:
- **A single-view study takes OFF's computation**: same modules, same tokens, aux keys masked to
  exactly zero weight. Floating-point results can still differ in the last bits when the batch also
  holds two-view studies, because the key axis is longer. G3 measures this.
- The anchor tokens, and therefore the Q-Former's input, are untouched.

What it does not guarantee:
- That the shared MHCAC, trained on joint frontal+lateral inputs, stays as good on single-view inputs.
  LF lost single-view wF1 (−0.0024*) for a related reason, and early fusion lost single-view quality on
  test. Rule 3 below is what catches this.
- That the expert tokens learn to use the lateral at all in about 30,000 two-view studies over 2
  epochs. The probe reports the attention mass each finding puts on aux tokens, as a description.

## Code change (opt-in, absent key = bit-identical)

- `mhcac/late_view_pool.py`: `VIEW_FUSION_MODES` gains `"tokens"`; `resolve_view_fusion_mode` applies
  late's refusals to it.
- `mhcac/mhcac_12.py`: the optional aux arguments to `AbnormalityClassificationModel.forward`
  (`aux_visual_tokens`, `aux_mask`, `aux_view_ids`), the two new parameter groups, the key padding mask
  through both layer paths, and the sparsity normalisation above.
- `model/lavis/models/blip2_models/blip2_qformer.py`: `__init__` / `from_config` accept the mode and
  the refusals; `_encode_image_streams` stashes the aux streams as in late mode; one helper builds the
  aux shared tokens (real rows only, scattered back) and passes them to every student `self.mhcac(...)`
  call that classifies the study: `forward` and `forward_image`. The teacher and view-consistency calls
  are unreachable in this mode (both are refused).
- `scripts/evaluate_explanation.py` and `scripts/diagnose_stream_scale.py` refuse `tokens`, as they
  refuse `late`.
- The YAML is **not** edited, because of the Stage-2 mtime trap. Set the mode with
  `run.phases.phase1b.model.view_fusion.mode=tokens`.

CPU tests in a new `tests/test_aux_token_fusion.py`:
- Mode resolution: absent → early; `tokens` without `multi_view` refused; refused with MPC, view
  consistency, explanation, `late_detach_aux` and `late_aux_head`.
- On a tiny MHCAC in fp32: a study without aux gives the same logits (atol 1e-6) whether it is batched
  alone, batched with aux studies, or run with no aux arguments at all; and its attention weights on
  aux keys are exactly 0.
- A study with aux changes its logits; gradient reaches `aux_pos_enc`, `aux_view_emb` and the shared
  projector from the aux side.
- With no aux in the batch, the forward pass equals the current code bit for bit, including the
  sparsity term.
- Other modes create no `aux_pos_enc` / `aux_view_emb`, and their parameter list and state-dict keys are
  unchanged.

The same commit updates the struct pages for `mhcac_12.py`, `blip2_qformer.py` and `late_view_pool.py`,
plus `CLAUDE.md`, the Vietnamese `README` and the multi-view-novelty skill.

## Step G — equivalence and smoke (~8 min GPU), abort on any failure

Smoke size as in LF / LD / AH: 2,000 train and 400 val studies, 1 epoch, seed 42.
- **G1:** an `early` smoke (P15) and a `late` smoke on the new code. All checkpoint tensors and the
  val probabilities must equal the smoke arms already on disk: `~/latefusion_pilot_20261026/G1_old`
  (early, code `a43a88d`) and `~/lateauxhead_20261027/G1_late` (late). OFF (`multi_view: false`)
  never reaches the new branches; the CPU bit-identity test covers it.
- **G2:** a `tokens` smoke. It must meet all of the following:
  - rc=0, no NaN;
  - the log shows `"mode": "tokens"`;
  - after training, `aux_pos_enc` and `aux_view_emb` differ from their init;
  - the checkpoint holds `mhcac.aux_*`;
  - evaluate-only reproduces the training-time val npz.
  It also records the parameter count, max mem and s/it.
- **G3:** score G2's checkpoint with the aux view and with `model.data.max_aux_views=0`.
  - Single-view studies: max |Δprob| ≤ 1e-4 and no argmax change. The exact values are reported. A
    failure here means the mask leaks, and it aborts.
  - Two-view studies must move.

## Step P — pilot (~1.7 h GPU), test is NOT scored

The recipe is the one every pilot has used since 2026-10-25:
- phase 1b only;
- the first 60,000 train studies;
- 2 epochs, warm-up 150;
- starting from `run_20261005_paper/checkpoint_phase1a.pth`;
- `test_splits=[]`.

Runs are strictly serial, one card, into one new directory: `~/auxtokens_20261027/`.

| arm | overrides | seed |
|---|---|---|
| TK_s42 | `multi_view=true` + `view_fusion.mode=tokens` in the 1b phase block, everything else as LF | 42 |
| TK_s17 | same | 17 |

Comparators already on disk, same recipe and studies:
- Deciding: OFF_s42 (`~/auxdrop_pilot_20261025/OFF`) and OFF_s17 (`~/seednoise_20261026/OFF_s17`).
- Descriptive: P15, LF, LD and AH at both seeds.

Each arm should take about 45–50 min: the MHCAC key axis doubles for two-view studies, and there is no
second MHCAC. Scoring with and without the aux view adds about 5 min.

Guards, abort on any failure:
- Before launch: the GPU is idle, the data drive is mounted with `ntfs3`, no other job is running
  (positional `ps` check, never `pgrep -f`), and the output directory does not exist.
- Per arm:
  - rc=0, with no Traceback, OOM or NaN;
  - the logged seed and `"mode": "tokens"`;
  - `val_predictions_epoch_best.npz` exists and has `num_views`;
  - no test predictions are written.

## Reading rule — fixed before any result (identical to LF, LD and AH)

Method:
- Cutpoints are cross-fitted. Rows are sorted by `sample_key`, and the fold is
  `default_rng(16).permutation(n) % 2`.
- Groups come from `num_views`: 988 single-view and 820 two-view val studies.
- Δ_s = TK_s − OFF_s for each seed, and Δ̄ is the mean over s in {42, 17}.
- The CI of Δ̄ is a paired bootstrap with 1,000 draws, seed 16.

| NB/√2 | wF1 | F1-5 | macro recall |
|---|---|---|---|
| all val | 0.0019 | 0.0022 | 0.0078 |
| views_1 | 0.0032 | 0.0048 | 0.0081 |

**TK advances to a full-run proposal only if all three hold:**
1. **Gain:** for all-val wF1 or for all-val F1-5, all four of these hold:
   - Δ42 > 0;
   - Δ17 > 0;
   - the CI lower bound of Δ̄ > 0;
   - Δ̄ > NB/√2.
2. **No loss elsewhere:** on all val, none of wF1, F1-5 or macro recall has Δ̄ < −NB/√2.
3. **Single-view protected:** on views_1, neither the wF1 Δ̄ nor the F1-5 Δ̄ is below −NB/√2.

Otherwise **stop** and record the result.

Descriptive only, never deciding:
- views_2;
- AUROC;
- TK − P15, TK − LF and TK − AH;
- TK scored without the aux view;
- the mean last-layer attention mass each finding's expert token puts on aux tokens, on two-view val
  studies (does any finding learn to read the lateral, e.g. Pleural Effusion?).

⚠ This is the **fourth** variant read against the same two OFF arms (LF, LD, AH, TK). Four tries at one
bar raise the chance of a lucky pass, and that must be stated beside any pass. NB rests on two draws per
arm.

## Honest odds

The gain must come from the 820 two-view val studies. To clear all-val wF1 Δ̄ > 0.0019 with single-view
studies at OFF's level, two-view studies need about +0.0042 wF1. The best two-view gain on record at
this budget is LF's +0.0021, and its CI crossed 0. TK is the first design in which each finding can
choose its view inside MHCAC, and it adds no second head (so no ensemble objection). But early fusion
also fed the aux view into one shared MHCAC and came out at about OFF. A pass is possible, not likely.

**If TK stops, the plan is direction 3:** write the negative result honestly, as agreed with the user.

## If it passes (not in this plan, each its own decision)

- Seed replicates, then a full run (1b for 5 epochs plus 1c), then a single test reading.
- Any change to the shipped YAML.

## Cost

- Code and CPU tests: about 1–2 h of planner work, plus a host test run.
- GPU: step G about 8 min; step P about 1 h 40 min plus scoring; about 2 h in total, one card, serial.
- Disk: two pilot checkpoint sets, a few GB, on `/home`.

## Data handling

The `.npz` files, checkpoints and logs stay on the host. Only aggregates come back here.
