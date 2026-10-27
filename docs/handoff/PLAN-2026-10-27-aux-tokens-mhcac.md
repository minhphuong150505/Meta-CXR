# PLAN 2026-10-27 — aux-view tokens fed straight into MHCAC (candidate 2, "TK")

Status: **APPROVED 2026-10-27** ("ok": code change, step G + step P, reading rule). **RAN 2026-10-27: ADVANCE by the rule, with caveats -- see Execution report.** User chose this direction ("hướng 1 đi") as the last
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

Neither design lets MHCAC itself decide, per token and per finding, which view to read. That is what
TK does. Since D-026, MHCAC has 8 shared expert tokens. Each one cross-attends over the image tokens,
and each finding then pools the 8 tokens with its own learned weights (`attention_pooling`). In TK the
image key/value sequence becomes the anchor tokens **plus** the aux view's tokens, with a view-type
embedding added to the aux tokens. An expert token can then put its attention mass on the frontal, the
lateral, or both, and a finding can lean on the tokens that read the view it needs. The anchor tokens
are not modified. The Q-Former still reads the anchor only.

(Corrected 2026-10-27 before any run: the first draft said "one expert token per finding". That
described the reference code's 14 tokens, not the D-026 configuration this run uses.)

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
  epochs. The probe reports the attention mass on aux tokens, as a description.

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
  - the logged phase-1b trainable count for `mhcac` is 80.78M (80.55M + 230,400), so the new
    parameters are in the optimizer and fade in with `mhcac`. (Changed before any run from "differ from
    their init": their init is drawn at model build and is in no checkpoint on disk, so it cannot be
    compared. That they receive gradient is pinned by the CPU test, and G3 shows the aux tokens are
    read.)
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
- the mean last-layer attention mass on aux tokens on two-view val studies: per expert token (8), and per
  finding as the finding's pooling weights times the tokens' aux mass (does any finding lean on the
  lateral, e.g. Pleural Effusion?).

⚠ This is the **fourth** variant read against the same two OFF arms (LF, LD, AH, TK). Four tries at one
bar raise the chance of a lucky pass, and that must be stated beside any pass. NB rests on two draws per
arm.

## Honest odds

The gain must come from the 820 two-view val studies. To clear all-val wF1 Δ̄ > 0.0019 with single-view
studies at OFF's level, two-view studies need about +0.0042 wF1. The best two-view gain on record at
this budget is LF's +0.0021, and its CI crossed 0. TK is the first design in which MHCAC's tokens can
choose between the views, and it adds no second head (so no ensemble objection). But early fusion
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

## Execution report

2026-10-27, host checkout `87351f9`, `~/auxtok.sh` → `~/auxtokens_20261027/`, launched three times. Launch 1
aborted before any GPU work: `/home` had 21 GB free against a 30 GB guard; the script now trims this run's own
checkpoints and the guard is 15 GB, and no user data was touched. Launch 2 aborted at the step-G eval: the attention
wrapper read `query_vectors` off `AbnormalitySpecificLoss` instead of its `attention_pooling`. The wrapper was fixed,
this run's output dir was removed, and nothing else changed. Launch 3 ran 15:13:50 → `AUXTOK_DONE` 17:17:47. The probe
`~/auxtok_probe.py` → `probe.{json,log}` reproduced every OFF / P15 / LF / LD / AH value of the earlier probes
exactly (max |Δ| 0.0).

**Step G — PASS.**
- G1 early: 809/809 tensors bit-identical to the earlier early smoke, val probabilities equal.
- G1 late: 724/724 tensors bit-identical to AH's G1_late.
- G2 (TK smoke): `mhcac` 80.78M trainable (80.55M + 230,400); `aux_pos_enc` 3 tensors (196 / 50 / 50 × 768),
  `aux_view_emb` (4, 768); max mem 5,088 MiB.
- G3: single-view studies (n=237) max |Δprob| 2.4e-7, 0 argmax changes; all 163 two-view studies moved;
  evaluate-only reproduces the training-time npz.

**Step P — two seeds.** Training 48m46s (s42) / 48m19s (s17), `mhcac` 80.78M trainable, peak about 5.1 GB.
The driver's `max mem` field came out glued to a timestamp ("50912026"); `nvidia-smi` read 5.6 GB in use during
s17. G3 was re-checked on both pilot models: views_1 n=988, max |Δprob| 2.4e-7, 0 argmax changes. On views_2
(n=820) every study moved, and 2.85% / 2.57% of argmax cells changed.

**Verdict by the pre-declared rule: ADVANCE** (rule 1 passes on F1-5, rule 2 passes, rule 3 passes).
Cutpoints are cross-fitted, the bootstrap is paired with 1,000 draws at seed 16, and `*` = CI excludes 0.

| TK − OFF, all val (n=1,808) | s42 | s17 | two-seed mean [CI] | NB/√2 |
|---|---|---|---|---|
| weighted F1 | +0.0022* | +0.0005 | +0.0014 [−0.0001, +0.0030] | 0.0019 |
| F1-5 | +0.0066* | +0.0006 | **+0.0036 [+0.0006, +0.0067]** | 0.0022 |
| macro recall | +0.0053* | +0.0003 | +0.0028 [−0.0002, +0.0065] | 0.0077 |
| AUROC mean | −0.0074* | +0.0077 | +0.0002 [−0.0049, +0.0056] | 0.0195 |

- Rule 1 on F1-5: Δ42 +0.0066 > 0, Δ17 +0.0006 > 0, CI lower bound +0.0006 > 0, mean 0.0036 > 0.0022. The wF1
  form fails (the CI crosses 0 and the mean is below NB/√2).
- Rule 3: views_1 TK − OFF wF1 +0.0012, F1-5 +0.0062*, both above −NB/√2.

⚠⚠ **Read the pass with four caveats. None of them changes the verdict, and all of them limit what it shows.**
1. **The gain is not on the two-view studies.** On views_2 (n=820), TK − OFF mean: wF1 +0.0010, F1-5 +0.0003,
   macro recall −0.0014, AUROC +0.0063, all crossing 0. It sits on views_1 (n=988): F1-5 +0.0062*, macro recall
   +0.0046*. There the aux keys are masked, and TK's probabilities equal TKnoaux's to 2.4e-7 (AUROC Δ exactly 0).
   So the views_1 gain combines two things:
   - a differently trained MHCAC: TKnoaux − OFF views_1 F1-5 +0.0029, CI crosses 0;
   - cutpoints fitted on all val: TK − TKnoaux views_1 F1-5 +0.0033*, and the probabilities are identical.

   The plan's "honest odds" said the gain must come from the two-view studies; it does not. This is the same
   pattern as the 2026-10-24 package ablation (gain equal on single-view studies, so not attributable to the
   second view).
2. **Seed 17 is nearly zero.** All the margin is seed 42's (F1-5 +0.0066* vs +0.0006). TK's own seed delta
   (17 − 42) on F1-5 is −0.0042*, larger than the two-seed gain.
3. **Fourth variant against the same two OFF arms** (LF, LD, AH, TK). Four tries at one bar make a lucky pass
   more likely, and NB rests on two draws per arm.
4. AUROC is flat (+0.0002), so this is an operating-point gain, not a ranking gain.

Descriptive only (cannot change the verdict):

| two-seed mean, Δ (all val) | wF1 | F1-5 | macro recall | AUROC |
|---|---|---|---|---|
| TK − P15 (early) | +0.0024* | +0.0051* | +0.0026 | +0.0016 |
| TK − LF | +0.0014 | +0.0040* | +0.0017 | +0.0002 |
| TK − LD | +0.0013 | +0.0029 | +0.0027 | −0.0045 |
| TK − AH | +0.0031* | +0.0032 | +0.0033 | −0.0018 |
| TK − TKnoaux | +0.0012 | +0.0018 | +0.0034* | +0.0032 |
| TKnoaux − OFF | +0.0002 | +0.0018 | −0.0006 | −0.0030 |

- On views_2, TK − TKnoaux (same model, aux view removed): wF1 +0.0021 (CI touches 0), weighted recall +0.0028*,
  F1-5 −0.0008, AUROC +0.0072. The aux tokens move two-view predictions a little; no deciding metric clears NB/√2.
- Attention on the aux tokens, last layer, the 820 two-view val studies. The aux keys are 296 of 592, so 0.5 is
  uniform.
  - The 8 expert tokens split into two groups at both seeds: some lean on the aux view (s42 four at 0.61–0.64;
    s17 three at 0.64–0.66), and the rest sit at 0.30–0.38.
  - Each finding's pooled aux share is 0.32–0.35 (s42) and 0.31–0.38 (s17). The findings pool mostly from the
    anchor-leaning tokens, and no finding leans on the lateral view.

Reading: TK is the first multi-view variant to pass the pre-declared pilot rule, but the pass rests on one seed and
on the single-view studies, where the aux view is not seen. The rule was met, and the evidence that the second view
causes the gain was not. A full run would need its own decision (plan "If it passes"), and seed replicates come
first. Pilot budget, two seeds, val only; test not scored.

## Seed replicate — approved 2026-10-27 ("a"), rule fixed before any new arm runs

Why: the pilot passed on one seed and on single-view studies. This step asks whether the pass survives two more
seeds and whether the two-view studies, the ones the aux view can actually change, gain beyond seed noise.

Arms: TK and OFF at `run.seed` 7 and 23 (seeds chosen now, before running). The recipe is the same as the pilot:
phase 1b only, 60,000 studies, 2 epochs, warmup 150, from `run_20261005_paper/checkpoint_phase1a.pth`; TK as in
step P, OFF as in `~/seednoise.sh` (`multi_view=false`, `lambda_mpc 0`, `lambda_view_consistency 0`). Code
`87351f9`. Run order: TK_s7, OFF_s7, TK_s23, OFF_s23, serial, on one card. Test is not scored. The existing
seed-42/17 arms are reused, which gives four seeds per arm.

**Gate G-OFF:** the new OFF arms run on `87351f9`, while OFF_s17 ran on `eaf942a` and OFF_s42 on `a43a88d`. Before
any arm, an OFF smoke (2,000 studies, 1 epoch, seed 42) is run at `eaf942a` (a temporary git worktree) and at
`87351f9`. All checkpoint tensors and val probabilities must be bit-identical, or the run aborts.

**Reading rule.** Cutpoints are cross-fitted as before, with the same fold. Δ_s = TK_s − OFF_s for s in
{42, 17, 7, 23}, and Δ̄ is their mean. The CI of Δ̄ is a paired bootstrap with 1,000 draws at seed 16, with the
same resampled studies for every arm. The bars stay at the pilot's NB/√2. That is stricter than the NB/2 a
four-seed mean would justify, and it is kept so the bar does not drop after a pass.

| NB/√2 | wF1 | F1-5 | macro recall |
|---|---|---|---|
| all val | 0.0019 | 0.0022 | 0.0078 |
| views_1 | 0.0032 | 0.0048 | 0.0081 |
| views_2 | 0.0021 | 0.0033 | 0.0050 |

TK passes only if there is a metric m in {wF1, F1-5} for which both of the first two rules hold:
1. **All val, as before:** all four Δ_s > 0, the CI lower bound of Δ̄ > 0, and Δ̄ > NB/√2 (all val).
2. **Two-view studies (new, required):** on views_2 (n=820), the CI lower bound of Δ̄ > 0 and Δ̄ > NB/√2 (views_2).

The other two rules must also hold:

3. **No loss:** on all val, none of wF1, F1-5 or macro recall has Δ̄ < −NB/√2.
4. **Single-view protected:** on views_1, neither the wF1 Δ̄ nor the F1-5 Δ̄ is below −NB/√2.

**PASS** leads to a full-run proposal, which is its own decision. **FAIL** means direction 3: write the negative
result. No further multi-view variant without a new decision.

Descriptive only:
- per-seed Δ;
- NB re-estimated from the four seeds of TK and of OFF;
- TK − TKnoaux and TKnoaux − OFF over four seeds;
- the attention aux share at seeds 7 and 23;
- G3 on the new TK models.

Cost: about 3 min of smokes, 4 × ~49 min of training, ~5 min of evals and ~25 min for the probe, about 4 h in
total. Disk: ~1 GB per arm on `/home` (19 GB free). The `.npz` files, checkpoints and logs stay on the host.

### Seed replicate — execution report

2026-10-27, host checkout `87351f9`, `~/auxtok4.sh` → `~/auxtok4_20261027/`, one launch, 17:31:03 → `AUXTOK4_DONE`
20:37:31. The probe `auxtok4_probe.py` → `probe.{json,log}` reproduced every TK / OFF / TKnoaux value at seeds 42
and 17 exactly (max |Δ| 0.0).

- **Gate G-OFF — PASS.** The OFF smoke at `eaf942a` and at `87351f9`: 722/722 checkpoint tensors bit-identical,
  val probabilities equal.
- **Training:** TK_s7 48m06s, OFF_s7 32m33s, TK_s23 47m55s, OFF_s23 34m56s. Peak about 5.1 GB (TK) and 4.8 GB (OFF);
  the driver's `max mem` field is still glued to a timestamp digit.
- **G3 on the new TK models — PASS.** views_1 max |Δprob| 2.4e-7, 0 argmax changes. On views_2 every study moved,
  and 2.55% (s7) / 2.83% (s23) of argmax cells changed. Evaluate-only reproduces the training-time npz.

**Verdict by the pre-declared rule: FAIL (direction 3).** Rule 1 fails on both metrics, rule 2 fails on both,
rule 3 passes, rule 4 passes.

| TK − OFF, all val (n=1,808) | s42 | s17 | s7 | s23 | four-seed mean [CI] | NB/√2 |
|---|---|---|---|---|---|---|
| weighted F1 | +0.0022* | +0.0005 | −0.0010 | +0.0021 | +0.0010 [−0.0002, +0.0024] | 0.0019 |
| F1-5 | +0.0066* | +0.0006 | **−0.0051*** | +0.0045* | +0.0016 [−0.0007, +0.0042] | 0.0022 |
| macro recall | +0.0053* | +0.0003 | −0.0046* | +0.0078* | +0.0022 [−0.0003, +0.0051] | 0.0077 |
| AUROC mean | −0.0074* | +0.0077 | +0.0017 | −0.0000 | +0.0005 [−0.0036, +0.0052] | 0.0195 |

| four-seed mean [CI] | views_1 (n=988) | views_2 (n=820) | NB/√2 views_2 |
|---|---|---|---|
| TK − OFF wF1 | +0.0004 [−0.0013, +0.0021] | +0.0010 [−0.0008, +0.0030] | 0.0021 |
| TK − OFF F1-5 | +0.0026 [−0.0007, +0.0064] | +0.0002 [−0.0031, +0.0038] | 0.0033 |
| TK − OFF macro recall | +0.0038* | −0.0038 [−0.0099, +0.0017] | 0.0050 |
| TK − OFF AUROC | −0.0046 | +0.0055 [−0.0033, +0.0158] | 0.0254 |
| TK − TKnoaux wF1 | −0.0004 | +0.0018 [+0.0000, +0.0037] | 0.0021 |
| TK − TKnoaux F1-5 | +0.0018 | −0.0002 [−0.0035, +0.0037] | 0.0033 |
| TK − TKnoaux macro recall | +0.0032* | −0.0051* | 0.0050 |
| TK − TKnoaux AUROC | 0 (identical probabilities) | +0.0103* | 0.0254 |

- **Rule 1:** seed 7 is negative on both wF1 and F1-5 (F1-5 −0.0051*), both CIs cross 0, and both means are below
  NB/√2.
- **Rule 2:** on two-view studies, TK − OFF is flat: wF1 +0.0010, F1-5 +0.0002, both crossing 0 and both below
  NB/√2.
- **Rules 3 and 4:** no all-val or single-view mean falls below −NB/√2.
- The pilot's F1-5 pass (+0.0036) does not replicate: with four seeds the mean halves to +0.0016 and its CI
  crosses 0. One of the two new seeds goes significantly the other way.

**Descriptive.**
- **NB re-estimated from four seeds** (pooled TK and OFF, same 2√2·σ̂ scale as the pilot NB):
  - all val: wF1 0.0036, F1-5 0.0081, macro recall 0.0101, AUROC 0.0138;
  - views_2: wF1 0.0043, F1-5 0.0066.

  The two-seed pilot NB (wF1 0.0027, F1-5 0.0031) underestimated seed noise. Against the four-seed NB/√2 (F1-5
  0.0057), the pilot's +0.0036 would not have passed.
- **What the aux view itself does on two-view studies (TK − TKnoaux, same model):**
  - weighted recall +0.0030* and wF1 +0.0018 (CI lower bound at 0);
  - macro recall −0.0051*;
  - F1-5 −0.0002;
  - AUROC +0.0103*.

  That is a small operating-point move toward the frequent classes plus a ranking gain, all within NB/√2 (AUROC
  0.0254).
- **TKnoaux − OFF on all val:** everything ~0 except AUROC −0.0038*. Training with aux tokens leaves MHCAC on
  single views at OFF's level.
- **Attention:**
  - Each finding's aux share is 0.31–0.36 at s7 and 0.32–0.34 at s23 (0.31–0.38 at s42/s17).
  - At every seed the expert tokens split into aux-leaning (~0.55–0.65) and anchor-leaning (~0.31–0.37) groups,
    and the findings pool from the anchor-leaning ones.
  - No finding leans on the lateral at any seed.

**Conclusion.** Feeding the aux view's tokens into MHCAC does not beat OFF at the pilot budget. The two-seed pass
was seed noise plus cutpoints on single-view studies. On two-view studies the aux view changes the operating point
and ranking slightly, within noise. This is the fourth variant (early P15/P30/P50/Swin-only, LF, LD, AH, TK)
against the same OFF arms, and none passes. As agreed beforehand: **direction 3**, write the negative result. No
further multi-view variant without a new user decision. No full run, no test reading.
