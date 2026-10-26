---
name: multi-view-novelty
description: The user's goal since 2026-10-25 — multi-view View Fusion is the paper's novelty, so Stage 1 with fusion ON must beat fusion OFF. Load before ANY work on multi-view, ViewFusion, StreamAdapter, MPC, view consistency, max_aux_views, aux-view dropout, the fusion-on/off comparison, or a thesis claim about multi-view. Every change needs the user's explicit confirmation first.
---

# Multi-view fusion as the novelty (user decision 2026-10-25)

## The goal

The user wants **View Fusion (multi-view) to be the novelty of the paper/thesis.**
The working target is therefore: **Stage 1 with fusion ON must score better than
fusion OFF** — same recipe, same budget, same test studies, paired bootstrap,
three-class protocol (D-023), on the test split of 3,269 studies.

## Rule 1 — confirm every change with the user first

Before ANY change aimed at this goal — code, YAML, loss weights, recipe, phase
lengths, LR, data/sampling, decision rule, evaluation protocol, which metric or
subgroup counts as "better", or a GPU launch — **stop and ask the user in plain
Vietnamese text** (no AskUserQuestion picker): what exactly changes, why, the
expected cost (GPU hours), and how the result will be read. Wait for an explicit
yes. Approval of one change does not extend to the next one, and a plan
approved earlier does not cover a modified plan. Small changes are not exempt.

## Rule 2 — the improvement must be real

The novelty only counts if it survives review. So:

- Choose designs and hyperparameters on **validation only**; the test split is
  scored **once per final candidate**, never used to pick between variants.
- Write the reading rule (metric, subgroup, noise bar) into the handoff plan
  **before** the run, and do not change it after seeing a result.
- The OFF arm gets the same recipe and budget as the ON arm. Do not weaken it,
  and do not drop metrics/subgroups that went the wrong way.
- One run per arm does not separate seed noise; before claiming a win, run seed
  replicates (ask the user — each is ~10–12 h).
- If ON still does not beat OFF, say so plainly and record it. Never present a
  post-hoc test-set observation as a result. A negative, well-controlled result
  is still reportable; a fabricated or cherry-picked one is not.
- **Noise-bar rule (user decision 2026-10-26, "vẫn áp dụng quy tắc ngưỡng").** A
  pilot advances, or an ON-vs-OFF claim is made, only if (1) its pre-declared CI
  rule passes AND (2) the deciding difference exceeds the measured seed-noise bar
  NB for that metric and subgroup at that budget. A single-seed "*" on AUROC or
  macro recall is not evidence. Every new plan writes the NB check into its
  reading rule. NB at the Stage-1 pilot budget (all val): wF1 0.0027, F1-5 0.0031,
  macro recall 0.0110, AUROC 0.0276 (views_1 / views_2 in
  `PLAN-2026-10-26-pilot-seed-noise.md`); Stage 2 at its pilot budget (val 1,360, beam 4):
  BERTScore 0.0054, CheXbert micro F1-14 0.0192, ROUGE-L 0.0062, BLEU-4 0.0084, CIDEr 0.0319
  (`PLAN-2026-10-26-stage2-seed-noise.md`). Where no NB exists for a budget (full
  runs), say so and ask the user before treating a CI alone as decisive.

## Where things stand (2026-10-25)

| arm | run | note |
|---|---|---|
| ON (reported model) | `run_20261005_paper` | multi-view on, `max_aux_views: 1` |
| OFF | `run_20261024_nomv` | `multi_view: false` via `run.phases.<phase>.model.*`, same 1a, 1b 5 ep + 1c |

- ON − OFF, test, argmax: F1-5 +0.0129 [+0.0090, +0.0174], weighted F1 +0.0057,
  macro recall +0.0060 (significant), AUROC +0.0054 (crosses 0). Below the
  ~0.015 bar, and **equal on single-view studies** → not from the second view.
  Under the headline cutpoints the arms tie (F1-5 0.749 vs 0.748).
  `docs/handoff/PLAN-2026-10-24-multi-view-ablation.md`.
- Step 0 (ON scored with its aux view removed, `model.data.max_aux_views=0`):
  fusion IS used (6.4% argmax changes on two-view studies) and **hurts** argmax
  F1-5 there (−0.0116 [−0.0176, −0.0051]) with AUROC flat. The aux view helps
  **Pleural Effusion** on every metric (wF1 +0.022, AUROC Pos +0.015, Unc
  +0.092) and costs Atelectasis / Consolidation wF1 ≈ −0.04 with AUROC flat →
  looks like an operating-point shift. Step 0b (same comparison under per-arm
  val cutpoints) is in `docs/handoff/PLAN-2026-10-25-multi-view-step0.md`.
- **Step 0b (2026-10-25), headline cutpoints, each arm its own val cutpoints:
  the aux view HELPS on two-view studies** -- A - A_noaux weighted F1 +0.0053
  [+0.0014, +0.0090], macro recall +0.0193, F1-5 +0.0067 [-0.0006, +0.0132];
  Pleural Effusion +0.016, Edema +0.017, Lung Lesion +0.019. ON vs OFF
  (`run_20261024_nomv`) under cutpoints: two-view studies F1-5 +0.0146
  [+0.0083, +0.0208] and all five metrics significant; single-view studies
  F1-5 -0.0084 (ON worse). So the step-0 harm was an operating-point artefact,
  and the remaining loss on single-view studies looks like one cutpoint set
  fitted on a mix of one- and two-view studies (hypothesis: fit cutpoints per
  view count on val). Descriptive, one run per arm.
- **Step 0c (2026-10-25): per-view-count cutpoints FAILED the pre-declared
  rule.** A_pv - B_pv on all 3,269: wF1 +0.0019 [-0.0001, +0.0038], F1-5
  +0.0022 [-0.0020, +0.0061]. They moved A's single-view F1-5 only +0.0016 and
  also helped B on two-view studies, so the single-view deficit is the MODEL
  (A trained with fusion is weaker on single-view inputs), not the cutpoints.
  Do not retry cutpoint tricks; the lever must be training-side (protect
  single-view quality, e.g. direction 4/1). Single-cutpoint ON - OFF wF1
  +0.0021 [+0.0001, +0.0041] = at the zero boundary, not established.
- **Raising `p_view_drop` FAILED its pre-declared val rule (pilot 2026-10-25).**
  YAML ships 0.15 (not 0.2); the drop is drawn independently per encoder. Pilot:
  1b only, 60,000 studies, 2 epochs, val n=1,808, cross-fitted cutpoints. More
  drop makes single-view studies WORSE, not better (P50 - P15 views_1 wF1
  -0.0056*, P30 -0.0029*); P50 is worse everywhere; P30 helps two-view studies
  (F1-5 +0.0058* over P15) but the total does not move. Full run NOT launched.
  Do not retry a higher drop unchanged. `docs/handoff/PLAN-2026-10-25-aux-view-drop.md`.
- **Structural reason fusion may help little (read from code 2026-10-25, not
  measured):** fusion receives BioViL tokens through `_biovil_tokens` with the
  default `legacy_reshape`, so its 196 BioViL "tokens" are channel groups, not
  patches, and 1b's BioViL projector is random and frozen. Only the PubMedCLIP
  (and Swin) streams give fusion real spatial tokens.
- **Per-stream ablation DONE (2026-10-25, inference only, val views_2 n=820,
  cross-fitted cutpoints): the SWIN stream carries the fusion gain.** F - noSW wF1
  +0.0045*, F1-5 +0.0070*, macro recall +0.0107*, AUROC +0.0162*; turning off Swin
  fusion alone ends at the no-fusion level. F - noBV: wF1 / F1-5 flat (-0.0001 /
  -0.0006) but AUROC +0.0097* (BioViL fusion helps ranking, not the operating point).
  F - noPMC: nothing but wP +0.0032*. Total F - N on val: wF1 +0.0041*, AUROC +0.0219*.
  Pre-declared verdict "H supported" -- but its wording "BioViL carries nothing" is
  too strong given AUROC. Nothing adopted. `docs/handoff/PLAN-2026-10-25-fusion-per-stream.md`.
- **Training with fusion in the Swin stream only FAILED its val pilot (2026-10-25;
  aux-drop pilot recipe, in-memory `_fuse` patch, P15/OFF reused):** SW - P15 all-val
  F1-5 -0.0071*, macro recall -0.0075*, AUROC -0.0149*, also on single-view studies
  (F1-5 -0.0088*), where no fusion block runs. The inference-time "Swin carries the
  gain" does not transfer to training. Its F1-5 losses are beyond the seed-noise bar
  measured next; its AUROC losses are not. Do not retry stream-subset fusion unchanged.
  `docs/handoff/PLAN-2026-10-25-swin-only-fusion.md`.
- **Seed noise at the pilot budget MEASURED (2026-10-26; P15 and OFF rerun at seed 17).**
  The same recipe at another seed moves val AUROC by -0.0159* (ON) / -0.0112* (OFF), and
  macro recall by -0.0073*. **The val bootstrap does not cover seed noise, so a
  single-seed "*" on AUROC or macro recall means nothing.** Noise bar NB (all val, 2 x
  RMS of the two seed deltas, from two draws only): wF1 0.0027, F1-5 0.0031, macro recall
  0.0110, AUROC 0.0276; views_2 F1-5 0.0047. **ON - OFF at pilot budget: seed 42 ~0, seed
  17 ON worse (F1-5 -0.0031); two-seed mean -0.0015.** Fusion does not beat OFF at this
  budget. P50 and SW are worse beyond NB; P30 is noise except views_2 F1-5 +0.0058 (NB
  0.0047). Future pilots must also clear NB: ADOPTED by the user 2026-10-26 (Rule 2).
  `docs/handoff/PLAN-2026-10-26-pilot-seed-noise.md`.
- **Stage 2 ON vs OFF, val pilot (2026-10-26): OFF BETTER by the pre-declared rule.** Stage 2
  retrained on each Stage-1 (`run_20261005_paper` vs `run_20261024_nomv`), 2026-10-16 refit-pilot
  recipe (10,000 studies, 476 updates, P2 + LR 3e-4, own cutpoint cues), beam 4 on all 1,360 val.
  ON' - OFF: CheXbert micro F1-14 -0.0171 [-0.0329, -0.0012] (precision and recall both lower),
  BERTScore -0.0006 [-0.0073, +0.0061]; every other metric crosses 0. ON' reproduced the old arm A
  on 1,360/1,360 reports (Stage 2 was deterministic at its fixed seed). Full Stage 2 (S2) not run.
  `docs/handoff/PLAN-2026-10-26-stage2-fusion-on-off.md`.
- **Stage-2 seed replicate (2026-10-26, both arms at `--seed 17`): "OFF better" is WITHIN Stage-2
  seed noise.** Seed 17 ON - OFF CheXbert micro F1-14 -0.0058 [-0.0205, +0.0098], BERTScore -0.0035;
  two-seed means -0.0115 / -0.0020 vs NB/sqrt2 0.0136 / 0.0038. The ON recipe alone moved CheXbert micro
  F1-14 +0.0134 between seeds. Seed-17 delta negative on all 9 metrics; three non-deciding CheXbert
  metrics have means beyond NB/sqrt2 in OFF's favour (descriptive). **Fusion ON gives Stage 2 no
  measurable benefit at this budget; do not claim either direction.** Two Stage-2 seeds, one Stage-1 run
  per arm. `docs/handoff/PLAN-2026-10-26-stage2-seed-noise.md`.
- **Late fusion val pilot (2026-10-26, 2 seeds, pilot recipe, NB rule): STOP.** LF - OFF two-seed
  mean, all val: wF1 +0.0000 [-0.0017, +0.0017], F1-5 -0.0005 [-0.0034, +0.0024], seeds disagree in
  sign; rule 1 fails, rules 2/3 pass. Single-view wF1 -0.0024* (within NB/sqrt2; logits are exactly the
  anchor's, so the cost comes from MHCAC also training on laterals), two-view wF1 +0.0021 (crosses 0),
  weighted recall +0.0036*. LF ~ P15 (early) on every deciding metric. The 112-parameter pool stayed at
  ~0.5 aux weight per finding = plain averaging. So neither early nor late fusion beats OFF at the pilot
  budget; the aux view's small two-view gain is offset by a small single-view loss every time.
  `docs/handoff/PLAN-2026-10-26-late-view-fusion.md`.
- **Late fusion with the aux pass detached (LD, `late_detach_aux`), val pilot 2026-10-27, same rule:
  STOP.** LD - OFF two-seed mean, all val: wF1 +0.0001 [-0.0017, +0.0018], F1-5 +0.0007
  [-0.0024, +0.0037], Δ42 negative on both; rule 1 fails, rules 2/3 pass. The detach removed LF's
  single-view loss (views_1 LD - LF wF1 +0.0023*, F1-5 +0.0037*) AND LF's two-view gain (views_2
  LD - LF wF1 -0.0029*, macro recall -0.0073*). Lesson: with ONE shared MHCAC, the aux view helps
  two-view studies only if MHCAC trains on laterals, and that training costs the anchor about as much.
  A direction that gives the aux view its own capacity (view-specific head / adapter, candidate 2)
  is what this points at -- untested. Pool stayed ~0.5 again.
  `docs/handoff/PLAN-2026-10-26-late-fusion-detach-aux.md`.
- Earlier variants that LOST: `max_aux_views: 3` (2026-10-07), EVOKE flags
  `detach_aux` + `mpc_temperature 0.5` (2026-10-10). Do not retry them unchanged.

## Candidate directions — NONE approved; each needs Rule 1

1. Per-finding gated fusion (each finding learns how much to use the aux view).
2. Aux tokens fed straight into MHCAC's token axis with a view-type embedding.
3. Late fusion: per-view MHCAC logits pooled by attention over views. Implemented opt-in
   (`model.view_fusion.mode: late`, `mhcac/late_view_pool.py`). **Val pilot 2026-10-26 at 2 seeds
   with the NB rule: STOP, it does not beat OFF** (see "Current evidence"). Do not retry it unchanged.
   **Variant `model.view_fusion.late_detach_aux: true` (aux-view pass without gradient) -- val pilot
   2026-10-27: STOP** (see "Current evidence"). Do not retry it unchanged.
   `docs/handoff/PLAN-2026-10-26-late-fusion-detach-aux.md`.
4. Train-side: separate LR for `view_fusion`, change the aux-view drop (`p_view_drop`, shipped 0.15).
5. Seed replicates of ON and OFF to measure the noise bar properly.

Pilot recipe used before (cheap): phase 1b only, ~60,000 studies, 2 epochs,
from `run_20261005_paper`'s `checkpoint_phase1a.pth`, read on val two-view
studies. Traps: phase blocks override `--options` (also `run.test_splits`:
phase1c sets `[test]`, so override `run.phases.phase1c.run.test_splits`); A's
training-time `.npz` has no `num_views` — use `~/nomv_probe_20261024/A_rescore.npz`.
