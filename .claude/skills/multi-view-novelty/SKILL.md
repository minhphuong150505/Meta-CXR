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
  (and Swin) streams give fusion real spatial tokens. Untested next step: per-stream
  ablation of fusion on the existing checkpoint (inference only).
- Earlier variants that LOST: `max_aux_views: 3` (2026-10-07), EVOKE flags
  `detach_aux` + `mpc_temperature 0.5` (2026-10-10). Do not retry them unchanged.

## Candidate directions — NONE approved; each needs Rule 1

1. Per-finding gated fusion (each finding learns how much to use the aux view).
2. Aux tokens fed straight into MHCAC's token axis with a view-type embedding.
3. Late fusion: per-view MHCAC logits pooled by attention over views.
4. Train-side: separate LR for `view_fusion`, change the aux-view drop (`p_view_drop`, shipped 0.15).
5. Seed replicates of ON and OFF to measure the noise bar properly.

Pilot recipe used before (cheap): phase 1b only, ~60,000 studies, 2 epochs,
from `run_20261005_paper`'s `checkpoint_phase1a.pth`, read on val two-view
studies. Traps: phase blocks override `--options` (also `run.test_splits`:
phase1c sets `[test]`, so override `run.phases.phase1c.run.test_splits`); A's
training-time `.npz` has no `num_views` — use `~/nomv_probe_20261024/A_rescore.npz`.
