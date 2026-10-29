# PLAN 2026-10-27 — late fusion with a separate MHCAC for the aux view (AuxLoss-style)

Status: **DONE 2026-10-27 — STOP, AH does not beat OFF** (rule 1 and rule 3 fail; see "Execution
report"). Approved 2026-10-27 ("ok duyệt cả 3, làm đi"): the code change, step G + step P, and the
reading rule. No full run.

Implementation note: the aux pass of LF / LD / AH now lives in one torch-only helper,
`mhcac.late_view_pool.late_aux_view_logits`, so the gradient split is tested on CPU with real autograd:
`CE(aux) + CE(pooled)` gives zero gradient to the anchor head, projector and text tower, and the full
loss gives them exactly `CE(anchor)`'s gradient. Removing any one of the three cuts makes that test
fail (checked). CPU suite: the same 21 baseline failures (no nltk / torchvision / env config on the dev
box), all `tests/test_late_view_fusion.py` passing. G1 checks that LF is still bit-identical on GPU.

## Why

Two late-fusion pilots, both STOP, show one trade. With one shared MHCAC, the trade cannot be avoided.

| pilot | single-view studies vs OFF | two-view studies vs OFF | all val |
|---|---|---|---|
| LF: MHCAC also trains on aux views | wF1 −0.0024* | wF1 +0.0021, w-recall +0.0036* | 0 |
| LD: aux pass without gradient | flat | gain gone (LD − LF wF1 −0.0029*) | 0 |

Training MHCAC on laterals is what makes the aux view useful. The same training costs the anchor
path, which is mostly frontal, about as much as it gains. So the aux view needs its own capacity.

Hashir et al. (MIDL 2020, PadChest PA+L) built a model called AuxLoss along these lines. It gives each
view its own branch and its own per-view loss, plus a joint loss on the combined prediction. As recalled
here (not re-checked for this plan), AuxLoss was among their strongest joint models, but the gain over PA
alone was small and limited to some findings. That is a reason to keep expectations low, not a promise.

This plan gives the aux view its own MHCAC. It also cuts every gradient path from the aux view into the
anchor path. The anchor MHCAC then trains on exactly OFF's objective.

## Design

New opt-in key: `model.view_fusion.late_aux_head: true | false`. When the key is absent it is `false`,
and the LF path runs bit-identically. The key is valid only in `mode: late`. It is refused together with
`late_detach_aux`, because the two keys overlap.

Modules:
- `mhcac`: the anchor MHCAC. It is the same module as today and runs the OFF path.
- `mhcac_aux`: new. It is a second MHCAC of the same class and configuration.
  - Its starting weights are a copy of `mhcac` right after the 1a checkpoint loads. The 1a checkpoint
    has no `mhcac_aux` tensors, and the copy is made only when they are missing. Both heads therefore
    start from the same point.
  - The cost is one extra MHCAC worth of parameters. Step G logs the exact count.
- `late_view_pool`: unchanged, 112 parameters.

Forward pass, for a study with anchor A and a real aux view L:
- `z_A = mhcac(proj(A))`. This is exactly as today.
- `z_L = mhcac_aux(sg(proj(L)), text = sg(text))`.
  - The aux view passes `ln_vision` and the shared projector under `no_grad`, as LD already does. Aux
    gradients therefore never reach the projector. The projector trains during the first 10% of 1b.
  - The report text embeddings are detached for the aux head, so aux gradients never reach the
    Q-Former text tower either.
- `z = pool(sg(z_A), z_L)`.
  - Stop-gradient on the anchor input of the pool. In LF and LD the pooled loss still sent gradient
    into `z_A`; here it does not.
  - A study with no real aux view still gets exactly `z_A`. The pool guarantees this, and it is
    unchanged.

Loss: `L = CE(z_A) + CE(z_L) + CE(z)`.
- The `CE(z_L)` and `CE(z)` terms apply only to studies that have a real aux view. Both are averaged
  over those studies.
- Each term has weight 1, with no new loss-weight key.
- The anchor term has weight 1.0, the same as OFF. LF's 0.5/0.5 split does not apply here.

What this guarantees, by construction:
- **The anchor MHCAC, the shared projector and the text tower get gradient only from `CE(z_A)`.**
  That is OFF's objective.
- The arm is still not bit-identical to OFF, because the data RNG and the dropout draws differ: the
  aux view is loaded and passes through MHCAC dropout. Single-view studies should therefore match OFF
  within seed noise. That is rule 3 below.
- At inference, a single-view study gets exactly the anchor logits, as in LF and LD.

What it does not guarantee:
- That `mhcac_aux` learns anything useful from roughly 30,000 two-view studies in 2 epochs.
- That the pool learns more than averaging. In LF and LD it stayed at about 0.5.
- See also "Honest odds" below.

Trainable lists: `mhcac_aux` does not match the prefix `mhcac`, because matching uses
`name == p or name.startswith(p + ".")`. It therefore has to be added to `phase1b.trainable` and to
`transition.fade_in`. **The YAML is not edited**, because of the mtime trap for Stage-2 caches. Instead
both lists are overridden on the command line in the phase block. Step G checks from the log that
`mhcac_aux` is trainable and fades in like `mhcac`. `checkpoint_keep` follows the trainable lists, so
the checkpoint keeps `mhcac_aux`.

## Code change (opt-in, absent key = bit-identical)

`model/lavis/models/blip2_models/blip2_qformer.py`:
- `__init__` and `from_config`: read and validate the key, then build `mhcac_aux` when it is true.
- `_late_view_logits`: when the key is true, the aux view goes through `mhcac_aux` with the projector
  under `no_grad` and the text detached, and the pool receives `z_A.detach()`. The function also
  returns `z_L` for the loss.
- The loss branch: when the key is true, compute the three-term loss above. The `late_classification_loss`
  path for LF and LD stays unchanged.
- A helper that copies `mhcac` into `mhcac_aux`. `pretraining/train.py` calls it after the init load,
  only when the key is on and the checkpoint has no `mhcac_aux.*` tensors. The copy is logged.

Tests, in `tests/test_late_view_fusion.py` (CPU):
- The key is false by default. Outside `late` it is refused, and it is refused together with
  `late_detach_aux`.
- **Gradient isolation, the main invariant.** On a tiny model, take `autograd.grad` of
  `CE(z_L) + CE(z)` with respect to the parameters of `mhcac`, the shared projector and the text
  tower. Every gradient must be None or zero. The same gradient with respect to `mhcac_aux` and
  `late_view_pool` must be non-zero.
- Single-view studies return exactly `z_A` with the key on.
- The copy helper makes `mhcac_aux` equal to `mhcac` tensor by tensor, and it does nothing when the
  checkpoint already holds `mhcac_aux`.

The same commit updates the docs: the struct pages for `blip2_qformer` and `late_view_pool.py`,
`train.py` where the init load is described, `CLAUDE.md`, the Vietnamese `README`, and the
multi-view-novelty skill.

## Step G — equivalence and smoke (~8 min GPU), abort on failure

Smoke size, as in the LF and LD plans: 2,000 train and 400 val studies, 1 epoch, seed 42.
- **G1:** run a late smoke with the key absent on the new code. All 724 checkpoint tensors and the val
  probabilities must equal LD's G1 run on the LF code. This keeps LF and LD valid comparators.
- **G2:** run a smoke with `late_aux_head=true`. It must meet all of the following:
  - rc=0 and no NaN;
  - the key is logged;
  - the log confirms the copy into `mhcac_aux`;
  - `mhcac_aux` appears in the trainable and fade-in groups;
  - after training, `mhcac_aux` differs from `mhcac`, and `late_view_pool` has moved from zero;
  - the checkpoint holds `mhcac_aux.*`.
  Step G also records the parameter count, max mem and s/it.
- **G3:** score G2's checkpoint with and without the aux view. Single-view studies must be
  bit-identical, and two-view studies must move.

## Step P — pilot (~1.8 h GPU), test is NOT scored

The pilot uses the same recipe as every pilot since 2026-10-25:
- phase 1b only;
- the first 60,000 train studies;
- 2 epochs, warm-up 150;
- starting from `run_20261005_paper/checkpoint_phase1a.pth`;
- `test_splits=[]`.

Runs are strictly serial on one card, into one new directory: `~/lateauxhead_20261027/`.

| arm | overrides | seed |
|---|---|---|
| AH_s42 | LF's overrides, plus `phase1b.model.view_fusion.late_aux_head=true`, plus the two list overrides | 42 |
| AH_s17 | same | 17 |

Comparators already on disk, with the same recipe and the same studies:
- Deciding: OFF_s42 (`~/auxdrop_pilot_20261025/OFF`) and OFF_s17 (`~/seednoise_20261026/OFF_s17`).
- Descriptive: LF_s42 and LF_s17, LD_s42 and LD_s17, and P15 at both seeds.

Each arm should take about 50–55 min, similar to LF, because the aux pass again has a backward pass,
now through `mhcac_aux`. Evaluate-only scoring with and without the aux view adds about 5 min.

Guards, abort on any failure:
- Before launch: the GPU is idle, the data drive is mounted with `ntfs3`, no other job is running, and
  the output directory does not exist.
- For each arm:
  - rc=0, with no Traceback, OOM or NaN;
  - the logged seed is the intended one;
  - the logged config shows `"mode": "late"` and `"late_aux_head": true`;
  - `mhcac_aux` is in the trainable groups;
  - `val_predictions_epoch_best.npz` exists and has `num_views`.
- No test predictions are written.

## Reading rule — fixed before any result (identical to LF and LD)

Method:
- Cutpoints are cross-fitted. Rows are sorted by `sample_key`, and the fold is
  `default_rng(16).permutation(n) % 2`.
- Groups come from `num_views`: 988 single-view and 820 two-view val studies.
- Δ_s = AH_s − OFF_s for each seed, and Δ̄ is the mean over s in {42, 17}.
- The CI of Δ̄ is a paired bootstrap with 1,000 draws, seed 16.

| NB/√2 | wF1 | F1-5 | macro recall |
|---|---|---|---|
| all val | 0.0019 | 0.0022 | 0.0078 |
| views_1 | 0.0032 | 0.0048 | 0.0081 |

**AH advances to a full-run proposal only if all three hold:**
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
- AH − LF and AH − LD;
- AH scored without the aux view;
- `mhcac_aux` alone on the two-view studies, i.e. the lateral-only prediction;
- the mean aux pool weight per finding.

⚠ This is the **third** late-fusion variant read against the same two OFF arms. Three tries at one bar
raise the chance of a lucky pass, so this must be stated beside any pass. NB rests on two draws per
arm.

## Honest odds

The gain can only come from the 820 two-view val studies; single-view studies are by design about
OFF.
- To clear the all-val bar of wF1 Δ̄ > 0.0019, the two-view studies need about +0.0042 wF1 over OFF.
- LF's two-view gain was +0.0021, and its CI crossed 0.
- So a pass needs `mhcac_aux` and the pool to roughly double what LF got from the aux view, while
  keeping the anchor at OFF's level. That is possible, because LF paid for its gain on the anchor side
  and this design does not, but it is not likely.

The pilot is worth running because it is the first design that removes the trade both earlier pilots
measured. A STOP here would also be informative: it would close the "aux view needs its own capacity"
explanation at this budget.

## If it passes (not in this plan, each its own decision)

- **A capacity control** before any claim. In the AH design, two-view studies get two MHCAC heads, so
  a reviewer can say the gain is an ensemble effect. The control is an arm where `mhcac_aux` reads the
  anchor image a second time, with the same parameters and losses. If AH beats this control, the gain
  is the second view's.
- Seed replicates and a full run (1b for 5 epochs plus 1c), then a single test reading.
- Any change to the shipped YAML.

## Cost

- Code and CPU tests: about 1–2 h of planner work, plus a host test run.
- GPU:
  - step G: about 8 min;
  - step P: about 1 h 50 min plus scoring;
  - total about 2 h, one card, serial.
- Disk: two pilot checkpoint sets, a few GB in total, on `/home`.

## Data handling

The `.npz` files, checkpoints and logs stay on the host. Only aggregates come back here.

## Execution report

2026-10-27, host checkout `c5162d8`, `~/ahead.sh` → `~/lateauxhead_20261027/`, launched once at 10:46
(a first launch aborted at G2 on a false alarm in the script's own grep, which looked for
`'mhcac_aux'` while the log prints `mhcac_aux 80.55M`; the grep was fixed, the output directory removed,
and nothing else changed). `AHEAD_DONE` 12:51. Probe `~/ahead_probe.py` → `probe.{json,log}`; its sanity
check reproduced every OFF / P15 / LF / LD value of the earlier probes exactly (max |Δ| 0.0).

**Step G — PASS.**
- G1 (LF) and G1b (LD) against the `fdb9bfa` arms: 724/724 tensors bit-identical, val probabilities equal.
- G2 (AH smoke): `mhcac_aux` 191 tensors mirroring `mhcac`, 80,550,186 parameters each, all 191 differ from
  `mhcac`, 190 moved from init; pool moved (|max| 2.9e-3); max mem 6,418 MiB.
- G3: single-view studies max |Δprob| 0.0 with and without the aux view; every two-view study moved;
  evaluate-only reproduces the training-time npz.

**Step P — two seeds.** Training 46m42s (s42) / 48m55s (s17), max mem 6,481 MiB, `mhcac_aux` 80.55M
trainable. G3 re-checked on both pilot models (views_1 n=988 max |Δprob| 0.0; views_2 n=820 all moved;
argmax cells changed 4.4% / 6.8%). Pool: mean aux weight per finding 0.46–0.50 at both seeds, i.e. a
plain average again.

**Verdict: STOP.** Rule 1 FAILS, rule 2 passes, rule 3 FAILS (by 0.0001). Cross-fitted cutpoints,
paired bootstrap 1,000, seed 16; `*` = CI excludes 0.

| AH − OFF, all val (n=1,808) | s42 | s17 | two-seed mean [CI] | NB/√2 |
|---|---|---|---|---|
| weighted F1 | −0.0009 | −0.0026* | −0.0017 [−0.0036, +0.0002] | 0.0019 |
| F1-5 | +0.0007 | +0.0001 | +0.0004 [−0.0027, +0.0034] | 0.0022 |
| macro recall | +0.0003 | −0.0014 | −0.0005 [−0.0046, +0.0032] | 0.0077 |
| AUROC mean | +0.0064 | −0.0026 | +0.0019 [−0.0040, +0.0082] | 0.0195 |

- Rule 3: views_1 (n=988) AH − OFF weighted F1 −0.0033* against −NB/√2 = −0.0032; F1-5 −0.0026.
  ⚠ The single-view probabilities of AH are `mhcac`'s alone, trained on OFF's objective (AH − AHnoaux
  views_1 AUROC exactly 0); AHnoaux − OFF views_1 wF1 is −0.0018 (CI crosses 0). The rest of the views_1
  shortfall comes from AH's cutpoints, which are fitted on all val and so move with its two-view
  predictions. The rule is read as written; rule 1 fails regardless.

Descriptive only (cannot change the verdict):
- views_2 (n=820), AH − OFF mean: AUROC +0.0229 [+0.0118, +0.0343] (NB/√2 0.0254, so within seed noise),
  weighted precision +0.0062*, weighted recall +0.0039*, but weighted F1 −0.0043* and macro recall
  −0.0248 [−0.0345, −0.0173] — both beyond NB/√2 (0.0021 / 0.0050), at both seeds. F1-5 +0.0006.
  The aux head raises ranking on two-view studies and moves the operating point against the minority
  classes; the cutpoints do not recover it.
- AH − LF (all val): wF1 −0.0017*, F1-5 +0.0009. AH − LD: wF1 −0.0018*, F1-5 −0.0003. AH − P15: wF1
  −0.0007, F1-5 +0.0019. None beyond NB/√2 in AH's favour.
- AH − AHnoaux (same model, aux view removed), all val: wF1 −0.0007, F1-5 +0.0019, AUROC +0.0036; on
  views_2: AUROC +0.0183*, macro recall −0.0269*.
- AHauxonly − AH (the aux head's logits alone on two-view studies): views_2 F1-5 −0.0161*, AUROC
  −0.0210*. The aux head alone is clearly weaker than the pool; pooling helps.
- AHnoaux − OFF (does the anchor path stay OFF-equivalent?): all val every metric within NB/√2 (wF1
  −0.0010, F1-5 −0.0015, AUROC −0.0017). Consistent with the gradient cut working.
- AH seed delta (17 − 42), all val: AUROC −0.0202* — the same size as the OFF / P15 seed swings.

Reading: a separate aux head removes the anchor-path cost that LF had (the anchor is OFF-equivalent),
but the second view still adds no deciding-metric gain. On two-view studies it raises AUROC by about
the seed-noise bar and costs macro recall well beyond it. This is the third late variant read against the
same OFF arms (LF, LD, AH); none passes. Do not retry AH unchanged. One pilot budget, two seeds, val only;
test not scored.
