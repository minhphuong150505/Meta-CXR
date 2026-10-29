# PLAN 2026-10-27 — what does the second view buy? Evidence package on val (inference only)

Status: **DONE 2026-10-27 — H1 (second view helps Pleural Effusion) NOT CONFIRMED on val;
report below.** Direction chosen by the user 2026-10-27 ("a đi"), plan approved ("ok").

Scope: **inference only, val only.** No training, no change to repo code or YAML, test
never scored. GPU ~15 min (4–6 evaluate-only val passes of ~2–3 min each). CPU ~20 min.

## Why: what the literature says, and what it implies here

Measured gains from adding the lateral view to a frontal chest X-ray model are small
everywhere they have been measured with care:

- Hashir et al., MIDL 2020 (PadChest, 5 runs): PA-only AUC 0.793 ± 0.007, best joint
  model 0.809 ± 0.003. DualNet ignores the lateral view entirely. The robust joint model
  is "AuxLoss": a separate branch and loss per view plus a joint loss.
- van Tulder et al., MICCAI 2021 (CheXpert, 4 runs): frontal 0.827 ± 0.007, late join
  0.829, cross-view transformer 0.834 ± 0.002. No significance test on CheXpert. The
  clearest per-finding gain is Pleural Effusion (0.945 → 0.954–0.955).
- EVOKE (the origin of this repo's ViewFusion/MPC): multi-view input without MPC lowers
  CheXbert F1-14 (0.484 → 0.477). No CIs.

So ON ≈ OFF in aggregate (this repo, 2026-10-24 → 10-27) matches the literature. The
same papers show how to report the second view honestly:

- per-finding analysis with a clinical rationale;
- a missing-view table (both views / frontal only / lateral only, Hashir Table 1);
- an AUC-vs-proportion-of-studies-with-a-lateral curve (Hashir Fig. 2);
- mean ± SD over seeds.

Clinical rationale for the one pre-declared finding: a lateral film shows a pleural
effusion from ~50 mL, an upright PA film from ~200 mL (Blackmore et al., Acad Radiol
1996). Blunting of the posterior costophrenic angle corresponds to ~26 mL on CT
(J Thorac Imaging 2015).

⚠ **Disclosure, to be carried into any paper text.** The Pleural Effusion hypothesis
was formed AFTER seeing test (step 0 / 0b, 2026-10-25: A − A_noaux on two-view test
studies, wF1 +0.022, AUROC Pos +0.015, Unc +0.092). Those test numbers cannot confirm
it. This plan tests it on val, which was never read per finding for this question, and
replicates it on two independently seeded pilot models.

## Models

| id | checkpoint | role |
|---|---|---|
| A | `run_20261005_paper/phase1c/.../checkpoint_best.pth` | ON, full budget (reported model) |
| B | `run_20261024_nomv/phase1c/.../checkpoint_best.pth` | OFF, full budget |
| R42 | `~/auxdrop_pilot_20261025/P15` best (phase 1b) | ON, pilot budget, seed 42 |
| R17 | `~/seednoise_20261026/P15_s17` best (phase 1b) | ON, pilot budget, seed 17 |

## Input conditions (val, 1,808 studies)

| condition | how | needed for |
|---|---|---|
| `both` | as trained (`max_aux_views=1`) | A, R42, R17. Existing val npz: A's `~/nomv_probe_20261024/A_val.npz`; R42 / R17 from their training runs |
| `frontal` | `model.data.max_aux_views=0` | A: existing `A_noaux_val.npz`. **New:** R42, R17. B is always frontal (`multi_view=false`) |
| `lateral` | `model.data.anchor_priority=[lateral,PA,AP] model.data.max_aux_views=0` | **New:** A, B |

`model.data` is set by no phase block, so `--options` reaches it (verified for
`max_aux_views` in step 0). `anchor_priority=[lateral,PA,AP]` changes the anchor only for
studies that have a lateral (`build_study_index`, `model/lavis/data/mimic_cxr_utils.py`).
Studies without one keep their frontal anchor.

**New GPU passes: 4** (A-lateral, B-lateral, R42-frontal, R17-frontal), plus R42-both /
R17-both re-scores only if their training npz lacks `num_views` or fails the
reproduction guard.

## Subgroups (rebuilt on the host from the val CSV, never copied here)

The analysis script rebuilds the val study index with `build_study_index`
(`anchor_priority=[PA,AP,lateral]`, `max_aux_views=1`) in dataset order and asserts
row-by-row that `1 + len(aux)` equals A's `num_views`, then defines:

- **PL (paired frontal + lateral):** anchor PA/AP and aux view id 2 (LATERAL/LL). This is
  the primary subgroup, the analogue of Hashir's paired set.
- **F1:** single-view frontal studies (no aux).
- **other:** everything else (PA+AP pairs, lateral-only, unknown), reported as counts only.

## Commands (host) — template

```bash
cd ~/Meta-CXR && git pull --ff-only          # record the sha
PY=~/.venvs/meta-cxr-stage1-311/bin/python
OUT=~/mv_evidence_20261027; test ! -e $OUT || exit 1; mkdir -p $OUT
# guards before EVERY pass: findmnt -no FSTYPE /mnt/drive1tb == ntfs3; nvidia-smi idle;
# no pretraining.train / run_medgemma_qlora process (ps -eo pid,args positional match)
A=~/run_20261005_paper; AD=$A/phase1c/mimic_cxr_full_blip2
touch $OUT/.marker_A_lat
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled $PY -m pretraining.train \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --options run.phase=phase1c run.phase_root=$A run.evaluate=True run.save_predictions=True \
    run.output_dir=$OUT/A_lat run.resume_ckpt_path=$AD/checkpoint_best.pth \
    run.distributed=false run.world_size=1 run.phases.phase1c.run.test_splits=[val] \
    model.data.max_aux_views=0 "model.data.anchor_priority=[lateral,PA,AP]" > $OUT/A_lat.log 2>&1
# B: same, with B's paths plus the NOMV phase overrides of PLAN-2026-10-24 (phase1c block).
# R42 / R17: run.phase=phase1b, their own phase_root / checkpoint_best,
#   run.phases.phase1b.run.test_splits=[val], model.data.max_aux_views=0.
# After each pass: move the fresh val_predictions_epoch_provided.npz (-newer marker) to $OUT/<id>.npz
$PY ~/mv_evidence_probe.py ... > $OUT/probe.log   # ad hoc, aggregates only
```

## Guards (any failure → stop and report; nothing is a result)

- Each pass: `Loading evaluation weights from` × 1; dataset line `[val] 1808 studies`;
  no `[test]` dataset line; no new test npz; rc 0, no Traceback.
- Dataset line: `0 with a complementary view` for every `frontal` / `lateral` pass.
- Reproduction: the B training-time val npz versus the B pass on the F1 subgroup (frontal
  anchor, identical input), max |Δp| ≤ 1e-4. A_lat versus A_val.npz on F1: the same
  check. On PL, A_lat must differ from A_val (> 1e-3), or the anchor override did not
  take.
- R42 / R17 `both` npz: re-scoring at HEAD reproduces it (max |Δp| ≤ 1e-4) when it is
  re-scored. `frontal` versus `both` on F1: identical (≤ 1e-4).
- Index rebuild matches A's `num_views` on every row, otherwise STOP.

## Reading rule — fixed before any result

Cutpoints are cross-fitted on val as in every pilot (rows sorted by `sample_key`, fold =
`default_rng(16).permutation(n) % 2`). Each (model, condition) fits its own. Paired
bootstrap 1,000 draws, seed 16, `*` = 95% CI excludes 0.

### H1 (primary, the only claim this plan can make): the second view helps Pleural Effusion

On PL, within-model counterfactual `both − frontal`:

- **CONFIRMED** iff all of:
  1. A: Pleural Effusion one-vs-rest AUROC (Positive) CI > 0;
  2. A: Pleural Effusion weighted F1 (cross-fitted cutpoints) point estimate > 0;
  3. R42 and R17: Pleural Effusion AUROC-Pos point estimate > 0 for both.
- **PARTIAL**: 1 holds, but 2 or 3 fails. Report it as "seen in the reported model,
  not robust across seeds / operating point".
- **NOT CONFIRMED**: 1 fails. The Pleural Effusion claim is dropped from the paper; the
  test observation of step 0 stays a post-hoc remark.

AUROC is primary because it does not depend on cutpoints. Step 0 showed that the
second view moves the operating point, and an operating-point effect is exactly what
must not be mistaken for information. The comparison is within one set of weights, so
seed noise does not enter it. Generality across training seeds is what condition 3
checks. No seed-noise bar (NB) exists per finding, and none is invented here.

### Descriptive only (no claim, reported in full whatever the sign)

- **Missing-view table (Hashir style)** on PL, for A and B under `both` / `frontal` /
  `lateral` (B has no `both`): wF1, F1-5, macro recall, AUROC mean. Plus the F1 subgroup
  (A − B, the known single-view deficit). Every ON − OFF cell is single-seed and is
  read against NB (pilot-budget NB, stated as such: no full-budget NB exists).
- **Per finding, all 14**, A `both − frontal` on PL: AUROC Pos / Unc, wF1. 42 cells, no
  multiplicity correction, so none of them is a claim.
- **Proportion curve (CPU only):** for p in {0, 0.1, …, 1}, on PL, take A's `both` row
  for a random fraction p of studies and the `frontal` row otherwise (20 draws per p,
  `default_rng(16)`). The cutpoints are A-`both`'s cross-fitted ones, held fixed. Curves:
  wF1, AUROC mean, Pleural Effusion AUROC-Pos, with B as a flat reference line. The
  endpoints equal the two conditions, by construction.

### What this plan does NOT do

- It does not claim ON beats OFF overall (still not established at any budget).
- It does not touch test. A test confirmation of H1 would be a separate step, decided
  after this one.
- It does not adopt anything: no YAML, no cutpoint file, no code.

## Next steps this plan feeds (each needs its own approval)

- If H1 is CONFIRMED: one test pass of the same PL counterfactual for Pleural Effusion
  only (rule copied from H1). Optionally an external check on IU-Xray (nearly every
  study has frontal + lateral; labels would come from the CheXpert labeler on its
  reports).
- The training-side direction the literature points to (Hashir's AuxLoss: view-specific
  capacity plus a loss per view) is candidate 2 in the skill. It is untested and separate.

## Data handling

`.npz`, JSON, logs and the rebuilt index stay on the host. Only aggregates come back
here. No `subject_id` / `study_id` / `dicom_id` / paths / report text.

## Execution report

Approved by the user 2026-10-27 ("ok"). Host checkout `fa58119`, chain `~/mv_evidence.sh`
launched once 01:31. Four val passes 01:31 → 01:35, all guards passed: weights line × 1,
`[val] 1808 studies ... 0 with a complementary view`, no test split built or scored, no
fusion built in B. Artifacts stay on the host: `~/mv_evidence_20261027/`, plus the probe
`~/mv_evidence_probe.py` (ad hoc, not in the repo).

**Two deviations, both in the analysis script; no GPU pass was repeated.**
1. Subgroups are built from the val CSV with `build_study_index`, as planned. The
   eval `sample_keys` digest `batch["image_id"]`, the annotation row of the ANCHOR image,
   not the study id. A lateral-anchor pass therefore keys a study by its lateral row. The
   probe re-keys the two `lateral` passes to the frontal-anchor key per study. The first
   probe attempt aborted on the row guard for this reason (`probe.attempt1.log`).
2. The "anchor override took" guard compares against the 800 studies with a frontal
   anchor and ANY lateral row (`HASLAT`). Three of them have a lateral that is not their
   `max_aux_views=1` aux (PA + AP + lateral), so the primary subgroup PL is 797 studies.

⚠ **The primary comparison was first computed during a CPU dry run of the probe**, before
the GPU passes. Its two inputs (`A_val.npz`, `A_noaux_val.npz`) already existed. The rule
above had already been written and was not changed. The final run reproduces the dry-run
numbers exactly (same bootstrap stream).

**Guards (final probe): PASS.**
- Keys matched 1,808 / 1,808, and the rebuilt `num_views` matches all three `both` files.
- The lateral anchor changed exactly the 800 HASLAT studies, for both A and B.
- Identical inputs give identical outputs (max |Δp| 0.0): A_lat vs A_both on F1, B_lat vs
  B_fr on F1, every `frontal` vs `both` on views_1.
- Subgroups: PL 797, F1 (single frontal) 930, other 81.

### H1 — pre-declared verdict: **NOT CONFIRMED**

| PL (797), `both − frontal` | Pleural Effusion AUROC Pos | Pleural Effusion wF1 |
|---|---|---|
| A (full budget) | +0.0056 [−0.0009, +0.0129] | +0.0101 [−0.0035, +0.0224] |
| R42 (pilot, seed 42) | +0.0056 [−0.0012, +0.0129] | −0.0064 [−0.0179, +0.0041] |
| R17 (pilot, seed 17) | +0.0037 [−0.0038, +0.0113] | −0.0045 [−0.0168, +0.0073] |

Conditions: 1 (A's AUROC-Pos CI > 0) **fails**; 2 (A's wF1 point > 0) holds; 3 (both
seeds' AUROC-Pos point > 0) holds. By the rule, the Pleural Effusion claim is dropped, and
step 0's test observation (+0.015 *) stays a post-hoc remark. The AUROC direction is
positive in all three models, but no single CI excludes zero at n = 797. The wF1 sign
splits between the full-budget model and the pilots.

### Descriptive (no claim)

**Within-model `both − frontal` on PL, all three models, aggregates:**

| | wF1 | F1-5 | macro recall | AUROC mean |
|---|---|---|---|---|
| A | +0.0031 [−0.0004, +0.0065] | +0.0028 | +0.0066 | **+0.0229 [+0.0078, +0.0350]** * |
| R42 | +0.0006 | −0.0014 | **+0.0121** * | **+0.0193 [+0.0063, +0.0280]** * |
| R17 | −0.0008 | −0.0015 | +0.0047 | **+0.0169 [+0.0005, +0.0299]** * |

Adding the lateral raises the mean one-vs-rest AUROC on paired studies in all three
independently trained models (+0.017 to +0.023). It does not raise wF1 / F1-5 at the
cutpoint operating point. This is a within-model counterfactual, so training-seed noise
does not enter each row, and the three rows are three seeds. But it was NOT pre-declared
here, it comes from one val set, and `auroc_mean` pools 42 cells including the noisy
Uncertain class. Treat it as a candidate hypothesis for a pre-declared check, not as a
result.

**Missing-view table (Hashir style), PL 797, cross-fitted cutpoints:**

| PL | wF1 | F1-5 | macro recall | AUROC mean |
|---|---|---|---|---|
| A both − B (OFF, frontal) | +0.0031 [−0.0001, +0.0064] | **+0.0060** * | −0.0019 | **+0.0146** * |
| A frontal − B frontal | −0.0000 | +0.0032 | −0.0085 [−0.0195, +0.0003] | −0.0083 |
| A lateral − B lateral | **+0.0034** * (lower bound 0.0000) | +0.0007 | −0.0051 | +0.0045 |
| A both − A lateral | **+0.0076** * | **+0.0135** * | **+0.0237** * | **+0.0346** * |

These ON − OFF cells are single-seed, full budget. No full-budget noise bar exists. Against
the pilot-budget views_2 NB (wF1 0.0029, F1-5 0.0047, macro recall 0.0071, AUROC 0.0360),
only F1-5 +0.0060 lies beyond it, and AUROC +0.0146 is well inside. Not a claim.

**Single frontal studies (F1, 930), A − B:**
- Aggregates are flat: wF1 +0.0019, F1-5 +0.0029, AUROC +0.0015.
- Pleural Effusion AUROC Pos −0.0054 [−0.0103, −0.0002] *, AUROC Unc −0.0274 * (single
  seed).

**All val (1,808), A − B:**
- wF1 +0.0026 [+0.0004, +0.0048] * (pilot NB 0.0027: within).
- F1-5 +0.0045 [+0.0000, +0.0088] * (NB 0.0031: beyond, at the zero boundary).
- AUROC +0.0107 * (NB 0.0276: within).

**Proportion curve (PL, A's `both` cutpoints fixed, 20 draws per p).** The metrics rise
roughly linearly from p = 0 to p = 1:
- wF1 0.9036 → 0.9063;
- F1-5 0.8830 → 0.8871;
- AUROC mean 0.7773 → 0.8002;
- Pleural Effusion AUROC Pos 0.9543 → 0.9599.

B (OFF) reference on the same studies: 0.9032 / 0.8811 / 0.7856 / 0.9540. A's AUROC
crosses B's at p ≈ 0.4.

### Reading

- **The pre-declared Pleural Effusion claim fails on val.** It cannot go in the paper as
  a finding.
- **What survives, as a hypothesis only:** the second view consistently raises AUROC
  (ranking) on paired studies across three seeds, without moving the operating-point
  metrics. This is the same pattern as step 0 on test (AUROC flat on test aggregates,
  though: −0.0005). The two splits disagree on aggregate AUROC, which is one more reason
  not to claim it.
- ON − OFF at full budget stays inside or at the edge of the pilot noise bar on every
  aggregate.
- Nothing adopted. Test not touched. No code or YAML changed.
