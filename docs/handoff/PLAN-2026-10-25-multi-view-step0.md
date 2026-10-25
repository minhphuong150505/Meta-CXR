# PLAN 2026-10-25 — Multi-view step 0: does arm A actually use its auxiliary view?

Planner/executor: Claude (planning session over SSH). User go-ahead 2026-10-25.
Status: **DONE 2026-10-25 11:15** -- verdict: fusion is used and HURTS argmax F1-5 on two-view studies (report below).
Follows `PLAN-2026-10-24-multi-view-ablation.md`, whose F1 gain was the same on
single-view and two-view studies (so not attributable to the second view).
**Inference only, no training.** One evaluate-only GPU pass (~10 min) + CPU.

## Question

Is `ViewFusion` in `run_20261005_paper` (arm A) doing anything? Its residual
branches (`w_o`, last FFN Linear) start at zero and get 5 epochs of 1b + 1 of 1c.
If they stayed near zero, the multi-view design is untested rather than
refuted, and step 1 (a design change) is justified.

## Measurements

1. **Counterfactual:** score A's own `phase1c/checkpoint_best.pth` on test with
   `model.data.max_aux_views=0` (multi-view still on, so the same StreamAdapters
   run on the anchor; fusion is gated to exactly zero for every study). The
   only difference from `A_rescore.npz` (plan of 2026-10-24, G1) is the
   auxiliary view. `max_aux_views` lives in `model.data`, which no phase block
   sets, so `--options` reaches it.
2. **Weights:** RMS of each fusion block's `w_o` and `ffn.3` (zero-init) against
   `w_q` / `w_v` (random-init) in the same checkpoint.
3. **Per finding** on views_2 (views_1 as control): weighted F1 and one-vs-rest
   AUROC (Pos, Unc), A − B (multi_view false run) and A − A_noaux, paired
   bootstrap 1,000 × seed 16. Clinically, the lateral view should matter most
   for Pleural Effusion (posterior costophrenic angle), retrocardiac
   Consolidation / Atelectasis, and Cardiomegaly.

## Commands (host)

```bash
cd ~/Meta-CXR && git pull --ff-only
PY=~/.venvs/meta-cxr-stage1-311/bin/python
A=~/run_20261005_paper; AD=$A/phase1c/mimic_cxr_full_blip2; OUT=~/nomv_probe_20261024
# guards: ntfs3, GPU idle, no training process, $OUT/A_noaux.npz absent
touch $OUT/.s0_marker
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled $PY -m pretraining.train \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --options run.phase=phase1c run.phase_root=$A run.evaluate=True run.save_predictions=True \
    run.output_dir=$OUT/A_noaux run.resume_ckpt_path=$AD/checkpoint_best.pth \
    run.distributed=false run.world_size=1 model.data.max_aux_views=0 > $OUT/A_noaux.log 2>&1
grep -c "Loading evaluation weights from" $OUT/A_noaux.log        # must be 1
grep "\[test\].*complementary" $OUT/A_noaux.log                    # 0 with a complementary view (multi_view=True, max_aux_views=0)
f=$(find $A $OUT -name test_predictions_epoch_provided.npz -newer $OUT/.s0_marker | head -1); mv "$f" $OUT/A_noaux.npz
$PY scripts/compare_stage1_predictions.py --subgroups-from baseline \
  --baseline $OUT/A_rescore.npz --candidate $OUT/A_noaux.npz --output $OUT/noaux_vs_A.json > $OUT/noaux_vs_A.log
$PY ~/aux_probe.py $AD/checkpoint_best.pth $OUT/A_rescore.npz $OUT/A_noaux.npz \
  ~/run_20261024_nomv/phase1c/mimic_cxr_full_blip2/result/test_predictions_epoch_best.npz $OUT/aux_probe.json
```

`~/aux_probe.py` is an ad-hoc analysis script (numpy, sklearn, torch.load; reads
the three `.npz` and the checkpoint, prints aggregates only). Not in the repo.

## Reading rule — fixed before any result

Sanity (else STOP): on `views_1` every A − A_noaux delta is 0 and
`changed_argmax` is 0 (identical input).

On `views_2` (1,247 studies), from `noaux_vs_A.log` (it prints A_noaux − A;
flip the sign):

- **Fusion inert:** `changed_argmax` < 0.01, A − A_noaux on `auroc_mean` and
  `mean_weighted_f1_5` both with CIs crossing zero and |Δ| < 0.005, and the
  zero-init weights' RMS well below the random-init ones. → the multi-view
  design was never really trained; step 1 (design change, pilot on val) is
  justified.
- **Fusion used and helps:** A − A_noaux on `auroc_mean` or
  `mean_weighted_f1_5` > 0 with a CI excluding zero. → the second view
  contributes inside A; the novelty claim rests on this counterfactual plus the
  per-finding table (strongest if it is the lateral-sensitive findings).
- **Fusion used and hurts:** the same, significantly < 0.
- Anything else: "used, effect not measurable".

Per-finding results are descriptive (14 findings × 2 metrics, no multiplicity
correction): report them, do not select on them.

## Data handling

`.npz`, JSON and logs stay on the host. The report below carries aggregates only.

## Execution report

Host checkout `8f4dc89`, `~/step0.sh` launched once, 11:04 -> 11:15. Guards
passed (ntfs3, GPU idle, no job). Eval: `Loading evaluation weights from` = 1;
dataset `3269 studies ..., 0 with a complementary view (multi_view=True,
max_aux_views=0)`; 2.5 min.

**Sanity passed:** on `views_1` (2,022) every delta is exactly 0, changed argmax 0.

**Fusion is NOT inert.** On `views_2` (1,247): changed argmax 0.064, mean
|dprob| 0.057, p99 of the per-cell max |dprob| 0.36. Zero-init weights moved
well off zero: `w_o` RMS 7.1e-3 / 7.7e-3 / 1.8e-2 (BioViL / PubMedCLIP / Swin),
`ffn.3` 8.1e-3 / 8.8e-3 / 6.9e-3, against random-init `w_q` 1.8e-2 / 2.1e-2 /
2.8e-2 -- i.e. 25-65% of the random-init scale.

**A - A_noaux on `views_2`** (the log prints A_noaux - A; signs flipped here):

| metric | A - A_noaux |
|---|---|
| `mean_weighted_f1_5` | **-0.0116 [-0.0176, -0.0051]** * |
| `weighted_f1` | **-0.0056 [-0.0090, -0.0024]** * |
| `weighted_recall` | **-0.0078** * |
| `macro_recall` | -0.0036 [-0.0099, +0.0031] |
| `auroc_mean` | -0.0005 [-0.0091, +0.0092] |
| AUROC Pos / Neg / Unc | -0.0025 / +0.0011 / -0.0002 (all cross zero) |

A itself scores F1-5 0.7509 with its aux view and **0.7624 without it**; arm B
(no multi-view training) 0.7378. Over all 3,269 studies A - A_noaux: F1-5
-0.0052 *, weighted F1 -0.0027 *, AUROC mean +0.0040 [-0.0003, +0.0084],
AUROC Unc +0.0115 [+0.0003, +0.0227] *.

**Per finding, `views_2`, A - A_noaux** (descriptive; `*` = CI excludes 0):

| finding | wF1 | AUROC Pos | AUROC Unc |
|---|---|---|---|
| Pleural Effusion | **+0.022** * | **+0.015** * | **+0.092** * |
| Lung Lesion | **+0.021** * | -0.011 | +0.034 |
| No Finding | -0.003 | **+0.014** * | -- |
| Lung Opacity | -0.001 | +0.007 | **+0.064** * |
| Pneumonia | **-0.020** * | -0.001 | **+0.035** * |
| Consolidation | **-0.038** * | -0.014 | -0.016 |
| Atelectasis | **-0.040** * | +0.011 | +0.025 |
| Pneumothorax | **-0.011** * | -0.029 | -0.037 |
| Cardiomegaly | -0.011 | -0.008 | -0.011 |

(other findings within +-0.01 on wF1, or n too small to read).

### Reading, by the pre-declared rule

**"Fusion used and hurts"** (F1-5 significantly < 0). Two qualifications that
matter for step 1:

1. **The ranking barely moves; the operating point does.** AUROC is flat on
   aggregate, while argmax F1 drops -- concentrated in Atelectasis and
   Consolidation, whose AUROC does not fall. So the aux view mostly shifts
   probability mass across the argmax boundary for those findings rather than
   destroying information. Hypothesis, not tested.
2. **Pleural Effusion is the one finding where the lateral view helps on every
   metric** (wF1, AUROC Pos, AUROC Unc), which is the finding where a lateral
   view is clinically expected to help most (posterior costophrenic angle).
   The retrocardiac findings (Atelectasis, Consolidation) go the other way.

Also from this run: A - B on `views_1` Cardiomegaly wF1 +0.052 and `views_2`
+0.109, while A - A_noaux Cardiomegaly is -0.011: the largest A-over-B gain
does not come from the second view at all.

Caveats: test only, one checkpoint, 14 x 3 per-finding cells without
multiplicity correction; "multi-view training + single-view inference" being
best on `views_2` is a post-hoc observation on test and must not be adopted
from this table. Val predictions without aux were saved on the host
(`A_noaux_val.npz`) for a val-side check.

## Step 0b — does the harm survive calibration? (user go-ahead 2026-10-25)

Step 0 found AUROC flat but argmax F1 down, i.e. possibly an operating-point
shift. Here each arm is scored under the headline rule with **its own**
val-fitted cutpoints: A with its aux view uses the committed
`configs/stage1_cutpoints/run_20261005_paper.json` (refit on A's val first and
required to match to 1e-9); A without aux view (N) gets cutpoints fitted on a
new evaluate-only val pass with `model.data.max_aux_views=0` (~2 min GPU); arm
B uses `cutpoints_nomv.json` from the 2026-10-24 plan. Paired bootstrap 1,000 x
seed 16 on all / views_1 / views_2, metrics weighted P/R/F1, F1-5, macro
recall, plus per-finding weighted F1 for seven findings (descriptive).
Scripts: `~/step0b.sh`, `~/cut_probe.py` (ad hoc, aggregates only).

Reading rule, fixed before the run, on `views_2`, A - N under cutpoints:

- **Aux view helps after calibration:** F1-5 or weighted F1 > 0 with CI
  excluding zero, neither significantly < 0. → the novelty claim is supported
  under the headline rule.
- **Harm was an operating-point artefact:** both CIs cross zero. → aux view is
  neutral in aggregate; the per-finding Pleural Effusion result is the
  evidence to report; step 2 (per-finding gated fusion) is the next design.
- **Aux view hurts even after calibration:** either significantly < 0. → step 2.

Note: N's cutpoints also change the decision on `views_1`, so `views_1` deltas
here are a calibration difference, not an input difference.

### Step 0b — execution report (2026-10-25 11:40 -> 11:48)

Attempt 1 aborted correctly on its dataset-line guard: `run.test_splits=[val]`
was overridden by the phase1c block (`test_splits: [test]`, YAML line 1148),
so it scored test again. Its stray test `.npz` was deleted from A's result dir;
log kept as `~/step0b.attempt1.log`. Attempt 2 used
`run.phases.phase1c.run.test_splits=[val]`. A's cutpoints refit from its val
file reproduce the committed file exactly (max diff 0.0).

Headline rule, each arm its own val cutpoints, paired bootstrap 1,000 x seed 16.
A = ON with aux view, N = ON with aux view removed, B = `run_20261024_nomv`.

| | views_2 (1,247): A - N | views_2: A - B | views_1 (2,022): A - N | views_1: A - B | all: A - B |
|---|---|---|---|---|---|
| weighted F1 | **+0.0053 [+0.0014, +0.0090]** * | **+0.0093** * | **-0.0025** * | **-0.0030** * | **+0.0021** * |
| F1-5 | +0.0067 [-0.0006, +0.0132] | **+0.0146 [+0.0083, +0.0208]** * | **-0.0038** * | **-0.0084** * | +0.0008 |
| macro recall | **+0.0193 [+0.0117, +0.0275]** * | **+0.0147** * | -0.0031 | +0.0006 | **+0.0057** * |
| weighted precision | **+0.0079** * | **+0.0068** * | +0.0010 | **+0.0034** * | **+0.0054** * |
| wF1 Pleural Effusion | **+0.0162** * | **+0.0138** * | 0 | -0.0027 | +0.0041 |

Values on views_2 (A / N / B): weighted F1 0.8517 / 0.8464 / 0.8423, F1-5
0.8142 / 0.8075 / 0.7996, macro recall 0.4546 / 0.4353 / 0.4398. Per finding on
views_2, A - N, significant: Pleural Effusion +0.016, Edema +0.017, Lung Lesion
+0.019; nothing significantly negative.

**Reading, by the pre-declared 0b rule: "aux view helps after calibration"** --
weighted F1 +0.0053 excludes zero and F1-5 is not significantly negative. The
step-0 argmax harm was an operating-point artefact.

Descriptive (A - B by subgroup under cutpoints was computed but not part of the
0b rule): under the headline rule the multi-view model beats the OFF run on
two-view studies on all five metrics (F1-5 +0.0146) and LOSES on single-view
studies (F1-5 -0.0084); the A - N deltas on `views_1` are pure calibration (same
probabilities, different cutpoints), which says one cutpoint set fitted on a
mix of one- and two-view studies is mis-set for the single-view ones. That is a
hypothesis for the next step (cutpoints per view count, fitted on val), not a
result. One run per arm; test, many metrics, no multiplicity correction.

## Step 0c — cutpoints fitted per view count (user go-ahead 2026-10-25)

Hypothesis from 0b: one cutpoint set fitted on val mixing one- and two-view
studies is mis-set for single-view studies (ON loses F1-5 -0.0084 there while
its probabilities equal A_noaux's). Fit cutpoints on val SEPARATELY for
`num_views == 1` and `num_views >= 2`, for BOTH arms, and score test once.

- GPU (~3 min): evaluate-only VAL pass of arm A at HEAD with its aux view
  (`run.phases.phase1c.run.test_splits=[val]`), because A's training-time val
  npz predates `num_views`. Sanity: probabilities must match that npz
  (max |dprob| < 1e-3, changed argmax 0), else abort.
- Groups come from A's `num_views` for both arms (B ran `multi_view=false`, so
  its own field is the anchor only); rows aligned by `sample_key`, asserted.
- Per arm, per group: `fit_severity_cutpoints` (weighted F1, same grid as the
  headline file) on that group's val rows. Test rows are decided with their
  group's cutpoints; metrics computed by `evaluate_classification` on the whole
  split. 1,000 paired bootstrap draws, seed 16.

**Reading rule, fixed before any result.** Primary comparison: test, all
3,269 studies, A (ON) per-view cutpoints minus B (OFF) per-view cutpoints.
ON WINS iff the 95% CI of F1-5 OR weighted F1 lies above zero AND none of
weighted P / R / F1, F1-5, macro recall has a CI below zero. Anything else:
"per-view cutpoints do not make ON beat OFF". Descriptive only (no claim):
views_1 / views_2 subgroups; A per-view minus A single (does it fix views_1);
B per-view minus B single. A win does NOT change the headline rule by itself;
adopting per-view cutpoints is a separate user decision.
