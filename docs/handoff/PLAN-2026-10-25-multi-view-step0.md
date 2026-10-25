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
