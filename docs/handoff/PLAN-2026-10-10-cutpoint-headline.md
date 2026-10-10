# PLAN 2026-10-10 — Stage-1 headline numbers use validation-fitted cutpoints

User decision (2026-10-10): fit the per-finding thresholds, commit them, and
make the thresholded numbers the headline. Argmax (the paper's protocol) is
kept beside them as the reference.

## Rule

Per finding, two cutpoints `t1 <= t2` on `s = p_pos / (p_pos + p_neg)`:
Negative below t1, Uncertain in [t1, t2), Positive at or above t2. Fitted on
VALIDATION (`val_predictions_epoch_best.npz` of `run_20261005_paper` phase 1c,
n = 1,808) to maximise each finding's sklearn-weighted three-class F1
(`fit_severity_cutpoints`, 121-quantile grid). File:
`configs/stage1_cutpoints/run_20261005_paper.json`.

Why two cutpoints, not one: equal F1-5 (0.7491 vs 0.7492), higher weighted F1
(0.8196 vs 0.8183) and macro recall (0.4460 vs 0.4404), and it keeps the
Uncertain class alive (568 calls vs 0), in line with D-023.

What the paper does instead (Sec. V-C, Eq. 22, Fig. 11): one-vs-rest ROC per
(finding, class), threshold nearest (0, 1); used to choose the findings put in
the LLM prompt. The paper does not say which split it fitted them on (its only
ROC figure, Fig. 5, is on test) and releases no threshold file. Eq. 22 fitted on
our val gives weighted F1 0.702 / F1-5 0.637 on test -- worse than argmax.

## Execution report (2026-10-10, training host)

Code from a scratch copy (`~/cutfit_src_20261010`) of the uncommitted change;
host tests for the touched files `rc=0`, all passed (1 skip: the shipped file
did not exist yet in that copy).

```
calibrate_thresholds.py --rule cutpoints  -> ~/eval_20261010_cutpoints/run_20261005_paper.json
evaluate_stage1.py --cutpoints ...        -> ~/eval_20261010_cutpoints/test/   (rc=0, 1,000 bootstrap)
```

The fitted cutpoints are bit-identical to the "two cutpoints" rule in
`~/eval_20261005_paper/cutpoints/cutpoints_report.json` (2026-10-06).

Test, n = 3,269:

| | two cutpoints (headline) | argmax (paper protocol) | paper |
|---|---|---|---|
| weighted precision | 0.8271 [0.8231, 0.8321] | 0.8392 | 0.87 |
| weighted recall | 0.8339 [0.8300, 0.8376] | 0.7814 | 0.78 |
| weighted F1 | 0.8196 [0.8152, 0.8236] | 0.7795 | 0.73 |
| F1-5 | 0.7491 [0.7411, 0.7571] | 0.6687 | 0.701 |
| macro recall | 0.4460 [0.4419, 0.4570] | 0.4609 | -- |
| AUROC Pos / Neg / Unc | 0.7836 / 0.7780 / 0.6633 | same | Fig. 5 |

Paired vs argmax (from the 2026-10-06 cutpoint report, seed 16): weighted F1
+0.0400 [+0.0370, +0.0431], F1-5 +0.0805 [+0.0743, +0.0866], macro recall
-0.0149 [-0.0192, -0.0113], weighted precision -0.0121 [-0.0147, -0.0084].

Five findings, weighted F1, cutpoints (argmax): Atelectasis 0.692 (0.615),
Cardiomegaly 0.691 (0.486), Consolidation 0.857 (0.803), Edema 0.724 (0.690),
Pleural Effusion 0.783 (0.750).

## Not changed

Stage 2. Every recorded Stage-2 number (v3, `--cue-rule paper_thresholds`) used
the Eq. 22 thresholds for its prompt cues; there is no cutpoint cue rule, so the
NLG numbers are unaffected by this change.
