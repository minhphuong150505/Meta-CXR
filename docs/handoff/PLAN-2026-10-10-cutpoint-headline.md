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

## Follow-up 2026-10-10/11 — `--cue-rule cutpoints` on the v3 Stage-2 test set

User asked for a cutpoint cue rule and a test regeneration (commit `eb89324`).
Same v3 adapter (trained with `paper_thresholds` cues), same 2,800 test
studies, NF4, 256 tokens; only the P/N/U cues change. Host probe
`~/cutcue_test.py` + `~/cutcue_run.sh` (from `~/Meta-CXR-v3` at `eb89324`),
output `~/stage2_v3_test_cutcue_20261010/`, `CUTCUE_ALL_DONE`, 0 failures.

Checks before scoring: soft tokens bit-identical 2,800/2,800; control arm (old
cues, greedy, first 100 studies) reproduced the v3 training-eval outputs
100/100, so greedy is like for like. Cue groups changed on 2,557/2,800 studies;
mean P/N/U listed per study 4.37 / 7.07 / 1.53 (Eq. 22) -> 1.99 / 10.84 / 0.17
(cutpoints, every finding listed).

Paper protocol, test n = 2,800:

| | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore |
|---|---:|---:|---:|---:|---:|---:|
| v3 greedy (Eq. 22 cues) | 0.285 | 0.073 | 0.120 | 0.250 | 0.084 | 0.361 |
| greedy, cutpoint cues | 0.271 | 0.069 | 0.117 | 0.247 | 0.086 | 0.363 |
| v3 beam 4, lp 2 (Eq. 22 cues) | 0.357 | 0.098 | 0.137 | 0.265 | 0.128 | 0.369 |
| beam 4, lp 2, cutpoint cues | 0.351 | 0.097 | 0.135 | 0.265 | 0.132 | 0.370 |

Paired per-study bootstrap (2,000, seed 16), cutpoints minus Eq. 22:
greedy BERTScore +0.0016 [-0.0012, +0.0044], ROUGE-L -0.0030 [-0.0051, -0.0009],
CIDEr +0.0021 [-0.0069, +0.0115], BLEU-4 -0.0031 [-0.0050, -0.0012];
beam BERTScore +0.0010 [-0.0015, +0.0035], ROUGE-L +0.0004 [-0.0016, +0.0024],
CIDEr +0.0040 [-0.0073, +0.0157], BLEU-4 +0.0007 [-0.0015, +0.0030].
Identical outputs: greedy 651/2,800, beam 730/2,800. Median words 37 -> 34
(greedy), 47 -> 46 (beam).

Verdict: better Stage-1 decisions do NOT improve generation at inference.
Beam (the reported decoding) is flat on every metric; greedy loses a little
ROUGE-L and BLEU-4. Consistent with the v3 input ablation (swapping cues
between studies moves BERTScore only -0.014). Caveat: the adapter was trained
with Eq. 22 cues, so this is a train/test cue mismatch; a Stage-2 retrain with
cutpoint cues (~22 h) is the untested fair version. Reported Stage-2 numbers
stay the v3 Eq. 22 ones.
