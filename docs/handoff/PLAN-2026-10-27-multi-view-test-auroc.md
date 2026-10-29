# PLAN 2026-10-27 — test check: does the lateral view raise AUROC on frontal+lateral studies?

Status: DONE 2026-10-27 — NOT CONFIRMED (PL `auroc_mean` −0.0001 [−0.0080, +0.0105]). Approved by the user 2026-10-27 ("a trước đi rồi xem có cần làm b không"),
including deciding by the CI alone (no noise bar exists for this comparison).
CPU only, no GPU, no new forward pass.

## Why, and the caveat that must travel with the result

The val evidence package (`PLAN-2026-10-27-multi-view-evidence-val.md`) found, NOT
pre-declared, that adding the lateral view raises `auroc_mean` on val frontal+lateral
studies in all three models (A +0.0229*, R42 +0.0193*, R17 +0.0169*), with wF1 / F1-5
flat. This plan turns that into a pre-declared hypothesis and scores it on test.

⚠ **Not a clean held-out test.** Step 0 (2026-10-25) already scored A on test with
and without the aux view: on the 1,247 two-view test studies `auroc_mean`
A − A_noaux was −0.0005. The PL subgroup below is most of those studies (val:
797 of 820). The result is therefore largely known before it is computed; it is
run so that the hypothesis is closed formally, and it must be reported with this
caveat.

## Inputs (host, already on disk)

- A_both = `~/nomv_probe_20261024/A_rescore.npz` (test, `max_aux_views=1`, has `num_views`)
- A_fr = `~/nomv_probe_20261024/A_noaux.npz` (test, `model.data.max_aux_views=0`)
- For the descriptive cutpoint rows: cutpoints fitted on ALL val, each condition its
  own (A_both on `A_val.npz`, A_fr on `A_noaux_val.npz`), `tc.fit_severity_cutpoints`.

## Subgroup

PL = test studies whose anchor is PA/AP and whose aux view (`max_aux_views=1`,
`anchor_priority [PA, AP, lateral]`) is lateral; rebuilt from
`lc.PROCESSED_TEST_CSV` exactly as the val probe did (keys = anchor annotation row).

## Reading rule (pre-declared, not to be changed after the run)

- Deciding metric: `auroc_mean` on PL, A_both − A_fr, paired bootstrap 1,000 draws,
  seed 16, 95% percentile CI.
- **CONFIRMED** if the CI lower bound > 0. **NOT CONFIRMED** otherwise.
- No noise bar exists for a within-model input ablation; the user accepted the CI
  alone. The val-side pilot NB for AUROC (views_2 0.0360) is quoted beside the result.
- Descriptive only (cannot change the verdict): wF1, F1-5, macro recall under argmax
  and under the val cutpoints; Pleural Effusion AUROC Pos; the same on all test.

## Guards (any failure = no result)

1. Both npz have identical sorted `sample_keys` and labels; n = 3,269.
2. `num_views` rebuilt from the CSV equals A_both's `num_views`.
3. A_fr equals A_both (max |Δprob| ≤ 1e-4) on every single-view study.
4. A_fr differs from A_both on PL (max |Δprob| > 1e-3).

## Execution report

2026-10-27, host checkout `fa58119`, `~/mv_test_auroc.py` (ad hoc, aggregates only),
run once, CPU, output `~/mv_evidence_20261027/test_auroc.{json,log}`. The first two
launches never reached the host (Tailscale approval links expired); nothing was
computed on them. Inputs: `A_rescore.npz` (Oct 24), `A_noaux.npz` (Oct 25),
`A_val.npz`, `A_noaux_val.npz`.

Counts: n 3,269; PL 1,196; views_2 1,247; views_1 2,022. Guards PASS (rows and
labels equal, 0 keys missing, `num_views` rebuilt = npz, single-view max |Δprob| 0.0,
PL max |Δprob| 0.65).

**Verdict: NOT CONFIRMED.** A_both − A_fr, paired bootstrap 1,000, seed 16:

| group, rule | auroc_mean | wF1 | F1-5 | macro recall | PE AUROC Pos |
|---|---|---|---|---|---|
| PL, argmax | **−0.0001 [−0.0080, +0.0105]** | −0.0056* | −0.0113* | −0.0029 | +0.0159* |
| PL, val cutpoints | −0.0001 | +0.0054* | +0.0064 [−0.0011, +0.0136] | +0.0202* | +0.0159* |
| all, argmax | +0.0040 [−0.0004, +0.0083] | −0.0027* | −0.0052* | −0.0032* | +0.0080* |
| all, val cutpoints | +0.0040 | −0.0002 | −0.0011 | +0.0009 | +0.0080* |

Reading:
- The val `auroc_mean` gain on frontal+lateral studies (+0.0229*) does **not** replicate
  on test (−0.0001), as step 0 had already shown on views_2.
- Pleural Effusion AUROC Pos is +0.0159* on test PL, but the pre-declared Pleural
  Effusion hypothesis was scored on VAL and failed there (+0.0056 [−0.0009, +0.0129]);
  the test number was already known from step 0 (+0.015) and is what the hypothesis was
  formed from. It is descriptive and cannot be claimed. Across the two splits the
  two effects swap: val shows the aggregate AUROC gain, test shows the PE gain; neither
  replicates on the other split.
- The argmax vs cutpoint rows reproduce step 0 / 0b: the aux view shifts A's operating
  point (argmax F1 worse, val-cutpoint wF1 / macro recall better on PL).
