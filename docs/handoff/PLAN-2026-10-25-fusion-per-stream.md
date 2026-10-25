# PLAN 2026-10-25 — which encoder stream's fusion does the work? (inference only, val)

Status: **DONE 2026-10-25 22:31 -- the Swin stream carries the fusion gain; BioViL adds AUROC only (report below).** User go-ahead 2026-10-25 ("làm ý 1"). Inference only: no training, no
code or YAML change in the repo, test is never scored. ~10–15 min GPU + ~10 min CPU.

## Why

Fusion runs once per encoder stream (`Blip2Qformer._fuse`, one `ViewFusionModule`
each for biovil / pubmedclip / swin). The BioViL stream reaches fusion through
`_biovil_tokens` with the default `legacy_reshape`, so its 196 "tokens" are groups
of channel maps, not patches (CLAUDE.md, "The BioViL stream is mis-read"). An
anchor-to-aux cross-attention over such tokens has no spatial correspondence to
use. Hypothesis H: **BioViL-stream fusion carries nothing measurable; the
two-view gain comes from the PubMedCLIP and/or Swin streams.** If H holds, the next
training proposal is fusion on those streams only (or BioViL fusion on patch tokens).
That proposal would need its own confirmation.

## What runs

`run_20261005_paper/phase1c/checkpoint_best.pth`, scored on val (1,808 studies, 820
with an aux view) with the same evaluate-only command as step 0c's `A_val`. The
command runs through `~/fusion_stream_eval.py`, which patches `Blip2Qformer._fuse` in
memory so that the streams listed in `$FUSION_SKIP` return their anchor tokens. The
StreamAdapter is still applied, exactly as for a study with no aux view. The other
streams fuse as trained.

| arm | fusion off in |
|---|---|
| F | none (must reproduce step 0c's `A_val.npz`) |
| noBV | biovil |
| noPMC | pubmedclip |
| noSW | swin |
| N | all three (must reproduce step 0b's `A_noaux_val.npz`) |

The single-stream arms are not fully out of distribution. Training draws the aux
drop independently per encoder (`p_view_drop` 0.15), so "this stream unfused, the
other two fused" is the input of ~11% (0.15 × 0.85²) of two-view training studies.

Scripts: `~/fusion_stream.sh` (chain, launched once), `~/fusion_stream_eval.py`
(patch) and `~/fusion_stream_probe.py` (reading). These are ad hoc host files, not
in the repo.

## Guards (any failure → stop; nothing is a result)

- Each arm: GPU idle, `ntfs3`, no other job, `Loading evaluation weights from` ×1,
  dataset line `[val] 1808 studies ... 820 with a complementary view`, no `[test]`
  dataset, no Traceback, no new test `.npz`.
- Each skipped stream is reported skipped with an aux view present
  (`FUSION_SKIP_FIRST <stream>`), and F skips nothing.
- max |Δp|: F vs `A_val.npz` ≤ 1e-4; N vs `A_noaux_val.npz` ≤ 1e-4. On views_1
  (input identical) every arm vs F ≤ 1e-4. On views_2 every off-arm differs from
  F (> 1e-3).

## Reading rule — fixed before any result

Cutpoints are cross-fitted on val exactly as in the aux-drop pilot: rows sorted by
`sample_key`, fold = `default_rng(16).permutation(n) % 2`, each arm fits its own.
Metrics: weighted P / R / F1, F1-5, macro recall, plus `auroc_mean` (no cutpoints).
Paired bootstrap, 1,000 draws, seed 16, on all / views_1 / views_2. **Decisions use
views_2 only.** views_1 differences come only from calibration (same inputs).

For each comparison c ∈ {F − N, F − noBV, F − noPMC, F − noSW} on views_2:

- **helps**: the CI of weighted F1 or F1-5 is above 0, and none of weighted F1 /
  F1-5 / `auroc_mean` has a CI below 0;
- **hurts**: the CI of weighted F1 or F1-5 is below 0, and none of the three has a
  CI above 0;
- otherwise **not separable at this n**.

Verdict:

- **H supported**: F − noBV is "hurts", or "not separable" with both the weighted F1
  and F1-5 point estimates ≤ 0; AND F − noPMC or F − noSW is "helps".
- **H refuted**: F − noBV is "helps".
- Otherwise **inconclusive**: the per-stream question cannot be answered by
  inference at val n = 820.

Nothing is adopted from this probe. An inference-time "fusion off in one stream"
model is not a candidate here. Making it one would be a separate decision, scored
on test once.

## Data handling

`.npz`, JSON and logs stay on the host. Only aggregates come back here.

## Execution report

Launched once 22:17 (host code `3c8256a`), five val passes 22:17 -> 22:25 (~1.5 min
each), probe done 22:31; GPU idle afterwards. The probe was first dry-run on CPU with
existing files (every "arm" = A without aux view): sanity PASS, the four comparisons
identical, as they must be.

**Guards: all passed.** Weights line x1 and the 820-complementary-view dataset line in
every log, no test split built or scored. The wrapper skipped each listed stream in
all 199 val batches that carried an aux view, and F skipped nothing. max |Δp| was
F vs `A_val` 0, N vs `A_noaux_val` 0, and 0 on views_1 for every arm (bit-identical).
On views_2 the off-arms changed probabilities by up to 0.61 (BioViL) / 0.32
(PubMedCLIP) / 0.76 (Swin) / 0.75 (all).

### Result — views_2 (820 two-view val studies), cross-fitted cutpoints

Values (wP / wR / wF1 / F1-5 / macro recall / AUROC mean):

| arm | views_2 |
|---|---|
| F (all fused) | 0.9022 / 0.9130 / 0.9035 / 0.8830 / 0.4538 / 0.7968 |
| noBV | 0.9020 / 0.9137 / 0.9037 / 0.8836 / 0.4538 / 0.7870 |
| noPMC | 0.8990 / 0.9130 / 0.9034 / 0.8833 / 0.4547 / 0.7944 |
| noSW | 0.8993 / 0.9121 / 0.8991 / 0.8760 / 0.4430 / 0.7806 |
| N (none fused) | 0.8992 / 0.9127 / 0.8994 / 0.8791 / 0.4432 / 0.7749 |

Paired bootstrap 1,000 x seed 16, * = CI excludes 0:

| views_2 | wF1 | F1-5 | macro recall | AUROC mean | wP | label |
|---|---|---|---|---|---|---|
| F - N (all fusion) | **+0.0041 [+0.0007, +0.0078]** * | +0.0039 [-0.0021, +0.0102] | **+0.0106** * | **+0.0219 [+0.0076, +0.0346]** * | +0.0030 | helps |
| F - noBV | -0.0001 [-0.0026, +0.0020] | -0.0006 [-0.0047, +0.0033] | -0.0000 | **+0.0097 [+0.0012, +0.0142]** * | +0.0002 | not separable |
| F - noPMC | +0.0001 [-0.0013, +0.0017] | -0.0003 [-0.0028, +0.0025] | -0.0009 | +0.0024 [-0.0023, +0.0062] | **+0.0032** * | not separable |
| F - noSW | **+0.0045 [+0.0011, +0.0079]** * | **+0.0070 [+0.0009, +0.0128]** * | **+0.0107** * | **+0.0162 [+0.0039, +0.0273]** * | +0.0029 | helps |

All val (1,808), descriptive: F - N wF1 +0.0017 [-0.0002, +0.0036], AUROC +0.0130 *.
On views_1, AUROC deltas are exactly 0 (identical inputs). The other views_1 deltas are
calibration only (each arm fits its cutpoints on all of val).

**VERDICT by the pre-declared rule: H SUPPORTED.** F - noBV is "not separable" with the
wF1 and F1-5 point estimates <= 0, and F - noSW is "helps".

**Reading.**
- **The Swin stream carries the whole operating-point gain of fusion on two-view
  studies.** Turning off Swin fusion alone removes it: noSW ends at N's wF1 / F1-5 /
  macro recall.
- **The verdict's wording "BioViL fusion carries nothing measurable" is too strong.**
  Removing BioViL fusion costs AUROC +0.0097 *. BioViL fusion improves ranking but adds
  nothing to wF1 / F1-5 at the cutpoint operating point. The rule ignored AUROC in its
  "not separable" branch; that is recorded here, not changed after the fact.
- **PubMedCLIP fusion adds nothing measurable except weighted precision**, although its
  tokens are real patches plus CLS. So "only BioViL's channel-group layout is the
  problem" does not explain everything.
- **Consistent with step 0's weights.** The Swin block's zero-init `w_o` moved the
  most (RMS 1.8e-2 vs 7.1e-3 BioViL, 7.7e-3 PubMedCLIP).

**Limits.**
- Inference only, one checkpoint, val n = 820.
- The 24 cells (6 metrics x 4 comparisons) carry no multiplicity correction.
- Nothing is adopted. Test was not touched.
