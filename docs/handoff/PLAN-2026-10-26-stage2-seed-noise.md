# PLAN 2026-10-26 — Stage-2 seed noise: is "OFF better" beyond run-to-run variation?

Status: **DONE 2026-10-26 13:37 -- WITHIN Stage-2 seed noise: S1-a's "OFF better" is not confirmed (both
deciding metrics lean OFF, neither beyond NB/√2); see the execution report.** User chose option 2 on
2026-10-26 ("2"): add `--seed` to `run_medgemma_qlora.py` and
retrain both Stage-2 arms at another seed (~5 h GPU). Val only; test is not scored.

## Why

`PLAN-2026-10-26-stage2-fusion-on-off.md` found ON' - OFF CheXbert micro F1-14 -0.0171
[-0.0329, -0.0012] at the fixed Stage-2 seed. That CI is a bootstrap over val studies, and for
Stage 1 the same kind of CI was shown NOT to cover seed noise (`PLAN-2026-10-26-pilot-seed-noise.md`).
Stage 2 could not be rerun at another seed: the seed was hardwired, and the rerun of ON' reproduced
the old arm A on 1,360 of 1,360 reports.

## Code change (this commit)

`training/run_medgemma_qlora.py --seed`, default `fig9.SEED` (16) = bit-identical to before. It sets
the global RNG (`set_seed`: LoRA / `img_proj` init, dropout, the augmentation draws of the train
Stage-1 records), `train_fine`'s batch order and the `oracle_mix` draw. Cohorts do not change: the
train set is the first 10,000 studies whatever the seed. A non-default seed adds `train_seed` to the
cache fingerprint of augmented train records. Pinned by `tests/test_stage2_seed.py`; full host suite
rc=0 (2 skipped) with the change.

The host runs it from a NEW checkout, `~/Meta-CXR-s2seed`, so `~/Meta-CXR-v3` stays at `3b69007`.
`configs/env_config.yaml` is copied from `~/Meta-CXR-v3` (identical non-comment lines, manifest v3).
`git diff 3b69007` on `training/ stage2/ model/ mhcac/ vision_encoders/ pretraining/` and the
generation/scoring scripts: the seed flag, plus `model/lavis/data/external_eval.py` wiring in
`ReportDataset.py` that is a no-op when `external_eval` is absent (it is).

## Step G — equivalence smoke (~20 min), abort on failure

ON arm commands of S1-a with `--train-limit 200 --val-limit 20 --test-limit 10
--val-generation-limit 2`:

- G1 in `~/Meta-CXR-v3` (old code, no flag);
- G2 in `~/Meta-CXR-s2seed` with `--seed 16`;
- G3 in `~/Meta-CXR-s2seed` with `--seed 17`.

Every tensor of `adapter_model.safetensors` and `img_proj.pt` must be EQUAL in G1 and G2 (the new
code at the default seed is the old code), and must DIFFER in G3 (the flag reaches training).

## Arms

S1-a exactly (`~/s2fusion.sh`: 10,000 studies, P2 + LoRA LR 3e-4, 1 epoch, batch 2 x 8, own cutpoint
cues, beam 4 / lp 2 on all 1,360 val), plus `--seed 17`, from `~/Meta-CXR-s2seed`:

| arm | Stage-1 | seed |
|---|---|---|
| ON17 | `run_20261005_paper` | 17 |
| OFF17 | `run_20261024_nomv` (+ the OFF `--stage1-options`) | 17 |

The seed-16 arms are `~/s2fusion_20261026/gen_{ON,OFF}_beam`.

Guards (abort on any): GPU idle, `ntfs3`, no other job, `~/s2seed_20261026` absent, checkout at this
commit and clean; Stage-1 load `missing=758 unexpected=0` in both; `"seed": 17` in each
`run_manifest.json`; 1,360 written / 0 failed; all four val JSONLs aligned (same `sample_key` order
and references).

## Reading rule — fixed before any result

Same scorers as S1-a, paired bootstrap 2,000 resamples, seed 16. Deciding metrics: BERTScore and
CheXbert micro F1-14; all others descriptive.

- Seed deltas: d_ON = ON17 - ON16, d_OFF = OFF17 - OFF16.
- Noise bar NB = 2 x sqrt((d_ON^2 + d_OFF^2) / 2), the same definition as the Stage-1 pilot: a rough
  95% bar for a difference between two single-seed arms, from two draws only (true SD ~0.5x-6x).
- Δ16 = ON16 - OFF16 (S1-a), Δ17 = ON17 - OFF17, Δ̄ = (Δ16 + Δ17) / 2; the bar for Δ̄ is NB / √2.
- **OFF better beyond seed noise**: for BERTScore or CheXbert micro F1-14, Δ16 < 0, Δ17 < 0 and
  Δ̄ < -NB/√2; and for neither deciding metric Δ̄ > +NB/√2.
- **ON better beyond seed noise**: the mirror image.
- Otherwise **within Stage-2 seed noise**: S1-a's "OFF better" is not confirmed.

The S1-a CI rule applied to Δ17 alone is reported beside it. Both Stage-1 checkpoints are single
runs, so this measures Stage-2 seed noise only, not Stage-1 run-to-run variation.

## Data handling

Reports, JSONL, adapters and caches stay on the host. Only aggregates come back here.

## Execution report

### Attempt 1 — 08:00, ABORTED at step G (as designed)

G1 (old code) and G2 (new code, `--seed 16`) differed on all 478 adapter tensors (val loss 1.63044
vs 1.63389). Cause, from the logs, NOT the seed flag: the `oracle_mix` draw keys its uniform numbers on
the record `sample_key` = `stable_fingerprint({cohort_id, dicom_id})`, and `cohort_id` contains
`file_identity()` (size + **mtime_ns**) of `pretraining/configs/mimic_cxr_full.yaml`. The fresh clone
had the same bytes but mtime 2026-10-26 instead of 2026-10-15, so at seed 16 the draw differed (from
MHCAC 435 cells in G1, 429 in G2; train cache ids `4c12...` vs `bbb8...`).

Consequences, recorded:
- A fresh clone of the same commit does not reproduce a Stage-2 run bit for bit; ON' == A held only
  because both ran in the same checkout.
- In S1-a, ON and OFF already used different oracle draws (their `cohort_id`s differ by checkpoint,
  cutpoints and `--stage1-options`). Random, not a bias; seed 17 moves the draw as well, so the noise bar
  covers this source.

User go-ahead 2026-10-26 ("ok") for the fix, no code change: aborted dir moved to
`~/s2seed_20261026_abort1`; `touch -r` copied the old checkout's mtime to the new one (content already
identical); the chain gained a guard that the config's size and `st_mtime_ns` match across checkouts.
A code fix (draw independent of file mtime) would change every default result; by user decision it lives on branch `fix/oracle-mix-stable-key` (`ed0af0a`), not merged. This run does not use it.

### Attempt 2 — launched once 08:21 (host code `948964e`)

Ran 08:21 -> 13:37:48, `S2SEED_DONE`, no abort. Host not rebooted (up since 2026-10-24 17:18), `ntfs3`.

**Step G passed (08:29).** G1 (old code) and G2 (new code, `--seed 16`): best val loss
1.630438510576884 in both, 0 of 478 adapter / `img_proj` tensors differ (max abs diff 0). G3
(`--seed 17`): val loss 1.6248727560043335, 478 of 478 differ (max abs diff 8.12e-2). So the new code
at the default seed is the old code, and the flag reaches training.

**Guards, checked on the artifacts (not only the chain's log):** Stage-1 load `missing=758 unexpected=0`
in both arms (3 lines each); `"seed": 17` in each arm's `summary.json` and `run_manifest.json`;
generation 1,360 written / 0 failed in both; the four val JSONLs aligned (same `sample_key` order and
references, n = 1,360 each).

| arm | train | best val loss | generation |
|---|---|---|---|
| ON17 | 08:29 -> 09:41 | 1.19078 (ON16 1.18770) | 4,722 s |
| OFF17 | 11:01 -> 12:11 | 1.18329 (OFF16 1.18162) | 4,894 s |

OFF has the lower val loss at both seeds (no CI). Identical reports out of 1,360: ON17 vs ON16 248,
OFF17 vs OFF16 211, ON17 vs OFF17 294 (ON16 vs OFF16 274).

**Seed deltas, noise bar and ON - OFF** (val n = 1,360, paired bootstrap 2,000, seed 16; d = seed 17 -
seed 16, Δ = ON - OFF, CI only for Δ17):

| metric | d_ON | d_OFF | NB | Δ16 | Δ17 [95% CI] | Δ̄ | NB/√2 |
|---|---:|---:|---:|---:|---|---:|---:|
| **BERTScore** (decides) | +0.0009 | +0.0037 | 0.0054 | -0.0006 | -0.0035 [-0.0099, +0.0034] | -0.0020 | 0.0038 |
| **CheXbert micro F1-14** (decides) | +0.0134 | +0.0021 | 0.0192 | -0.0171 | -0.0058 [-0.0205, +0.0098] | -0.0115 | 0.0136 |
| ROUGE-L | -0.0016 | +0.0040 | 0.0062 | -0.0002 | -0.0059 [-0.0111, -0.0009] | -0.0030 | 0.0044 |
| BLEU-4 | -0.0038 | +0.0046 | 0.0084 | +0.0009 | -0.0075 [-0.0129, -0.0026] | -0.0033 | 0.0059 |
| CIDEr | -0.0217 | +0.0061 | 0.0319 | +0.0016 | -0.0262 [-0.0656, +0.0119] | -0.0123 | 0.0225 |
| lexicon CE macro F1 | -0.0036 | +0.0046 | 0.0082 | +0.0049 | -0.0032 [-0.0209, +0.0153] | +0.0009 | 0.0058 |
| CheXbert macro F1-14 | -0.0024 | +0.0034 | 0.0059 | -0.0021 | -0.0079 [-0.0268, +0.0101] | -0.0050 | 0.0041 |
| CheXbert micro F1-5 | +0.0016 | +0.0030 | 0.0048 | -0.0084 | -0.0099 [-0.0327, +0.0125] | -0.0091 | 0.0034 |
| CheXbert macro F1-5 | -0.0079 | +0.0051 | 0.0133 | -0.0038 | -0.0168 [-0.0399, +0.0057] | -0.0103 | 0.0094 |

**VERDICT (pre-declared rule): WITHIN Stage-2 seed noise -- S1-a's "OFF better" is NOT confirmed.**
Both deciding metrics have Δ16 < 0 and Δ17 < 0, but neither mean clears its bar: BERTScore -0.0020 vs
-0.0038, CheXbert micro F1-14 -0.0115 vs -0.0136. Neither favours ON. The S1-a CI rule on seed 17 alone:
not established (CheXbert micro F1-14 -0.0058, CI crosses 0).

Seed-17 per-run values (corpus level, descriptive):

| run | BLEU-1/2/3/4 | METEOR | ROUGE-L | CIDEr | BERTScore | words median | CheXbert micro P/R/F1-14 |
|---|---|---:|---:|---:|---:|---:|---|
| ON17 | 0.4290/0.3044/0.2335/0.1881 | 0.1795 | 0.3575 | 0.5087 | 0.4654 | 35 | 0.5409/0.4543/0.4938 |
| OFF17 | 0.4386/0.3125/0.2408/0.1941 | 0.1830 | 0.3634 | 0.5349 | 0.4689 | 37 | 0.5536/0.4553/0.4996 |

**Reading, beyond the verdict (descriptive, not decisive):**
- The deciding S1-a result was partly seed luck: the same ON recipe moved CheXbert micro F1-14 by
  +0.0134 between seeds, about the size of the ON - OFF gap at seed 16 (-0.0171). At seed 17 the gap is
  -0.0058 and crosses 0.
- The single-seed bootstrap CI again does not cover seed noise: ROUGE-L and BLEU-4 are significant at
  seed 17 (CI excludes 0) and flat at seed 16.
- The DIRECTION is consistent: Δ17 is negative on all 9 metrics, Δ̄ negative on 8 of 9 (lexicon CE
  +0.0009). Three non-deciding CheXbert metrics have Δ̄ beyond their NB/√2 in OFF's favour (macro F1-14
  -0.0050 vs 0.0041, micro F1-5 -0.0091 vs 0.0034, macro F1-5 -0.0103 vs 0.0094). Pre-declared as
  descriptive, so they do not change the verdict; NB rests on two draws per arm (true SD ~0.5x-6x).
- So: at this Stage-2 pilot budget, ON does not beat OFF in report generation, and if anything leans
  worse. Fusion ON gives Stage 2 no measurable benefit.

Limits: one Stage-1 run per arm (so this is Stage-2 seed noise only, not Stage-1 run-to-run variation);
two Stage-2 seeds per arm; 10,000 train studies, 1 epoch; val only, test not scored. Artifacts stay on
the host: `~/s2seed_20261026/` (adapters, JSONL, `score/`), chain log `~/s2seed.log`.
