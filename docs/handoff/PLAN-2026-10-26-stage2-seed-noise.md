# PLAN 2026-10-26 — Stage-2 seed noise: is "OFF better" beyond run-to-run variation?

Status: **user chose option 2 on 2026-10-26 ("2"): add `--seed` to `run_medgemma_qlora.py` and
retrain both Stage-2 arms at another seed (~5 h GPU).** Val only; test is not scored.

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
