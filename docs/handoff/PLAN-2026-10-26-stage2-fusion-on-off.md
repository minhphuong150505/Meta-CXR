# PLAN 2026-10-26 — does fusion ON beat OFF in Stage 2 report generation?

Status: **user go-ahead 2026-10-26 for S1-a ("cách a"): both arms retrained; S2 still needs its own confirmation.** The user asked (2026-10-26) to "train and
recompute the Stage-2 NLG metrics, like the earlier report" after the seed-noise run. Read here as:
compare Stage 2 trained on Stage-1 ON (`run_20261005_paper`) vs Stage-1 OFF (`run_20261024_nomv`),
which is how EVOKE (arXiv 2411.10224) measured multi-view, on report metrics. No repo code or YAML
change.

## Step S1 — val pilot

⚠ Checked on the host 2026-10-26: `~/qrefit2_20261015/pilot_A` (A's adapter and record cache) **no
longer exists**; only `gen_A_beam/generated_val.jsonl` (1,360 reports) and its scores remain. So A
cannot be regenerated to prove it is reproducible. Two options for the user:

- **S1-a (recommended, ~5.3 h GPU):** train BOTH arms again in one chain (ON' then OFF), same commands.
  The paired comparison is ON' − OFF. Comparing ON' with the old A is a free check: identical reports
  mean the pipeline is deterministic; differences are one sample of Stage-2 run-to-run noise.
- **S1-b (~2.7 h GPU):** train OFF only and use A's surviving reports as ON. The code is identical
  (same checkout, same commit), but the environment since 2026-10-16 cannot be verified.

The arms below are written for S1-a; under S1-b the ON arm is not run.

- **ON arm:** the recipe of arm A of the 2026-10-16 refit pilot. That arm used the `run_20261005_paper/phase1c`
  soft tokens, the first 10,000 train studies, P2 (`paper_context`, `oracle_mix`, share 0.2) +
  LoRA LR 3e-4, 1 epoch (476 updates), batch 2 x 8, cues `cutpoints` from the committed
  `configs/stage1_cutpoints/run_20261005_paper.json`, checkout `~/Meta-CXR-v3` at `3b69007`, and
  manifest v3. Val: all 1,360 studies, beam 4 / lp 2, max 256 new tokens.
- **OFF arm: new.** Identical commands, in the same checkout at the same commit (unchanged since
  `3b69007`). Only two things differ:
  - `--stage1-checkpoint ~/run_20261024_nomv/phase1c/mimic_cxr_full_blip2/checkpoint_best.pth`
    with `--stage1-options model.multi_view=false model.loss.lambda_mpc=0.0
    model.loss.lambda_view_consistency=0.0`. Stage 2 builds the Stage-1 model from the base
    `model:` block (no phase is applied), so these options reach it.
  - Cues from OFF's own val-fitted cutpoints, `~/nomv_probe_20261024/cutpoints_nomv.json`.

Guards (abort on any):
- GPU idle, `ntfs3`, the checkout is at `3b69007`, and the output dir is absent.
- S1-a: ON' Stage-1 load prints the same `missing=758 unexpected=0` as A did.
- OFF Stage-1 load prints `unexpected=0`.
- The OFF val JSONL has the same `sample_key` list, in the same order and with the same references,
  as ON's (and as A's).
- 0 generation failures.

## Metrics — the same scorers as every earlier Stage-2 result

- `scripts/score_stage2_paper_protocol.py`: BLEU-1..4, METEOR, ROUGE-L, CIDEr, BERTScore (paper
  setting), lexicon CE.
- `scripts/score_ce_chexbert.py`: CheXbert micro / macro F1-14 and F1-5.
- EVOKE's set is covered except RadGraph F1, which this repo does not implement (reported as
  unavailable, never as 0).

## Reading rule — fixed before any result

Paired bootstrap, 2,000 draws, seed 16, ON − OFF on the 1,360 val studies.

- **ON better in Stage 2**: the CI of BERTScore OR of CheXbert micro F1-14 is above 0, and neither
  has a CI below 0.
- **OFF better**: the mirror image.
- Otherwise **not established**.

Every other metric is reported beside it, descriptive. A bound within 0.0005 of zero is "at the
zero boundary". One seed per arm; Stage-2 seed noise is not measured.

## Step S2 — only if S1 says "ON better" (separate confirmation, ~48 h + test)

Full Stage 2 (2 epochs) on OFF with the reported recipe. Then test (n = 2,800) once, against the
reported Stage 2 `stage2_p2_20261011` (built from `run_20261005_paper`). The code commit must match
that run's (`ce18c45`) or be shown not to touch Stage 2.

## Data handling

Generated reports, JSONL, adapters and caches stay on the host. Only aggregates come back here.

## Execution report

Chain `~/s2fusion.sh` (verdict `~/s2fusion_verdict.py`, dry-run on the 2026-10-16 B/A scores reproduced
their numbers and "not established") launched once; it waits for `~/seednoise.log` to end and the GPU to
be released before training.
