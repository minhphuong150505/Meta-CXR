# PLAN 2026-10-26 — does fusion ON beat OFF in Stage 2 report generation?

Status: **DONE 2026-10-26 -- S1-a verdict: OFF better than ON in Stage 2 (CheXbert micro F1-14 -0.0171
[-0.0329, -0.0012]; BERTScore flat). S2 not triggered.** ⚠ **Superseded by the seed replicate (2026-10-26,
`PLAN-2026-10-26-stage2-seed-noise.md`): at seed 17 the gap is -0.0058 [-0.0205, +0.0098]; the two-seed
mean -0.0115 does not clear the Stage-2 noise bar (0.0136), so "OFF better" is WITHIN seed noise. ON still
does not beat OFF.** User go-ahead 2026-10-26 for S1-a ("cách a"). The user asked (2026-10-26) to "train and
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

**DONE 2026-10-26 07:06 -- VERDICT (pre-declared rule): OFF BETTER than ON in Stage 2.** S2 is not
triggered.

- ON' trained 01:59 -> 03:10, beam 03:10 -> 04:30; OFF trained 04:30 -> 05:40, beam 05:40 -> 07:03. ~5.1 h.
- Guards: ON' Stage-1 load `missing=758 unexpected=0` (same as A); OFF `missing=758 unexpected=0`; 1,360
  written / 0 failed in both arms; ON', OFF and A JSONLs aligned (same `sample_key` order and
  references); checkout `3b69007`, clean. Bootstrap 2,000 resamples, seed 16, as declared.
- Val loss (teacher-forced, no CI): ON' 1.18770, OFF 1.18162.
- **Rerun check: ON' is identical to the old arm A on 1,360 of 1,360 reports** (every delta exactly 0).
  Stage 2 is deterministic at its fixed seed (`fig9.SEED`; `run_medgemma_qlora.py` has no `--seed`
  flag). So this check gives NO sample of Stage-2 run-to-run noise; Stage-2 seed noise is still
  unmeasured.

### ON' - OFF, val n = 1,360, beam 4 / lp 2, paired bootstrap (per-study means)

| metric | ON' - OFF | 95% CI |
|---|---|---|
| BERTScore (decides) | -0.0006 | [-0.0073, +0.0061] |
| **CheXbert micro F1-14 (decides)** | **-0.0171** | **[-0.0329, -0.0012]** |
| CheXbert macro F1-14 | -0.0021 | [-0.0225, +0.0176] |
| CheXbert micro F1-5 | -0.0084 | [-0.0324, +0.0154] |
| CheXbert macro F1-5 | -0.0038 | [-0.0326, +0.0254] |
| ROUGE-L | -0.0002 | [-0.0055, +0.0052] |
| BLEU-4 (per study) | +0.0009 | [-0.0047, +0.0066] |
| CIDEr | +0.0016 | [-0.0364, +0.0398] |
| lexicon CE macro F1 | +0.0049 | [-0.0137, +0.0246] |

274 of 1,360 reports are word-for-word identical between the arms. The deciding upper bound (-0.0012)
is 0.0012 from zero, so it is not "at the zero boundary" by the 0.0005 rule, but it is close.

### Per-run values (corpus level, descriptive)

| run | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore | words median |
|---|---|---|---|---|---|---|---|---|---|
| ON' (= A) | 0.4300 | 0.3055 | 0.2353 | 0.1904 | 0.1782 | 0.3591 | 0.5304 | 0.4645 | 35 |
| OFF | 0.4398 | 0.3117 | 0.2388 | 0.1916 | 0.1821 | 0.3594 | 0.5288 | 0.4651 | 37 |

| run | CheXbert micro P / R / F1-14 | macro F1-14 | micro F1-5 | macro F1-5 | lexicon CE |
|---|---|---|---|---|---|
| ON' | 0.5245 / 0.4431 / 0.4804 | 0.3699 | 0.5191 | 0.4539 | 0.316 |
| OFF | 0.5446 / 0.4579 / 0.4975 | 0.3720 | 0.5275 | 0.4577 | 0.311 |

RadGraph F1: unavailable (not implemented in this repo), not 0.

### Reading

- By the rule fixed before the run, OFF is better: CheXbert micro F1-14 is lower for ON' (both
  precision and recall), and no metric favours ON' with a CI above 0. Every other metric crosses zero.
- Limits: one seed per arm in Stage 2, and Stage-2 seed noise is unmeasured (the rerun was bit-identical
  by construction). The two Stage-1 checkpoints are single full runs each, so this compares two
  Stage-1 runs, not fusion alone; at the pilot budget Stage-1 seed noise moved val AUROC by ~0.015
  (`PLAN-2026-10-26-pilot-seed-noise.md`). Pilot budget (10,000 studies, 476 updates), val only.
- Descriptive parallel, different setup: EVOKE (arXiv 2411.10224, Table III) also reports multi-view
  input lowering CheXbert F1 unless MPC is added; here ON' has MPC and still loses on micro F1-14.
- S2 (full Stage 2 on OFF + test) is not run, as pre-declared.
