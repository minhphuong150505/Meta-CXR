# PLAN 2026-10-07 — Retrain Stage 2 (paper mode) on targets without exam headers

Planner: Claude (dev box). Executor: `~/stage2_v3_chain.sh` on the training host.

## Why

`stage2_paper_20261006b` (D-027 paper mode, 1 full epoch, `best_val_loss`
1.048) scored, with the paper's own evaluator (`MIMICEvalCap`: strip newlines and
`<s>`, nltk `word_tokenize`, lowercase, pycocoevalcap corpus BLEU/METEOR/ROUGE-L,
plus CIDEr and BERTScore distilroberta rescaled), n = 2,984 test studies:

| | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paper (Tables 2-3) | 0.390 | 0.255 | 0.175 | 0.102 | 0.173 | 0.280 | 0.291 | 0.426 |
| `20261006b` | 0.056 | 0.034 | 0.023 | 0.017 | 0.065 | 0.178 | 0.137 | 0.073 |

**1,709 of 2,984 generations (57%) are a bare exam header** ("AP CHEST, 10:11
A.M.,", <= 6 words) and nothing else; median generated length 4 words against a
reference 55; 454 distinct outputs. Not truncation: `clean_text` does not split
lines. Hypothesis-length / reference-length = 0.315, so BLEU's brevity penalty
alone multiplies every BLEU by ~0.11. On the 1,275 non-header generations
(diagnostic only, selected on the output): ROUGE-L 0.270, BERTScore 0.387.

Cause, measured on `full_allviews_v2` train: **9.8% of valid targets start with
an exam header and 3.4% are nothing else**, almost all via the
`NARRATIVE_BODY` fallback (reports with no FINDINGS tag). Some also carried the
`REASON FOR EXAM` line.

## Fix (commit 83bf724)

`mimic_report_parser.is_exam_header` / `_strip_exam_header` drop chest
exam-description lines (no lowercase, names CHEST, every other word is
projection/view vocabulary; clock times, `A.M.`, `___`, digits ignored) and keep
the content after a `<HEADER>:` prefix; `reason for exam` is an `indication`
alias. Old vs new over all 227,835 reports: 20,711 targets change, 6,177 become
empty (-> `IMPRESSION_ONLY`, masked from generation), 9 still start with a
header. Every target losing > 15 words lost the REASON FOR EXAM text; of 173
reports with text directly under that line (p10-p11), 171 are a wrapped reason
(line >= 60 chars, 1-2 short continuation lines), 2 have a > 20-word block.

## Isolation

- Manifest rebuilt to `/home/phuong/data/processed/full_allviews_v3`
  (`preprocess_mimic_cxr.py` defaults, same raw inputs). `full_allviews_v2` on the
  NTFS drive is untouched.
- Stage 2 runs from a **separate checkout `~/Meta-CXR-v3`** whose untracked
  `configs/env_config.yaml` points `processed_dir` at v3. `~/Meta-CXR`, run B
  (`run_20261007_aux3`) and every later Stage-1 run keep v2. Only the Stage-2
  targets move: same Stage-1 checkpoint (`run_20261005_paper` 1c best), same
  Eq. 22 thresholds, same flags as `20261006b`.

## Chain — `~/stage2_v3_chain.sh`

1. waits for `multiaux_chain.sh` (run B) and `multiaux_env_off.sh` to exit;
2. guards: no GPU job, 0 compute apps, `ntfs3`, checkpoint + thresholds present,
   `$OUT` absent, >= 25 GB free on `/home`; v3 CSVs exist, header-start rate in
   valid targets <= 0.2% on every split; checkout contains `83bf724`;
3. points the watchdog at the run (`export EXPECT_RUNNING=1`);
4. smoke (200/20/10) then full run into `~/stage2_paper_20261007_v3/`, adapter
   check (`complete`, `paper`, r 8, alpha 16);
5. paper-protocol scoring into `~/stage2_paper_eval_20261007/test_paper_protocol_v3.json`,
   then `EXPECT_RUNNING=0`.

Expected time: B ends ~2026-10-07 23:00; the previous full run took ~22 h
including the Stage-1 record pass and test generation, so ~2026-10-08 late.

## Caveats to report with the result

- v3 drops 6,177 header-only targets, so the **test cohort differs** from
  `20261006b` (fewer studies with `generation_mask`). Compare on the
  intersection by `sample_key` as well as on each run's own cohort.
- `preprocess_mimic_cxr.py` recomputes the train length bounds, so a few
  targets near the bounds may flip validity for that reason alone.
- Greedy decoding, `max_new_tokens` 256 -- unchanged from `20261006b`.

## Execution report

Appended by the planner from the host artifacts, 2026-10-09. Aggregates only.

The chain completed: smoke, full run into `~/stage2_paper_20261007_v3/full/`,
then paper-protocol scoring. Training eval loads NF4 (`quantize_4bit=True`);
`generate_stage2_reports.py` loads bf16, which changes generations (81/300
identical on val), so every probe below that compares with training eval
uses NF4.

### Test, paper protocol, n = 2,800

| | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paper | 0.390 | 0.255 | 0.175 | 0.102 | 0.173 | 0.280 | 0.291 | 0.426 |
| `20261006b` (v2) | 0.056 | 0.034 | 0.023 | 0.017 | 0.065 | 0.178 | 0.137 | 0.073 |
| v3 greedy, 256 tok | 0.285 | 0.168 | 0.107 | 0.073 | 0.120 | 0.250 | 0.084 | 0.361 |
| v3 beam 4, lp 2.0 | 0.357 | 0.213 | 0.140 | 0.098 | 0.137 | 0.265 | 0.128 | 0.369 |

Header-only generations: 0 (was 1,709). Greedy: median 37 words vs reference
56, 1,473 distinct outputs. Beam: 47 words, 1,536 distinct.
Artifacts: `~/stage2_paper_eval_20261007/test_paper_protocol_v3.json`,
`~/stage2_v3_test_beam_20261009/{test_paper_protocol_beam.json,paired.txt}`.

Beam − greedy, paired bootstrap (2,000, seed 16), same 2,800 studies:
BERTScore +0.0083 [+0.0037, +0.0129], ROUGE-L +0.0150 [+0.0112, +0.0188],
CIDEr +0.0439 [+0.0230, +0.0653]. BERTScore P/R 0.424/0.299 -> 0.390/0.349.

⚠ Beam is NOT in `generate_stage2_reports.py`: `SoftTokenEmbeddingWrapper`
validates per row, so the projected soft tokens must be `repeat_interleave`d by
`num_beams`. The probe (`~/len_v3.py`, `~/test_beam.py` on the host) wraps
`img_proj` to do that and monkeypatches `generate`; 0 failures. Decoding was
chosen on val 300, then run once on test.

### BERTScore diagnosis (greedy, test)

P 0.424 (paper F 0.426), R 0.299: the model omits content rather than writes
wrong content. Baselines on the same refs: shuffled refs 0.308, a random
other reference 0.281, the single most common prediction for every study
0.339. The model is only +0.053 above shuffled and +0.022 above a constant.

### Input ablation (val 300, bf16, derangement donor)

| arm | BERTScore delta | CIDEr delta | identical to full |
|---|---|---|---:|
| swap soft tokens | -0.072 [-0.094, -0.050] | -0.143 | 23/300 |
| swap cues (P/N/U + cue state) | -0.014 [-0.025, -0.003] | CI touches 0 | 76/300 |

The soft tokens carry image information; the cues add little.

### Length probe (val 300, NF4)

| arm | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore (P/R) | words |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy | 0.315 | 0.093 | 0.132 | 0.286 | 0.190 | 0.411 (0.468/0.355) | 27.5 |
| greedy, `min_new_tokens=60` | worse: BERTScore -0.018, ROUGE-L -0.013, CIDEr -0.071, all significant | | | | | | |
| beam 4 | 0.377 | 0.118 | — | 0.295 | 0.200 | 0.410 | — |
| beam 4, lp 2.0 | 0.400 | 0.124 | 0.148 | 0.296 | 0.205 | 0.411 (0.406/0.417) | 39 |

Beam 4 + lp 2 vs greedy on val: BERTScore/ROUGE-L/CIDEr CIs all contain 0.
Beam fixes length and surface wording (BLEU brevity penalty), not content.

### Zero-shot MedGemma under the same protocol

`~/thesis_eval_20260913/gen_zeroshot/generated_test.jsonl` (native image, no
adapter, 160 tokens) rescored with the paper evaluator. Its `sample_key`s do not
match the v3 run, so the two were joined on reference text, keeping references
unique on both sides: n = 2,159, scored against the v3 references.

| | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore (P/R) | words |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero-shot | 0.334 | 0.077 | 0.150 | 0.250 | 0.075 | 0.332 (0.334/0.329) | 61 |
| v3 greedy | 0.284 | 0.075 | 0.122 | 0.254 | 0.082 | 0.371 (0.436/0.308) | 36 |
| v3 beam | 0.364 | 0.104 | 0.141 | 0.273 | 0.126 | 0.380 (0.400/0.361) | 47 |

Beam − zero-shot: BERTScore +0.0486 [+0.0440, +0.0538], ROUGE-L +0.0229
[+0.0189, +0.0271], CIDEr +0.0509 [+0.0360, +0.0667]. Greedy − zero-shot:
BERTScore +0.0394 [+0.0343, +0.0445], ROUGE-L +0.0043 [+0.0006, +0.0079],
CIDEr +0.0064 [-0.0071, +0.0192]. Zero-shot wins only METEOR (length/recall).
On its own 2,772-study cohort with v2 references: BLEU-1 0.316, BLEU-4 0.071,
METEOR 0.147, ROUGE-L 0.235, CIDEr 0.075, BERTScore 0.279. The older zero-shot
figures (METEOR 0.26, BERTScore raw 0.81) come from `evaluate_stage2.py`'s own
METEOR and raw BERTScore and are a different scale.
Artifacts: `~/stage2_v3_zs_compare_20261009/out.txt`.

### Caveats

One run, one seed; decoding chosen on val; lexical metrics only (no clinical
labeler in this repo); the zero-shot comparison is on a text-joined subset.
