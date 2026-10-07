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

(appended by the planner from the host artifacts)
