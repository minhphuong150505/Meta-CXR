# Matched Stage-2 evaluation

## Goal
Score existing Arm A and Arm C reports on the same 2,772 test studies.
No regeneration or training. Preserve existing artifacts.

## Preconditions
Training host checkout pulled with `git pull --ff-only`; clean, already current.
GPU idle; no train/evaluation process found. Use the approved Python 3.11 venv.

## Commands
Deploy `/tmp/meta_cxr_matched_eval_20260908.py` to the same path on the host.
Launch once with `setsid nohup /home/phuong/.venvs/meta-cxr-stage1-311/bin/python -u /tmp/meta_cxr_matched_eval_20260908.py`.
The script creates `/home/phuong/eval_matched_20260908` exclusively, checks
unique sample keys, exact cohort overlap and identical references, then runs
`scripts/evaluate_stage2.py` sequentially for A and C with
`--metrics bleu,rouge,meteor,cider,bertscore --skip-clinical-metrics
--bootstrap-samples 1000 --split test`. BERTScore retains default CPU/device
and model settings used in prior evaluations.

## Expected
2,772 matching records, zero reference mismatches, both evaluator exits 0,
metrics artifacts for both arms. Report clinical metrics as unmeasured.

## Abort if
Duplicate keys, mismatched references, unexpected cohort size, existing output
directory, busy GPU, or evaluator failure. Do not overwrite or retry blindly.

## Execution report — 2026-09-08, minhphuong

- Revision: `fd41cd4`; host pull returned already up to date.
- Launched the command above once. Both sequential evaluator commands exited 0;
  orchestration printed `COMPLETE`. All requested NLG metrics available.
- Cohort: 2,772 per arm, unique keys, identical order and references (0 mismatches).
- Both generation summaries: 160 new tokens, seed 16, no repetition controls.
- Arm A / Arm C: BLEU-4 0.076258 / 0.060451; ROUGE-L 0.235960 / 0.216142;
  METEOR 0.254718 / 0.254821; CIDEr 0.058438 / 0.016419;
  BERTScore-F1 0.795860 / 0.758685.
- Paired study bootstrap, 1,000 resamples, seed 16, delta C minus A:
  ROUGE-L -0.019818, CI95 [-0.023300, -0.016501];
  METEOR +0.000103, CI95 [-0.004818, +0.004795];
  CIDEr -0.042019, CI95 [-0.051758, -0.033385];
  BERTScore-F1 -0.037175, CI95 [-0.040275, -0.034275].
  Intervals use rounded per-sample scores; CIDEr reference IDF held fixed.
- Evaluator repetition heuristic: A 18.11%, C 35.32%. This is not the previous
  ad-hoc rep5 definition; do not compare those rates as the same measurement.
- Raw log: `/home/phuong/eval_matched_20260908.log`; arm logs and metric files
  under `/home/phuong/eval_matched_20260908/`. Aggregate `comparison.json` lives
  there too, produced by `/tmp/meta_cxr_compare_20260908.py` (exit 0).
- No training or regeneration; clinical metrics skipped. A trained 0.8621 epoch
  vs C 1 epoch, so this is a comparison of existing adapters, not an isolated
  architecture effect. Prior decoding probes inspected test data; disclose this
  when reporting. No evidence here establishes clinical correctness or that
  a particular component caused the observed differences.
