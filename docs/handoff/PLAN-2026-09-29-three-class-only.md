# PLAN 2026-09-29 — Three classes, never binary (D-023)

Planner: Claude (dev box, no GPU). Decision record: `struct/project/_meta/DECISIONS.md` D-023.

## What changed

The user removed every binary reduction of the CheXpert task that an earlier AI
assistant had added without verification. Every finding is Negative / Positive /
Uncertain in the loss **and** in every metric, as in the paper.

Removed from code:
- the mention gate (`mention_heads`, `MentionGateLoss`, `lambda_gate`, `gate_class_weights`);
- the mention-conditioned objective;
- `uncertain_policy`;
- `label_framing` (`study_presence` / `masked_polarity`) and `--score marginal_presence`;
- positive-only F1 as a headline;
- binary threshold calibration (`--objective f1 --selection plateau`);
- `calibrate_cue_precision.py` and the two threshold JSONs;
- the cue rules `marginal_positive` / `conditional_positive` / `mention_gated`;
- `--finding-tokens full`.

Kept, because the paper uses them:
- Table 4 CheXpert cross-domain (`p1/(p0+p1)`);
- Table 3 Clinical Efficacy.

Both live in `training/evaluation/paper_protocol.py`.

Guards:
- `pretraining/retired_keys.py` refuses the old config keys and drops the old
  heads from checkpoints, so old checkpoints still load.
- `tests/test_three_class_only.py` is an AST and YAML tripwire.

## Paper-comparable evaluation

```bash
# Stage 1, from any existing .npz (old runs can be re-scored this way)
python scripts/evaluate_stage1.py --predictions <test.npz> --output-dir <dir>
#   weighted P/R/F1 averaged over 14 findings  (paper Fig 10: 0.87 / 0.78 / 0.73)
#   mean_weighted_f1_5                          (paper Table 5/7: 0.701)
#   one-vs-rest AUROC per class                 (paper Fig 5)

# Eq. 22 per-(finding, class) thresholds — validation only, Stage-2 prompt only
python scripts/calibrate_thresholds.py --predictions <val.npz> --output <thr.json>

# Table 4 — needs a CheXpert-val .npz (no loader in the repo yet)
python scripts/evaluate_chexpert_crossdomain.py --predictions <chexpert_val.npz>

# Table 3 CE — needs labeler output for generated and reference reports
python scripts/evaluate_clinical_efficacy.py --generated-labels g.csv --reference-labels r.csv

# Table 3 BERTScore as the paper computed it
python scripts/evaluate_stage2.py --predictions <reports.jsonl> --metrics bertscore --paper-bertscore --output-dir <dir>
```

## Not verified

- **No GPU run.** Nothing in this change has been trained or evaluated on the host.
- **The host test suite has not run.** On the dev box every failure is an
  environment gap: nltk, torchvision and the private env config are missing.
  Those tests must pass on the host first:
  - `test_stage1_eval_hook`
  - `test_cue_contract` (CLI)
  - `test_generation_stop_tokens`
  - `test_native_independence`
- Table 4 lacks a CheXpert-val loader. Table 3 CE lacks a labeler (CheXbert or
  the CheXpert labeler). RadGraph/RadCliQ are absent.
- The reference code averages metrics per batch; this repo computes them over
  the whole split.

## Executor steps

```bash
cd ~/Meta-CXR && git pull
CUDA_VISIBLE_DEVICES="" ~/.venvs/meta-cxr-stage1-311/bin/python -m pytest tests/ training/ -q
```

Expected: everything passes except the 4 `test_native_independence` failures,
which are the known missing private config. Report the pass/fail counts and the
first failure, if any.

## Execution report

_(executor appends here)_
