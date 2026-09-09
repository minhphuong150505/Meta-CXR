# scripts/calibrate_cue_precision.py

CPU-only CLI for selective positive cues. Reads locally trusted validation NPZ
arrays (probabilities, mention probabilities, labels, pathology names), computes
`mention_probability * q_positive`, then calls `fit_selective_thresholds`.
No model, dataset images, report strings or GPU are loaded.

`fit_selective_thresholds` maximizes true positives subject to empirical
precision >= `--precision-floor` (default 0.70) and at least `--min-predicted`
(default 20) emitted cases. Ties prefer fewer predictions and a higher threshold.
Equal scores cannot be split to manufacture precision. A label without a feasible
threshold gets `positive_enabled=0`; it never falls back to 0.5. Missing labels
(-1 in evaluation exports, -100 in the dataset), negatives and uncertain labels
are non-positive under this declared report-presence framing. No Finding is excluded.

CLI requires `--split val|validation` and refuses contradictory stored split
metadata, invalid arrays and existing outputs. When the NPZ lacks split metadata,
the caller is responsible for supplying the real validation artifact. The CLI
writes threshold JSON plus `.metadata.json` with fitting settings, denominators,
per-label results and the limitation that the floor is not an out-of-sample guarantee.

Consumer: `training/train_eval_figure9_llm_variants_200.py` via `load_thresholds`,
`validate_selective_thresholds` and `classify_with_thresholds`. Both train and
generation select `--cue-rule marginal_positive --threshold-path <artifact>`.

Tests: `tests/test_selective_cues.py`. Validation/test empirical results and
privacy-safe artifact locations are in
[the handoff](../../../docs/handoff/PLAN-2026-09-08-stage2-root-cause.md).

Parent: [scripts index](_index.md).
