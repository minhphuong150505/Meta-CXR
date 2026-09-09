# validate_selective_thresholds(context, cue_rule)

An artifact carrying any `positive_enabled` flag must specify both that flag
and `marginal_positive` for every one of the 13 reportable findings. Flags are
binary. The artifact can only be used with `cue_rule=marginal_positive`.

Called by `build_stage1_records` before cache/model access and by
`classify_with_thresholds` for direct callers. Old threshold artifacts without
enable flags retain their existing behavior. Disabled marginal labels are
skipped before the score comparison, even when the score rounds to exactly 1.

Producer: `scripts/calibrate_cue_precision.py`. Consumer: common Stage-2 record
builder, used by training and generation. Tests: `tests/test_selective_cues.py`.

Parent: [engine](../train_eval_figure9_llm_variants_200.py.doc.md).
