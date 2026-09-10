# Default marginal cues for Stage 2

## Goal
User requests m * q_positive as the default whenever Stage-1 findings are fed
to Stage 2; below-threshold findings abstain. Native and no-structured-cue
pipelines must continue working without a Stage-1 dependency.

## Commands and scope
Change training/generation CLI resolution to choose marginal_positive for
structured Stage-1 pipeline modes when --cue-rule is omitted. Preserve explicit
conditional_positive for reproducing old runs. Keep the guided prompt guard,
per-label marginal thresholds and existing 0.5 fallback; never automatically
load a checkpoint-specific calibration file. Shared low-level defaults stay
explicitly documented as historical compatibility if not changed.

Run CPU regression tests on the training host in a fresh isolated review
worktree after git pull --ff-only, using the approved Python with CUDA hidden.
Check existing jobs and GPU occupancy before testing; no new GPU job or training.
Verify omitted flags actually route both CLIs to marginal, including records,
cache identity and prompt abstention. Update CLAUDE.md, Vietnamese README and
struct documentation in the same commit. Report actual test/lint outcomes.

## Abort if
Host access unavailable or review output exists; do not overwrite logs/results.
No dataset, report text, IDs, predictions or checkpoints enter Git.

## Execution report
Source and documentation edits complete. `git diff --check` passed.
No project tests, lint or GPU run executed: Tailscale SSH required browser
reauthentication before the host pull/status command could finish. The user
was sent the authentication URL; test status is **not run**, not passed.

Prepared source/test patch: `/tmp/meta-cxr-default-cues-20260910.patch`, against
local origin/main `79139fc` including the prior evaluation-cache fix. After
host access is restored, pull host main and create a fresh detached review
worktree `/home/phuong/meta-cxr-default-cues-20260910` at `79139fc`, apply the
patch, and link `configs/env_config.yaml` to the host's ignored config.
Run with `/home/phuong/.venvs/meta-cxr-stage1-311/bin/python`:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_cue_contract.py tests/test_cue_mention_gate.py tests/test_generate_stage2_reports.py -q
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q --ignore=tests/test_blip2_negative_sampling.py --ignore=tests/test_encoder_ablation.py
```

Use fresh logs on `/home`, shell noclobber, and refuse any existing review
output. Regression coverage includes omitted flags through both entrypoints,
explicit historical overrides, no-cue/native mode compatibility, low-mention
abstention and marginal-threshold precedence. No threshold file, loss, YAML,
checkpoint or low-level legacy default was changed.

