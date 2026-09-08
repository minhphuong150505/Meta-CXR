# Fixed-checkpoint Arm C signal intervention

## Goal
Generate and score four inference conditions on the same 100 validation studies:
full input, no P/N/U cues, zero projected soft tokens, and both interventions.
This tests reliance on signals, not retrained architectural ablation.

## Preconditions
Host pull --ff-only already current; GPU idle; no active training/evaluation.
Dataset mounted ntfs3. Use Python 3.11 venv and existing Arm C adapter and
Stage-1 record cache. Do not change repository training/inference code.

## Commands
Deploy `/tmp/meta_cxr_signal_probe_20260908.py` to the same path on the host.
Launch once: `setsid nohup /home/phuong/.venvs/meta-cxr-stage1-311/bin/python -u /tmp/meta_cxr_signal_probe_20260908.py`.
Log: `/home/phuong/signal_probe_val_20260908.log`.
Private outputs: `/home/phuong/signal_probe_val_20260908/` (exclusive creation).

## Design and verification
100 validation records sampled seed 16; one fixed Arm C checkpoint, native pixels
retained, greedy decoding, 160 new tokens, no anti-repetition controls.
Remove only PromptBuilder structured parts, not replace findings with empty/normal.
Zero img_proj output with a temporary hook (including bias); keep 32 placeholders.
Check every prompt contains 32 soft and 256 native image tokens; retain pixel_values.
Restore builder and hook in finally. Freeze all model/projector weights.
Write outputs per study for all four conditions, flushing each record.
Then score sequentially using existing evaluate_stage2.py, identical NLG settings,
1,000 bootstrap resamples. Pair per-study differences against full input.

## Expected
400 outputs, 100 input checks per arm, 200 zero-projection hook calls,
all four evaluator exits 0, comparison.json and COMPLETE in log.

## Abort if
Busy GPU, existing output directory, invalid input counts, missing checkpoint,
generation/evaluation failure. Do not retry an OOM or alter batch/config.
No report text or sample identifiers in this handoff or user-facing summaries.

## Execution report — started 2026-09-08, minhphuong

- Host revision `fd41cd4`; pull already current; GPU 170 MiB / 0% before launch.
- Host Python syntax check passed. Launched detached once, with
  `CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True`.
- Validation Stage-1 cache hit; all 100 no-cue prompt structural checks passed.
- First 5 studies completed in all four conditions: 20/400 outputs in 146.8 s.
  Per-generation native/soft token checks and zero-projection hook checks passed.
- Generation and four subsequent NLG evaluations run automatically in one
  host-side script. Final aggregate is `comparison.json`; completion is indicated
  by `COMPLETE` in the log. Results not yet available at this progress update.
- Existing training code and checkpoints were not modified.
