# Stage-1 to Stage-2 degradation: root-cause audit

## Goal
Separate cue decision errors, prompt/target truncation, soft-token reliance and
free-running generation failure. Fix demonstrated implementation defects; do
not change training loss or launch a new full run on an untested hypothesis.

## Preconditions
Host main pulled on 2026-09-08, revision `3f9b30c`. Review checkout
`/home/phuong/meta-cxr-cue-contract-20260908` contains the tested `88e3bb5`
cue-contract source changes. GPU busy with zero-shot generation: 11,814 MiB,
90% utilization. No new GPU run, no waiting queue.

## Commands and measurements
Run host-only CPU script `/tmp/meta-cxr-root-cause.py` with
`CUDA_VISIBLE_DEVICES=""` and
`/home/phuong/.venvs/meta-cxr-stage1-311/bin/python`, detached with setsid/nohup.
Output directory `/home/phuong/stage2_root_cause_20260908` must not exist.

- Use cached MedGemma processor/tokenizer only, no model weights. Verify image
  token expansion against the actual processor on a synthetic image before
  measuring all matched validation records. Report prompt/full lengths and
  target truncation at 768 for native, conditional, marginal and no-cue modes.
- Reanalyse existing fixed-checkpoint signal-probe outputs on identical 100
  validation cases. Report cue copying and polarity conflicts only as lexical
  diagnostics, with parser coverage; never label them clinical accuracy.
- Inspect existing four-arm score differences without new generation. Keep
  conditional effects and interactions separate from the unmatched A/C gap.

## Expected
Aggregate-only JSON with denominators and explicit limitations. No patient text,
identifiers, real image paths, prediction rows or cache tensors enter Git.

## Authorized implementation after CPU diagnosis
Add opt-in selective marginal cues: a per-label `positive_enabled` flag, checked
before emission (including saturated score 1), and a reproducible CPU calibration
CLI. Fit maximum recall at empirical validation precision >=0.70 with >=20
predicted cases; explicitly disable infeasible labels. Never fall back to 0.5
for those labels. Require a complete 13-label selective artifact and reject its
use with a different cue rule before cache/model access. Defaults stay unchanged.

Run new synthetic regression tests and the existing CPU suite in the isolated
review checkout. Generate the calibration artifact with the committed CLI and
verify numerical equality with the exploratory CPU calculation. No new Stage-1
or Stage-2 training, and no claim that cue precision alone proves generation gain.

## Abort if
Output exists, cohorts/references mismatch, token expansion differs, or cached
processor is absent. GPU work remains skipped while the existing job is active.

## Execution report

### CPU measurements completed 2026-09-08

- `/tmp/meta-cxr-root-cause.py` completed and wrote
  `/home/phuong/stage2_root_cause_20260908/summary.json`. Raw log:
  `/home/phuong/stage2_root_cause_20260908.log`. Actual processor versus explicit
  image-token expansion matched for both prompt/full text on two records per
  condition (16 exact token-ID comparisons). All 1,415 cached validation
  records had matching order, images and references internally.
- At max_length 768, conditional cues truncate **11/1,415 targets (0.78%)**,
  preserving **99.845%** of target tokens overall. Native, marginal and no-cue
  conditions truncate zero. Maximum full length: 582 / 793 / 749 / 716 tokens
  respectively. Prompt medians: 404 / 605 / 564 / 538. This establishes a small
  truncation issue on validation, not the dominant cause of degeneration and
  not a measurement of the training split. No length/recipe changes were made.
- Existing fixed-checkpoint outputs (100 matched validation studies): lexical
  positive claims without a matching reference-positive claim fall **121→103**
  when cues are removed; positive claims against an explicit reference negative
  fall **43→36**. Generated positive claims total 189→173. Parser coverage is
  **457/1,134 (40.3%)** generated sentences with cues, **430/1,082 (39.7%)**
  without; reference coverage **246/516 (47.7%)**. These are lexical counts,
  not clinical error rates; unmentioned reference findings are not proven absent
  in the image. They support investigating cue-driven overcalling, not a
  complete causal attribution of Arm C versus Arm A.
- The existing signal probe's repetition heuristic remains **41%→40%** after
  removing cues. Therefore cue removal alone has not fixed the generator's
  repetition. Zeroing soft tokens also lowers ROUGE-L by 0.0098 with its CI
  excluding zero, even though BERTScore/CIDEr changes are inconclusive: do not
  label all soft-token effects "neutral" or remove that branch as a proven cure.

### Selective precision experiment, CPU

Exact command: `CUDA_VISIBLE_DEVICES="" /home/phuong/.venvs/meta-cxr-stage1-311/bin/python /tmp/meta-cxr-selective-cues.py`, exit **0**.
Fixed in advance: precision floor 0.70, at least 20 predicted cases; maximize
validation recall subject to that constraint, disable infeasible labels.
Fit on **1,808 val** cases, then confirm once on **3,269 test** cases. No threshold
was subsequently tuned to these test results. The test set had been examined in
earlier work, so treat this as exploratory confirmation, not a pristine final test.

| Rule | Test micro precision | Test micro recall | Positive cues/study |
|---|---:|---:|---:|
| Conditional | 0.1796 | 0.7514 | 8.4607 |
| Existing marginal P-fit | 0.4266 | 0.3089 | 1.4644 |
| Selective marginal | **0.6800** | 0.2787 | 0.8287 |

All columns use the same 13-label `study_presence` framing (only positive report
labels count positive), not image-grounded truth. Precision is micro, unlike the
older macro 0.1887/0.4069 table. Selective precision bootstrap CI95:
**[0.6627, 0.6989]**, 1,000 study resamples, seed 16. The validation 0.70 constraint
does NOT become a test precision guarantee; roughly 32% of emitted test cues
still lack a matching positive report label. It emits cues on 1,967/3,269 cases.

Enabled labels: Lung Opacity, Edema, Pleural Effusion, Support Devices. The other
nine labels abstain; their absence from cues is never interpreted as negative.
The image remains available to Stage 2 for every finding.

Artifacts: `/home/phuong/stage2_root_cause_20260908/selective_thresholds.json`
and `selective_summary.json`; raw log
`/home/phuong/stage2_root_cause_selective_20260908.log`.

### Implementation verification

- Initial targeted suite: **52 passed**, exit 0, 1.76 seconds, before adding
  support for the exported missing-label sentinel. Raw log:
  `/home/phuong/root_cause_targeted_20260908.log`.
- First calibration CLI attempt: **exit 1**, final frame
  `fit_selective_thresholds`: `ValueError: expected masked P/N/U labels: -100, 0, 1, 2`.
  Inspection found exported labels use **-1**, as specified by
  `training/evaluation/schemas.py`; ReportDataset uses -100. Both are now accepted
  and counted as non-positive report labels. Updated test and real-data reruns
  passed after SSH reauthentication on 2026-09-09.
- Tailscale SSH requested an additional authentication check on 2026-09-09.
  No inference/training launch was issued while access was pending. After the
  user completed authentication, host main was pulled again (already current).
- Full CPU command in the review checkout:
  `CUDA_VISIBLE_DEVICES="" /home/phuong/.venvs/meta-cxr-stage1-311/bin/python -m pytest tests/ -q --ignore=tests/test_blip2_negative_sampling.py --ignore=tests/test_encoder_ablation.py`,
  **exit 0: 1,009 passed, 2 skipped**. This adds 17 passing selective-cue tests
  to the previous 992/2 result. Raw log:
  `/home/phuong/root_cause_full_tests_20260909.log`.
- Ruff: `PYTHONPATH=/home/phuong/.cache/cue-contract-lint /home/phuong/.venvs/meta-cxr-stage1-311/bin/python -m ruff check . --output-format json`,
  **exit 1**, 438 pre-existing diagnostics, no introduced/removed diagnostics
  relative to the unchanged baseline. Both new files pass Ruff (exit 0).
  Aggregate diagnostic file: `/home/phuong/root_cause_lint_20260909.json`.
- Real CLI: `CUDA_VISIBLE_DEVICES="" /home/phuong/.venvs/meta-cxr-stage1-311/bin/python scripts/calibrate_cue_precision.py --predictions /home/phuong/run_20260820_ft/mimic_cxr_full_blip2/result/val_predictions_epoch_best.npz --split val --precision-floor 0.70 --min-predicted 20 --output /home/phuong/stage2_root_cause_20260908/selective_reproduced.json`,
  **exit 0**, 1,808 cases, 4 enabled labels of 13. Parsed threshold objects
  exactly equal the exploratory artifact; only trailing newline differs.
- Host and local executable source match; the host copy of the Figure-9 module
  differs only in its two-line comment describing the new threshold keys.

### Interpretation and next verification

The established implementation problem is forced, overconfident cue emission
from a conditional polarity head, compounded by the empty-group normal statement
fixed in `88e3bb5`. Selective marginal emission reduces input noise with the
existing Stage-1 checkpoint. It is an opt-in mitigation, not a verified cure for
Stage-2 generation. Existing Arm C was trained with the old cues and its adapter
has not been retrained by this work.

When SSH is available, run the updated CPU suite and regenerate the threshold
artifact with `scripts/calibrate_cue_precision.py`. If the GPU is idle at that
new check, compare corrected `none` and selective marginal generation using the
same existing Arm C, the same 100 val cases restricted by the existing signal
probe, and the same greedy 160-token decoding. If busy, stop; do not queue.
Keep repetition controls off for this cue comparison so two changes are not
confounded. A subsequent repetition-control experiment is separate. Clinical
scorers remain unimplemented in this repo; never substitute lexical counts as
CheXbert/RadGraph scores.

Prepared runner: `/tmp/meta-cxr-selective-generation.sh`, paired artifact checks:
`/tmp/meta-cxr-selective-check.py`. Runner requires
fresh `/home/phuong/stage2_selective_probe_20260909`, refuses known jobs/non-idle
GPU, retains the same adapter/160-token greedy decoding, and checks 100 identical
references per arm before scoring. Launch only after successful CPU verification
and fresh idle checks, using `setsid nohup bash /tmp/meta-cxr-selective-generation.sh`.

### Generation comparison launched 2026-09-09

Fresh checks: GPU 170 MiB, 0% utilization, no compute process or known Python
training/evaluation job, output absent. Dataset driver verified ntfs3. Launched
once with `setsid nohup bash /tmp/meta-cxr-selective-generation.sh > /home/phuong/stage2_selective_probe_20260909.log 2>&1 < /dev/null &`, runner PID 106374.
The runner executes `none` then `selective`, checks exact cohort/reference
equality and scores both sequentially. Logs and outputs remain under
`/home/phuong/stage2_selective_probe_20260909`; completion and paired scores
are pending. This is a validation probe on a previously examined cohort, also
part of threshold fitting, not a held-out generation result.

### Additional stopping defect found during the probe

Read-only, CPU inspection of the locally cached MedGemma tokenizer and
GenerationConfig on 2026-09-09 returned tokenizer EOS **1**, model stop IDs
**[1, 106]**, and `<end_of_turn>` **106**. A synthetic assistant chat target
ends with 106. Both `VariantLLM.__init__` and `generate` overwrite the model's
stop list with tokenizer EOS alone. This is a demonstrated decoding-contract
defect, not yet proof of how much observed repetition it causes.

Authorized next fix: preserve the model's configured stop IDs, using tokenizer
EOS only if the model has none. Test model-specific list/scalar stops and the
fallback with a synthetic generation stream. Keep the current cue comparison
on its original source; prepare a separate host checkout for this fix. Once
the current run finishes, recheck the GPU and compare corrected stopping against
the same old-stop outputs on the identical cohort, keeping cues/decoding limits
fixed. No loss, optimizer, training target or checkpoint change is required.

Stop-fix review checkout: `/home/phuong/meta-cxr-stage2-stop-20260909`, detached
from host main `3f9b30c` after another successful pull, with the current source
patch applied. CPU synthetic stopping tests: **3 passed, exit 0**. Full suite
using the same exclusions/command above: **1,012 passed, 2 skipped, exit 0**;
raw logs `/home/phuong/stage2_stop_targeted_20260909.log` and
`/home/phuong/stage2_stop_full_tests_20260909.log`. Ruff remains 438 baseline
diagnostics with no new/removed diagnostics; new stop-token test and generation
CLI pass lint. Runner `/tmp/meta-cxr-stop-generation.sh` is prepared, not queued,
with exclusive output `/home/phuong/stage2_stop_probe_20260909` and the same
100 cases, two cue conditions, adapter and 160-token greedy decoding.

The no-cue run finished with exit 0, 100/100 outputs, zero failures, 732.5 seconds
and 10.805 GiB peak allocated VRAM. Only 39/100 outputs exactly reproduce the
historical no-cue probe. CPU audit confirms all 100 Q-Former embeddings are
bitwise equal and all no-cue prompt strings are equal between old cache and
both new cue caches. Thus historical outputs are not interchangeable with this
rerun; the cause of cross-run output drift remains unresolved. Aggregate
audit: `/home/phuong/stage2_selective_probe_20260909/cache_drift.json`.

To control this drift when testing stops, the prepared two-process stop runner
is superseded (never launched) by `/tmp/meta-cxr-stop-paired.sh` and
`/tmp/meta-cxr-stop-paired.py`. This selects 25 of the 100 cases with seed 16,
before observing corrected outputs, and runs four conditions **in one loaded
model instance**: none/selected cues crossed with EOS 1/[1,106]. Alternate
condition order between cases. Trace generated token IDs in memory and export
only aggregate counts of tokens emitted after the first end-of-turn; retain
generated text privately for matched NLG evaluation. This is a small mechanism
probe, not a full held-out generation evaluation. Launch only after the current
run exits and fresh GPU/job/output checks pass; no queued GPU process.
