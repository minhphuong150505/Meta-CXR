# Stage-1 cue contract and Stage-2 wiring

## Goal
Distinguish withheld cues, abstention and actual predictions. Neither withheld
nor abstained groups may emit a normal summary. Partial negatives must remain
specific. Thread the same cue rule through training, generation and fingerprints.
Keep loss, weights, training recipe and YAML hyperparameters unchanged.

## Preconditions
Training host `phuong@100.116.167.90`, Python
`/home/phuong/.venvs/meta-cxr-stage1-311/bin/python`. Pull host main first, then
apply the review diff in an isolated detached worktree under `/home`.
Never modify the checkout used by the active GPU process.

## Commands
On host, after `git pull --ff-only` in `~/Meta-CXR`:

```bash
test ! -e /home/phuong/meta-cxr-cue-contract-20260908
git worktree add --detach /home/phuong/meta-cxr-cue-contract-20260908 HEAD
cd /home/phuong/meta-cxr-cue-contract-20260908
git apply /tmp/meta-cxr-cue-contract.patch
```

Copy the new synthetic test file and this plan into the isolated worktree.
Run detached with log files outside the repository:

```bash
CUDA_VISIBLE_DEVICES="" /home/phuong/.venvs/meta-cxr-stage1-311/bin/python -m pytest \
  tests/test_cue_contract.py tests/test_cue_mention_gate.py \
  tests/test_stage2_prompts.py tests/test_generate_stage2_reports.py -q
CUDA_VISIBLE_DEVICES="" /home/phuong/.venvs/meta-cxr-stage1-311/bin/python -m pytest \
  tests/ -q --ignore=tests/test_blip2_negative_sampling.py \
  --ignore=tests/test_encoder_ablation.py
/home/phuong/.venvs/meta-cxr-stage1-311/bin/ruff check .
```

Use synthetic tensors and mocked model/data loaders; all project execution is
on the training host. Compare full-suite failures with the unchanged revision
when needed; do not alter the known environmental baseline.

## Expected
Regression coverage for empty/withheld cues, partial negatives, legacy cache
normalization, intact image and soft-token parts, and train/generation parity
for all three splits. Summarize counts and keep raw logs on the host.

## Abort if
Output/worktree exists, unexpected host changes, or tests attempt model loading.
Before any GPU run check known job names, `nvidia-smi`, mount driver and exclusive
output directory. GPU busy means skip GPU testing; do not queue or retry.

## Execution report

- Initial host pull: exit 0, revision `3f9b30c` already current.
- GPU busy: 11,814 MiB / 65%, active zero-shot report generation. No additional
  GPU run launched. CPU regression checks proceed in isolation.
- Source changes tested: base `3f9b30c` plus the cue-contract review diff in
  `/home/phuong/meta-cxr-cue-contract-20260908`.
- Initial targeted collection failed because the isolated checkout lacked
  git-ignored `configs/env_config.yaml`. Linked the host's existing config,
  then reran. No recipe or environment configuration contents were changed.
- Targeted command above: exit 0, 101 tests completed. Raw log:
  `/home/phuong/cue_contract_targeted_v2_20260908.log`.
- Full suite command above, adding `-o addopts=` for visible counts: exit 0,
  **992 passed, 2 skipped**, 9.26 seconds. Raw log:
  `/home/phuong/cue_contract_full_20260908.log`.
- Same command on unchanged `3f9b30c`, isolated baseline checkout: exit 0,
  **974 passed, 2 skipped**, 9.11 seconds. Delta: **18 added passing tests,
  zero new failures**. The historical five environmental failures are absent
  on this host in both revisions; no baseline tests were changed to achieve it.
  Raw log: `/home/phuong/cue_contract_baseline_tests_20260908.log`.
- The venv did not contain Ruff (initial executable check exit 127). Installed
  Ruff into isolated `/home/phuong/.cache/cue-contract-lint` with `pip --target`,
  without changing the active model venv. Invocation is
  `PYTHONPATH=/home/phuong/.cache/cue-contract-lint /home/phuong/.venvs/meta-cxr-stage1-311/bin/python -m ruff check .`.
  Both unchanged and patched revisions report **438 issues, exit 1**.
  Compared diagnostic multisets by relative filename, rule and message:
  **0 added, 0 removed**. Raw diagnostics remain on the host in
  `cue_contract_baseline_lint_20260908.json` and
  `cue_contract_patched_lint_20260908.json` under `/home/phuong/`.

### CPU audit on existing private validation caches

Executed `/tmp/meta-cxr-cue-audit.py` using the host Python and
`CUDA_VISIBLE_DEVICES=""`, detached. It compares all three cached cohorts for
equality internally and renders all 1,415 validation records per rule, plus
one fixed seed-16 subset of 100 per rule. No generated/reference text or keys
are written into this report. Every prompt retains one native image part and
32 soft-token placeholders, with unchanged embedding objects.

| Rule | n | Old normal statements | New normal statements | New structured blocks |
|---|---:|---:|---:|---:|
| conditional_positive | 1,415 | 0 | 0 | 1,415 |
| marginal_positive | 1,415 | 654 | 0 | 761 |
| none | 1,415 | 1,415 | 0 | 0 |

For the fixed 100: marginal has 46 abstentions and 54 structured blocks;
none has 100 withheld states and zero structured blocks. All assertions passed;
aggregate artifact exists at
`/home/phuong/cue_contract_audit_20260908/summary.json`, raw log at
`/home/phuong/cue_contract_audit_20260908.log`. This is a prompt audit, not
model generation or a clinical evaluation. The audit script is retained on host.

### Assessment

The semantics bug affects 46.2% of validation cases under marginal cues; it is
not merely an empty synthetic fixture edge case. The changes remove unsupported
normal summaries and establish train/generation rule parity. No claim of better
generated reports, clinical F1, or improved Stage-1 precision/recall is warranted:
weights, logits, thresholds and losses were not changed, and GPU generation was
skipped because the GPU was already in use. Do not queue behind that job.

The unchanged baseline was tested in
`/home/phuong/meta-cxr-cue-baseline-20260908`; commands and exit markers are
retained in `/tmp/meta-cxr-cue-baseline.sh`. Lint was installed only because the
required executable was missing; no unrelated lint issues were fixed.
