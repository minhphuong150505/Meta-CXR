# AGENTS.md — for Codex (and any non-Claude agent) working in `Meta-CXR-source/`

**`CLAUDE.md` in this directory is the authoritative technical document.** Read it
before running anything: it carries the training host's quirks, the mount UUID, the
venv trap, the config invariants and the loss/label policy. This file only states
*your role* and the rules that are unsafe to learn by trial.

## Your role: execute the plan, report back compactly

Claude plans and edits; you run. Concretely:

- Do run: `git pull`, venv commands, `pytest`, preflight, smoke runs, Stage-1/Stage-2
  launches, `scripts/supervise_stage1.sh`, log/`nvidia-smi` triage on the training host.
- Do not, unless the plan says so: change losses, YAML hyperparameters, encoder flags,
  the recipe, or the documentation. If the plan looks wrong, say so and stop — do not
  silently improvise a different run.
- The plan for a piece of work is `docs/handoff/PLAN-<YYYY-MM-DD>-<topic>.md`. Append
  your `## Execution report` to that same file (see `docs/handoff/README.md`).

## Reporting errors — summarize, never paste the log

Training logs are hundreds of megabytes of MetricLogger lines. Send back:

1. the exact command and its exit status;
2. the first error with ~20 lines of context, plus the final traceback frame only;
3. the numbers that decide anything — `s/it`, `max mem`, each loss term by name,
   `epoch`/`iter` at failure, `nvidia-smi` VRAM on an OOM;
4. where the raw log still lives on the host, so the next question can be a single
   `grep` instead of the whole file.

An OOM is a *capacity finding*, not a crash to retry blindly: report the batch/accum
and peak VRAM, and note that `supervise_stage1.sh`'s halve-batch fallback silently
reintroduces the ITM negative starvation the current config exists to avoid.

## Hard rules — violating any of these is worse than failing the task

- **MIMIC-CXR is PhysioNet credentialed data; this remote is public.** Never commit or
  paste report text, `subject_id` / `study_id` / `dicom_id`, real image paths, split
  CSVs, `*.npz`, `*.jsonl`, checkpoints or credentials — not in a commit, a handoff
  file, or a summary. Do not disable the notebook privacy pre-commit hook.
- **Nothing here runs locally.** This checkout has no GPU and no dataset. Everything
  that executes goes over SSH to the one training host — `ssh phuong@100.116.167.90`
  (Tailscale name `minhphuong`). Tailscale SSH needs an interactive browser approval:
  start it in the background, surface the URL to the user, wait. Always `git pull` on
  the host before running, or the results belong to the wrong revision.
- **Mount the dataset disk by UUID only:** `UUID=A4E6C088E6C05BE4`, `-t ntfs3 -o ro`.
  The `nvme0n1` / `nvme1n1` names have swapped across reboots three times; a
  device-named command has even odds of naming `/`. `sudo` needs the user.
- **Use `~/.venvs/meta-cxr-stage1-311/bin/python`.** Do not
  `pip install -r requirements-stage1.txt` unfiltered — it pins torch 2.5.1 (sm_90)
  onto an sm_120 card and fails hours later. The working procedure is in `CLAUDE.md`,
  "The venv".
- **`run.output_dir` must be on `/home`** (ext4). The data drive is mounted read-only.
- **There is no cloud path.** Do not reintroduce `cloud/`, GCS uploads, Kaggle flows or
  hardware-named configs; anything in git history describing them is dead.
- **A behavioral change is not finished until `CLAUDE.md`, `README.md` (Vietnamese) and
  `struct/` are updated in the same commit.** If you made the change, you own that
  update; if Claude made it, check that it happened before you commit.

## Verification you can run without a GPU

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q \
    --ignore=tests/test_blip2_negative_sampling.py --ignore=tests/test_encoder_ablation.py
ruff check .
python -m training.dataio.validate_manifest --section-mode findings_and_impression
```

Baseline as of 2026-08-17: 590 passed, 5 failed, 3 skipped. The 5 failures
(`test_native_independence` ×4, `test_stage1_eval_hook` ×1) are pre-existing and
environmental — do not "fix" them. Report a *change* from that baseline, not the
absolute number.

## Report honestly

State what ran and what it printed. "Not run", "failed", and "skipped" are all
acceptable answers; a claim that something passed when it was never executed is not.
Nothing in the current Stage-2 path has been validated on GPU — do not imply otherwise.
