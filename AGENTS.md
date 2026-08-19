# AGENTS.md — the executor's card

**`CLAUDE.md` in this directory is the authoritative technical document.** Read it
before running anything: it carries the training host's quirks, the mount UUID, the
venv trap, the config invariants and the loss/label policy. This file is the short
version for whoever is *executing* a plan — it states the role and the rules that
are unsafe to learn by trial.

Historical note: until 2026-08-19 the executor was Codex. That subscription lapsed;
the executor is now Claude Code, installed and authenticated on the training host.
Nothing about the role changed. This file stays agent-agnostic so it still works
if the executor changes again.

## Your role: execute the plan, report back compactly

One session plans and edits; you run. Concretely:

- Do run: `git pull`, venv commands, `pytest`, preflight, smoke runs, Stage-1/Stage-2
  launches, `scripts/supervise_stage1.sh`, log/`nvidia-smi` triage on the training host.
- Do not, unless the plan says so: change losses, YAML hyperparameters, encoder flags,
  the recipe, or the documentation. If the plan looks wrong, say so and stop — do not
  silently improvise a different run.
- The plan for a piece of work is `docs/handoff/PLAN-<YYYY-MM-DD>-<topic>.md`. Append
  your `## Execution report` to that same file (see `docs/handoff/README.md`).

⚠ **One GPU, one run.** Before launching anything that uses the card, check
`nvidia-smi` and `pgrep -af pretraining.train`. On 2026-08-19 a launch command was
issued twice 50 seconds apart; the second OOMed against the first and, sharing an
output path, overwrote the real report with its abort. If you find the GPU busy,
stop and report — do not queue behind it and do not retry.

## Reporting errors — summarize, never paste the log

Training logs are hundreds of megabytes of MetricLogger lines. Send back:

1. the exact command and its exit status;
2. the first error with ~20 lines of context, plus the final traceback frame only;
3. the numbers that decide anything — `s/it`, `max mem`, each loss term by name,
   `epoch`/`iter` at failure, `nvidia-smi` VRAM on an OOM;
4. where the raw log still lives on the host, so the next question can be a single
   `grep` instead of the whole file.

An OOM is a *capacity finding*, not a crash to retry blindly: report the batch/accum
and peak VRAM. Note that `supervise_stage1.sh`'s halve-batch fallback changes the
number of negatives every microbatch-local loss sees, so treat a fallback as a
result to report, not a recovery.

Report what ran and what it printed. "Not run", "failed" and "skipped" are all
acceptable answers; a claim that something passed when it was never executed is not.

## Hard rules — violating any of these is worse than failing the task

- **MIMIC-CXR is PhysioNet credentialed data; this remote is public.** Never commit or
  paste report text, `subject_id` / `study_id` / `dicom_id`, real image paths, split
  CSVs, `*.npz`, `*.jsonl`, checkpoints or credentials — not in a commit, a handoff
  file, or a summary. Do not disable the notebook privacy pre-commit hook.
- **Nothing runs in the dev checkout.** It has no GPU and no dataset. Everything that
  executes goes to the one training host — `ssh phuong@100.116.167.90` (Tailscale name
  `minhphuong`). Always `git pull` on the host before running, or the results belong to
  the wrong revision. Launch anything long with `setsid nohup ... &`.
- **Mount the dataset disk by UUID only:** `UUID=A4E6C088E6C05BE4`, `-t ntfs3 -o ro`.
  The `nvme0n1` / `nvme1n1` names have swapped across reboots three times; a
  device-named command has even odds of naming `/`. `sudo` needs the user. Note the
  desktop may auto-mount it `rw` at `/run/media/phuong/A4E6C088E6C05BE4` instead, which
  is *not* where the configs look.
- **Use `~/.venvs/meta-cxr-stage1-311/bin/python`.** Do not
  `pip install -r requirements-stage1.txt` unfiltered — it pins torch 2.5.1 (sm_90)
  onto an sm_120 card and fails hours later. Procedure in `CLAUDE.md`, "The venv".
- **`run.output_dir` must be on `/home`** (ext4). The data drive is mounted read-only.
- **There is no cloud path.** Do not reintroduce `cloud/`, GCS uploads, Kaggle flows or
  hardware-named configs; anything in git history describing them is dead.
- **A behavioral change is not finished until `CLAUDE.md`, `README.md` (Vietnamese) and
  `struct/` are updated in the same commit.**

## Verification you can run without a GPU

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q \
    --ignore=tests/test_blip2_negative_sampling.py --ignore=tests/test_encoder_ablation.py
ruff check .
python -m training.dataio.validate_manifest --section-mode findings_and_impression
```

Baseline as of 2026-08-19: 5 failed (`test_native_independence` ×4,
`test_stage1_eval_hook` ×1), both environmental. Do not "fix" them. Report a *change*
from that baseline, not the absolute number.
