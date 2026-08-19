# `docs/handoff/` — the Claude → Codex → Claude relay

Claude plans, Codex executes (`../../AGENTS.md`, and "Who does what" in
`../../CLAUDE.md`). The handoff is a file rather than a chat message so a session
that starts cold can pick the work up.

## One file per unit of work

`PLAN-<YYYY-MM-DD>-<topic>.md`, e.g. `PLAN-2026-08-19-qformer-gate-check.md`.

**Claude writes, before any execution:**

```markdown
# <topic>
## Goal            — what is true when this is done
## Preconditions   — branch/commit, disk mounted?, venv, checkpoint needed
## Commands        — exact, in order, copy-pasteable
## Expected        — what a healthy run prints; the numbers to read back
## Abort if        — the conditions that mean stop and report, not retry
```

**Codex appends, after execution:**

```markdown
## Execution report — <date>, <host>
- commit run:        <sha>
- command / status:  <cmd> → exit N
- result:            the numbers from "Expected", or the first error with ~20
                     lines of context and the final traceback frame
- raw log:           <path on the host> (not copied here)
- what I did NOT do: anything skipped or deviated from, and why
```

## Rules

- **No patient data, ever** — no report text, `subject_id`/`study_id`/`dicom_id`,
  real image paths, or pasted rows. Metrics and losses are fine.
- **No pasted logs.** Summarize; leave the raw file on the host and name its path.
- Amend the same file rather than opening a second one; the plan and what actually
  happened belong side by side.
- These files are committed to a public remote. Write them accordingly.
