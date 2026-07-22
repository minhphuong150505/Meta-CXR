# Final Merge Plan

**Companion to:** `docs/final_branch_integration_audit.md`
**Date:** 2026-07-22 · **Repo:** inner `META-CXR/` (`origin` → `Meta-CXR-Kaggle`, branch `main`)

The audit proved the repo is a single linear chain and `feature/optimize-stage2-prompts`
(`c3b477d`) is the universal superset. The plan below therefore integrates **one** branch by
fast-forward and adds the VM-readiness deliverables on top. No blind mass-merge, no
force-push, no branch deletion.

---

## 1. Branches to integrate

| Branch | Action | Reason |
|---|---|---|
| `feature/optimize-stage2-prompts` (`c3b477d`) | **merged (fast-forward)** | Strict superset of every other branch; contains all 36 commits. |

## 2. Branches skipped because already contained

| Branch | Action | Reason |
|---|---|---|
| `feature/complete-evaluator` | skipped_already_contained | Ancestor of tip (30/36). Local-only, but 100% preserved by the ff. |
| `feat/pretrained-medgemma-findings-first` | skipped_already_contained | Ancestor of tip (24/36). |
| `refactor/medgemma-real-runtime-validation` | skipped_already_contained | Ancestor of tip (20/36). |
| `refactor/stage2-real-medgemma-runtime` | skipped_already_contained | Ancestor of tip (18/36). |
| `refactor/stage2-runtime-integration` | skipped_already_contained | Ancestor of tip (16/36). |
| `refactor/clean-medgemma-xai-pipeline` | skipped_already_contained | Ancestor of tip (15/36). |
| `feat/medgemma-direct-default` | skipped_already_contained | Ancestor of tip (1/36), chain base. |
| `fix/gcs-bucket-validation` | skipped_already_contained | Already in main (main is its merge). |
| `fix/full-data-p0-p1-audit` | skipped_already_contained | Already in main (6 behind, 0 ahead). Local-only, preserved. |

## 3. Branches superseded / needing cherry-pick / manual review

**None.** No stale-but-partially-useful branch, no rewritten/dropped commit, no divergence.

## 4. Commits to cherry-pick individually

**None.** A fast-forward carries every commit; there is nothing to pick out.

---

## 5. Merge order (actual sequence)

Because there is one superset branch and the rest are its ancestors, the order is trivial:

1. `git tag backup-main-before-final-20260722 main` — annotated backup of current `main`.
2. `git checkout -b integration/final-vm-20260722 feature/optimize-stage2-prompts`
   — integration branch starts at the superset (all 36 commits present).
3. Add integration + VM-readiness artifacts on top of the integration branch, commit:
   - `docs/final_branch_integration_audit.md`, `docs/final_merge_plan.md`,
     `docs/final_merge_conflicts.md`
   - `docs/VM_TRAINING_FINAL.md`
   - `scripts/vm_preflight.py`
   - `configs/vm_2x3090_64gb/` (only if no equivalent exists)
4. Run all non-GPU gates (§7) on the integration branch.
5. `git checkout main && git merge --ff-only integration/final-vm-20260722`
   — pure fast-forward; `main` advances to include the 36 commits + integration artifacts.
6. `git tag -a vm-train-ready-20260722` on the new `main`.
7. Push `integration/final-vm-20260722`, then `main`, then the tag (see §8 / remote decision).

## 6. Expected conflicts

**None.** Fast-forward cannot conflict. `docs/final_merge_conflicts.md` therefore records
"no conflicts" with the evidence, per the audit template.

---

## 7. Test gate after each step (no GPU required)

Run on the integration branch before touching `main`:

```bash
# Gate 1 — git integrity
git fsck --full && git diff --check && git status --short

# Gate 2 — syntax
CUDA_VISIBLE_DEVICES="" python -m compileall -q stage2 training scripts runtime safety tests medgemma_inference

# Gate 3 — CPU tests (baseline recorded: 465 passed)
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q

# Gate 4 — CLI --help smoke (Stage 1/2 train, evaluators, calibration, prompt export, preflight)
# Gate 5 — YAML config load
# Gate 6 — synthetic dry-run (prompt builder, manifest, label masking, evaluator)
```

**Rule:** final test count must not drop below the pre-merge baseline (465) without a
recorded reason. Since the merge is a fast-forward, the tree at `main`-after == the tree at
the tip, so the after-merge suite is byte-identical plus the new (additive) integration
artifacts.

## 8. Remote step (no force-push, no branch deletion)

- Push `integration/final-vm-20260722` first.
- Advance `main` by **fast-forward only** (`git merge --ff-only`), then `git push origin main`.
- If `main` moved on the remote or branch protection blocks a direct push: **do not force**.
  Rebase/merge the updated remote `main` into the integration branch, re-run gates, and open
  a PR `integration/final-vm-20260722 → main`.
- Old branches are **kept**. Backup tag `backup-main-before-final-20260722` is the rollback
  point (`git reset --hard <tag>` is available to the user but is **not** run here).

## 9. Explicit non-goals for this integration

- No training run of any size beyond CPU dry-runs.
- No model metrics reported (none have been produced on GPU by this pipeline).
- No hyperparameter changes made "to make the merge pass".
- No deletion of any local or remote branch.
