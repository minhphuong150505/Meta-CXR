# Final Branch Integration Audit

**Repo (inner, this is what pushes):** `META-CXR/` → `origin git@github.com:minhphuong150505/Meta-CXR-Kaggle.git`
**Audit date:** 2026-07-22
**Auditor branch at audit time:** `feature/optimize-stage2-prompts` @ `c3b477d`
**Working tree:** clean (no uncommitted changes, no stashes, single worktree)

> **Bottom line / Kết luận:** Toàn bộ lịch sử repo là **một chuỗi tuyến tính duy nhất**.
> Nhánh `feature/optimize-stage2-prompts` (`c3b477d`, 36 commit trên `main`) là **superset
> chặt của mọi nhánh khác** — kể cả hai nhánh local chưa push. Không có nhánh nào phân kỳ,
> không có commit độc lập nằm ngoài tip. Tích hợp = **fast-forward `main` → `c3b477d`**,
> không conflict, không thể mất commit.

---

## 1. Evidence — how "linear superset" was proven, not assumed

Three independent checks, all consistent:

1. **Ahead/behind vs `main`** — every non-`fix` branch is `N ahead / 0 behind`, ahead
   counts strictly increasing (1, 15, 16, 18, 20, 24, 30, 36). Both `fix/*` are `0 ahead`.
2. **`git merge-base --is-ancestor <B> feature/optimize-stage2-prompts`** — returns true
   for **all 10** other branches (main, both fixes, both feats, all four refactors,
   complete-evaluator).
3. **`git rev-list --all --not feature/optimize-stage2-prompts | wc -l` = `0`** — no commit
   anywhere in the repo (heads, remotes, tags, even the stale `original/main` fork root) is
   unreachable from the tip.
4. **`git log --merges main..tip`** — empty. The 36-commit span is pure linear history.

---

## 2. Branch inventory

| Branch | Local/Remote | HEAD SHA | Ahead main | Behind main | Ancestor of tip? | +commits over predecessor | FilesΔ vs main | Classification | Planned action |
|---|---|---|---|---|---|---|---|---|---|
| `feature/optimize-stage2-prompts` | local + remote | `c3b477d` | 36 | 0 | (is the tip) | +6 | 145 | **superset_of_other_branches** | **integrate (ff-merge)** |
| `feature/complete-evaluator` | **local only** | `19399bc` | 30 | 0 | yes | +6 | 116 | direct_ancestor_of_another_branch · local_unpushed | skipped_already_contained |
| `feat/pretrained-medgemma-findings-first` | local + remote | `9b8e27f` | 24 | 0 | yes | +4 | 87 | direct_ancestor_of_another_branch | skipped_already_contained |
| `refactor/medgemma-real-runtime-validation` | local + remote | `4bc10f9` | 20 | 0 | yes | +2 | 65 | direct_ancestor_of_another_branch | skipped_already_contained |
| `refactor/stage2-real-medgemma-runtime` | local + remote | `05728bc` | 18 | 0 | yes | +2 | 63 | direct_ancestor_of_another_branch | skipped_already_contained |
| `refactor/stage2-runtime-integration` | local + remote | `8411647` | 16 | 0 | yes | +1 | 53 | direct_ancestor_of_another_branch | skipped_already_contained |
| `refactor/clean-medgemma-xai-pipeline` | local + remote | `b03fbc9` | 15 | 0 | yes | +14 | 51 | direct_ancestor_of_another_branch | skipped_already_contained |
| `feat/medgemma-direct-default` | local + remote | `b174ca3` | 1 | 0 | yes | +1 (chain base) | 12 | direct_ancestor_of_another_branch | skipped_already_contained |
| `main` | local + remote | `94e8842` | 0 | 36 | yes | — | — | (target) | fast-forwarded to tip |
| `fix/gcs-bucket-validation` | local + remote | `2548a1e` | 0 | 1 | yes | — | 0 | **already_in_main** | skipped_already_contained |
| `fix/full-data-p0-p1-audit` | **local only** | `6a4efa3` | 0 | 6 | yes | — | 8¹ | **already_in_main** | skipped_already_contained |

¹ `fix/full-data-p0-p1-audit` is 6 commits *behind* main; the 8-file diff is main's own
forward changes shown in reverse. It has **zero** commits ahead of main — fully contained.

**Local-only branches (would be lost if not accounted for):** `feature/complete-evaluator`
and `fix/full-data-p0-p1-audit`. Both are **already ancestors of the tip**, so the
fast-forward preserves 100% of their work. Nothing local is orphaned by this integration.

---

## 3. What each branch contributed (linear segments of the chain)

Read oldest→newest; each segment builds on the one above it.

| Segment (branch) | Commits | Substance |
|---|---|---|
| `feat/medgemma-direct-default` | `b174ca3` | Make native MedGemma (`medgemma_direct`) the default Stage-2 pipeline; add IMPRESSION target column. |
| `refactor/clean-medgemma-xai-pipeline` | `daea922`…`b03fbc9` (14) | Shared `shared_visual_tokens` for MHCAC + META-Former; score FINDINGS/IMPRESSION separately; **replace mutable global config with `Stage1Context`**; add `pyproject.toml` (ruff/pytest); claim-level safety + XAI pipeline; enforce native-MedGemma independence from Stage 1; extract soft-token injector to `training/medgemma`; counterfactual audit + clinical-metric adapters; trainer state + checkpoint manager with resume-equivalence test; privacy fixture. |
| `refactor/stage2-runtime-integration` | `8411647` | Reject silent text-only fallback in `medgemma_direct` (fail-fast, no silent degradation). |
| `refactor/stage2-real-medgemma-runtime` | `943eefe`,`05728bc` | Notebook privacy guard; record runtime blocker + round-5 baseline. |
| `refactor/medgemma-real-runtime-validation` | `d13a1c8`,`4bc10f9` | Keep dirty notebook fixtures out of the tracked-notebook scan; external MedGemma findings loader. |
| `feat/pretrained-medgemma-findings-first` | `9344b2d`…`9b8e27f` (4) | Findings-first inference runner + CLI; findings phase-one guards; GPU pilot checklist + deferred-teardown plan. |
| `feature/complete-evaluator` | `fc0ea79`…`19399bc` (6) | **Classification metric framework (AUROC/AUPRC + threshold calibration + bootstrap CIs); generation metrics (BLEU/ROUGE/METEOR/CIDEr/BERTScore) + error analysis + subgroups + plots + CLIs; wire prediction dump into Stage 1; config validation.** |
| `feature/optimize-stage2-prompts` | `20b90c6`…`c3b477d` (6) | **Typed Stage-2 prompt builder + policies; versioned prompt configs + visual modes; integrate prompt builder into Stage-2 train + inference; prompt audit/analysis scripts; CPU prompt tests; architecture docs.** |

---

## 4. Overlap / superseded analysis

- **No two branches rewrite the same function on divergent lines.** Because the graph is a
  single ancestor chain, "conflict" is structurally impossible — later commits *replace*
  earlier code in place, and the tip already contains the final form of every file.
- **`already_in_main`:** `fix/gcs-bucket-validation` (main is its merge) and
  `fix/full-data-p0-p1-audit` (an ancestor of main). Merging them would create empty/no-op
  merges — skipped per plan §9.5.
- **`stale_or_superseded`:** none. Every intermediate branch's work survives verbatim in the
  tip (it is a literal prefix of the tip's history).
- **`requires_manual_review`:** none. No orphaned commits, no rewritten/dropped commits
  (verified: `git rev-list --all --not tip = 0`).

---

## 5. Tags & stale refs (for the record — untouched by integration)

| Ref | Points at | Note |
|---|---|---|
| `before-pretrained-inference-only-4bc10f9` | `4bc10f9` | backup tag, contained in tip |
| `round5-green-baseline-4bc10f9` | `4bc10f9` | backup tag, contained in tip |
| `refactor-round2-safe-backup` | `b03fbc9` | backup tag, contained in tip |
| `stage2-runtime-before-real-env-8411647` | `8411647` | backup tag, contained in tip |
| `refactor-round2-3b4e4b7` | `3b4e4b7` | intermediate commit, contained in tip |
| `original/main` (remote-tracking) | `e97d709` "author list updated" | fork root of a since-removed `original` remote; ancestor of tip. Left as-is. |

No tag or branch is deleted by this integration (per constraint: no branch deletion).

---

## 6. Conclusion feeding `final_merge_plan.md`

- **Merge exactly one branch:** `feature/optimize-stage2-prompts` (`c3b477d`).
- **Method:** fast-forward (linear, no conflicts). No cherry-picks, no manual conflict
  resolution, no `--no-ff` empty merges.
- **Everything else:** `skipped_already_contained`.
- **Data-loss risk:** none — the tip reaches every commit in the repository.
