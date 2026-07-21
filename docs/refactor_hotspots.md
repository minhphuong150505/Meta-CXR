# Refactor hotspots

Size and responsibility inventory of the Stage-2 and safety code, measured at
commit `103fb95`. Line counts are `wc -l`; class and function counts are
top-level `class`/`def` plus one level of method.

## Inventory

| File | LOC | Classes | Funcs | Responsibilities | Decision | Reason |
|---|---:|---:|---:|---|---|---|
| `training/train_eval_figure9_llm_variants_200.py` | 1615 | 2 | 60 | model loading, quantization, LoRA targeting, prompts, encoding, collation, forward, train loop, early stopping, generation, 5 NLG metrics, adapter I/O, GCS upload, privacy verification, Figure-9 plotting | **Split — in progress** | Still the largest file by 3×. Round 2 removed the Stage-1 imports and the soft-token injector from it; the trainer and metric blocks are next. |
| `training/run_medgemma_qlora.py` | 489 | 0 | 8 | CLI parsing (110 lines), mode resolution, record building, per-mode orchestration, upload allow-list | **Split later** | Over the ~300-line orchestration target, driven by 45 `argparse` lines. Splitting the parser into `config/` is low-risk but was not needed to close any Critical finding. |
| `safety/reconciler.py` | 328 | 3 | 11 | claim editing, hedging, measurement stripping, reconciliation | **Keep** | Three cohesive classes around one workflow; no single class dominates. |
| `training/stage2_utils.py` | 298 | 0 | 15 | 15 unrelated helpers: fingerprints, threshold selection, label masking, LoRA target names, section omission, bucket policy | **Split later** | This is the `utils.py` the brief warns about. Its members belong in `privacy/`, `data/` and `medgemma/`. Deferred because every one of them is imported by name from two entrypoints and moving them touches more surface than round 2's Critical work justified. |
| `safety/claims.py` | 286 | 4 | 13 | claim datamodel, sentence splitting, polarity detection, lexicon parsing | **Keep** | Cohesive; the parser and the datamodel it produces belong together. |
| `training/evaluation/counterfactual.py` | 268 | 5 | 12 | protocols, config, evaluator, recursive privacy check | **Keep** | New in round 2. Four of the five "classes" are `Protocol`/dataclass declarations. |
| `safety/verifiers.py` | 252 | 9 | 14 | verifier protocols and implementations | **Keep** | Nine classes, but six are `Protocol` declarations or small adapters. |
| `training/dataio/manifest.py` | 248 | 1 | 8 | split CSV reading, record building, leakage assertions | **Keep** | Single responsibility: turning split CSVs into native records. |
| `training/evaluation/clinical.py` | 185 | 6 | 12 | optional clinical-metric adapters | **Keep** | New in round 2. Classes are one base plus three thin subclasses. |
| `safety/pipeline.py` | 180 | 2 | 7 | safety orchestration | **Keep** | |
| `training/evaluation/perturbations.py` | 180 | 1 | 8 | 8 image perturbations | **Keep** | New in round 2. Pure functions plus one dataclass. |
| `training/trainer/state.py` | 174 | 2 | 11 | RNG capture, run counters, provenance | **Keep** | New in round 2. |
| `training/trainer/checkpointing.py` | 120 | 1 | 7 | atomic checkpoint I/O | **Keep** | New in round 2. |
| `training/stage1/lavis_loader.py` | 112 | 0 | 6 | every LAVIS import in Stage-2 | **Keep** | New in round 2. Its size is not the point — its *isolation* is. |
| `training/medgemma/soft_tokens.py` | 89 | 1 | 3 | soft-token substitution | **Keep** | New in round 2. |
| `training/torch_io.py` | 26 | 0 | 1 | checkpoint loading | **Keep** | Exists so Stage-1 and Stage-2 can share one loader without importing each other. |

## The remaining god script

`train_eval_figure9_llm_variants_200.py` went 1685 → 1615 lines. That is a
small dent, and the honest reading is that round 2 prioritised *correct
dependency direction* over line count: the file no longer forces LAVIS onto a
native run, and its riskiest component now has tests. What it still holds:

| Block | Lines (approx) | Target module | Blocked on |
|---|---:|---|---|
| `VariantLLM.__init__` — model load, NF4, LoRA attach | 120 | `training/medgemma/loader.py` | Needs `transformers`/`peft` to test; unverifiable here. |
| `VariantLLM._chat_texts`, `_native_messages`, prompt builders | 110 | `training/medgemma/prompts.py` | Testable — good next step. |
| `VariantLLM.encode_train_example`, `collate_train` | 70 | `training/medgemma/collators.py` | Needs a tokenizer. |
| `VariantLLM.train_fine` | 177 | `training/trainer/trainer.py` | `training/trainer/` now exists to receive it. |
| `VariantLLM.generate` | 55 | `training/medgemma/generation.py` | Needs a model. |
| `compute_nlg`, `compute_sectioned_nlg` | 115 | `training/evaluation/nlg.py` | Needs `nltk`/`bert_score`/`pycocoevalcap`. |
| `assert_private_gcs_destination`, `upload_*` | 45 | `training/artifacts/gcs.py` | Testable — good next step. |
| `run_family`, `plot_family`, `main` | 130 | stays — this is the actual Figure-9 orchestration | — |

Everything above the "blocked on" column marked *needs X* can be moved, but the
move cannot be verified in this environment. Moving 400 lines of untestable
code is how a refactor introduces the bug it was meant to prevent, so those are
sequenced behind an environment that can run them, or behind extending the
conftest stubs.

## Targets versus reality

| Target from the brief | Status |
|---|---|
| CLI entrypoint ≤ ~150 lines | `run_medgemma_qlora.py` is 489. Not met. |
| Experiment orchestration ≤ ~300 lines | Figure 9 is 1615. Not met. |
| No new catch-all `utils.py` | Met — no new one. The pre-existing `stage2_utils.py` remains. |
| One class, one responsibility | `VariantLLM` still holds ~10. Not met. |
| Functions independently testable | Met for everything extracted in round 2. |
