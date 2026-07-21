# Second-pass source audit

Evidence-based re-read of the working tree at the start of refactor round 2.
Round 1's findings live in `docs/code_audit_latest.md`; this file supersedes it
where the two disagree, because commits landed after that audit was written.

## 1. Source state

| Field | Value |
|---|---|
| Repo | `META-CXR/` (inner) — `git@github.com:minhphuong150505/Meta-CXR-Kaggle.git` |
| Branch | `refactor/clean-medgemma-xai-pipeline` |
| Base SHA (round 1 final) | `ee3ba2f8be02d7f87b3df126c2998b74e0b5ac44` |
| SHA at audit time | `ee3ba2f` (identical — no work lost) |
| Working tree | clean (`git status --porcelain` empty) |
| Pushed to remote? | **No.** `git log --decorate` shows no `origin/refactor/clean-medgemma-xai-pipeline`. The branch exists only locally. |
| Tracked files | 198 |

### Commits inherited from round 1

```
ee3ba2f feat: add claim-level safety and XAI pipeline
fc9807d docs: record which audit findings round 1 closed
6ec8d99 chore: add pyproject with ruff/pytest config; move stray test
76677ba refactor: replace mutable global configuration with Stage1Context
cd9fb77 feat: score FINDINGS and IMPRESSION separately
f23f4b7 refactor: one shared_visual_tokens for MHCAC and META-Former
daea922 docs: add evidence-based audit of latest source
```

Branch point: `b174ca3` (`feat/medgemma-direct-default`).

## 2. Baseline, measured before any edit

Environment: Python 3.12.3, Linux 6.17.0-40-generic, interpreter
`/home/phuong/venv/bin/python`.

| Command | Result |
|---|---|
| `python -m compileall -q .` | exit 0, no output |
| `python -m pytest -q` | **141 passed, 0 failed, 0 skipped, 2.6 s** |
| `ruff check .` | **427 errors** (373 auto-fixable) |
| `ruff format --check .` | 59 files would be reformatted, 16 already formatted |
| `mypy` | **not installed**; no mypy config in `pyproject.toml` |

`ruff` was not present in the venv either; it was installed for this pass
(`pip install ruff` — a self-contained binary wheel with no Python
dependencies, so it cannot perturb the torch pin).

### Test count is 141, matching round 1's report

| File | Tests |
|---|---|
| `tests/test_safety_pipeline.py` | 31 |
| `tests/test_shared_visual_tokens.py` | 20 |
| `tests/test_manifest.py` | 17 |
| `tests/test_mimic_data_pipeline.py` | 13 |
| `tests/test_stage2_utils.py` | 13 |
| `tests/test_run_context.py` | 12 |
| `tests/test_section_metrics.py` | 11 |
| `tests/test_pipeline_modes.py` | 8 |
| `tests/test_multiview_losses.py` | 5 |
| `tests/test_view_fusion.py` | 5 |
| `tests/test_stage1_objectives.py` | 4 |
| `tests/test_training_core.py` | 2 |
| **Total** | **141** |

### Ruff errors by rule and by directory

| Rule | Count | | Directory | Count |
|---|---|---|---|---|
| `W293` blank-line-with-whitespace | 218 | | `mhcac/` | 271 |
| `I001` unsorted-imports | 31 | | `vision_encoders/` | 50 |
| `UP008` super-call-with-parameters | 30 | | `biovil_t/` | 38 |
| `UP035` deprecated-import | 27 | | `utils/` | 29 |
| `UP045` non-pep604-optional | 27 | | `training/` | 10 |
| `F401` unused-import | 21 | | `tests/` | 5 |
| everything else | 73 | | `pretraining/`, `preporcessing/` | 6 |

The mass is cosmetic whitespace in the frozen encoder code, not in the Stage-2
code this round touches. `pyproject.toml` also names `UP038` in its ignore
list; that rule was removed from ruff and the setting is now inert.

### What this environment can and cannot execute

Installed: `torch 2.12.0+cpu`, `pandas 3.0.3`, `numpy 2.5.1`, `PIL 12.3.0`,
`matplotlib 3.11.1`, `omegaconf 2.3.0`, `pytest`, `ruff`.

**Missing: `transformers`, `peft`, `accelerate`, `bitsandbytes`, `torchvision`,
`timm`, `nltk`, `bert_score`, `pycocoevalcap`.**

Consequence, stated plainly: `training/train_eval_figure9_llm_variants_200.py`
and anything importing it **cannot be imported in this environment at all**, so
no test in the suite covers `VariantLLM`, the trainer, the generator or the NLG
metric code. The 141 passing tests cover pure-Python and pure-torch helpers
only. Any claim about MedGemma runtime behaviour in this round is static
reasoning, not execution.

## 3. Findings

Severity: **Critical** = wrong results or a stated architectural guarantee that
the code does not actually provide. **High** = blocks a stated round-2 goal.
**Medium** = correctness/maintainability risk. **Low** = hygiene.

| ID | Sev | File | Symbol | Problem | Consequence |
|---|---|---|---|---|---|
| S1 | Critical | `training/run_medgemma_qlora.py:48` | module import | `import train_eval_figure9_llm_variants_200 as fig9` runs at module scope. That module at lines 116-128 does `sys.path.insert` then imports `model.lavis.tasks`, `model.lavis.common.config.Config`, `registry` and `MIMIC_CXR_Dataset`. | `--pipeline-mode medgemma_direct` **cannot start without the entire LAVIS/Stage-1 stack importable**, contradicting the docstring at lines 3-8 and CLAUDE.md's "imports neither LAVIS nor the Stage-1 model". The file's own comment at lines 44-47 already admits this. Acceptance criterion 3 fails today. |
| S2 | Critical | `training/train_eval_figure9_llm_variants_200.py` | whole module | 1685 lines holding model loading, quantization, LoRA targeting, prompts, collation, soft-token injection, train loop, validation, generation, 5 NLG metrics, checkpointing, GCS upload, privacy verification and Figure-9 plotting. | Nothing in it is unit-testable in isolation; it is a hard dependency of the full-data pipeline (S1). Round-2 goals P1/P3 are blocked on splitting it. |
| S3 | High | same, `VariantLLM` (lines 651-1310, ~660 lines) | class | One class loads the model, configures NF4, attaches LoRA, builds prompts, encodes, collates, forwards, trains, early-stops, generates, saves adapters and writes manifests. | Cannot substitute a native reporter for a Q-Former reporter without carrying both code paths; `image_mode` branches appear in 9 separate methods. |
| S4 | High | same, `build_stage1_records:485` | function | `include_stage1_features=False` (the native path) still calls `build_cfg()` → LAVIS `Config`, and `make_stage1_loader()` → `MIMIC_CXR_Dataset`. | Even the *data* path for native mode goes through Stage-1 machinery when reached via this function. `run_medgemma_qlora.build_native_records` correctly bypasses it, so the dead native branch inside `build_stage1_records` is a trap for the next caller. |
| S5 | High | `training/` | — | No trainer, checkpoint-manager, evaluator or distributed module exists. Training state lives inside `VariantLLM.train_fine` (177 lines). | Resume correctness is untestable; round-2 goals P3/P5 have nothing to build on. |
| S6 | High | repo-wide | — | No counterfactual evaluator and no clinical-metric adapter exist. `safety/` implements claim extraction/verification/reconciliation only. | Goals P4/P6 are greenfield, not touch-ups. |
| S7 | Medium | `train_eval_figure9...py:366` | `field_value` | `except Exception: return str(field)` swallows every error including `KeyError` on a malformed batch. | A structurally wrong Stage-1 batch is silently stringified into the prompt instead of failing. |
| S8 | Medium | same:726 | model load | `except Exception:` falls back from `AutoModelForImageTextToText` to `AutoModelForCausalLM` with no logging. | If the multimodal class fails for an unrelated reason (auth, OOM), the run silently continues as a text-only model — native image mode would then be a language-prior baseline mislabelled as a vision baseline. |
| S9 | Medium | same:1336-1351 | `compute_nlg` | Four nested bare `except Exception` around metric computation. | A metric that errors is reported as absent rather than as failed. |
| S10 | Medium | `pyproject.toml` | ruff config | `UP038` is ignored but no longer exists; `mypy` is configured nowhere despite being in the round-2 brief. | Silent no-op config; type checking claimed but absent. |
| S11 | Low | `inference.py:312` | Vicuna load | `device_map={"": 0}` hard-codes GPU 0. | Blocks goal P5's "no hard-coded GPU 0". `inference.py` is outside the Stage-2 path but is the one remaining literal-0 site. |
| S12 | Low | `pretraining/precompute_features.py:101,152` | — | `.cuda()` calls. | Same as S11; single-GPU script, but named explicitly by the brief. |
| S13 | Low | `mhcac/`, `vision_encoders/`, `biovil_t/` | — | 359 of 427 ruff errors, mostly `W293`. | Cosmetic. Frozen encoder code; reformatting would obscure future upstream diffs, same argument that already excludes `model/lavis`. |

### Findings deliberately NOT raised

- **`min(batch_idx, ...)` clamp** — already removed in round 1. The two grep
  hits (`stage2_utils.py:144`, `fig9:627`) are comments explaining the removal,
  and `SoftTokenEmbeddingWrapper.forward` now calls `validate_soft_token_batch`
  per sample. Verified by reading lines 618-637.
- **`import *`** — only in `pretraining/train.py` and
  `precompute_features.py`, where LAVIS registry population genuinely requires
  it. Already excepted in `pyproject.toml`.
- **Mutable global config** — removed in round 1 by `Stage1Context`
  (`76677ba`). The remaining `global` statements are all in `inference.py`'s
  Gradio callbacks, which is out of scope.
- **`device_map` in fig9:721** — resolves `torch.cuda.current_device()`, not a
  literal 0, and is guarded by a comment explaining the single-GPU intent.

## 4. Round-2 execution order

Derived from the dependency graph, not from the brief's numbering: S1 cannot be
fixed without first moving the LAVIS-importing functions out of fig9, and the
trainer cannot be extracted before the reporter split.

1. Move every LAVIS/Stage-1-importing symbol into `training/stage1/`, leaving
   fig9 free of module-scope Stage-1 imports → closes S1, S4.
2. Extract MedGemma loading/LoRA/soft-token/prompt code into
   `training/medgemma/` → starts S2, S3.
3. Extract trainer + checkpoint manager → S5.
4. Reduce fig9 to orchestration → closes S2.
5. Counterfactual evaluator, clinical-metric adapters, distributed context →
   S6.

Each step lands as its own commit with its own test run.
