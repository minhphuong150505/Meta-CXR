# Migration guide

Import paths, CLI flags and config keys that changed, and what to do about it.

## Round 2 (branch `refactor/clean-medgemma-xai-pipeline`)

### Moved symbols

| Old | New | Deprecated from | How to migrate |
|---|---|---|---|
| `train_eval_figure9_llm_variants_200.SoftTokenEmbeddingWrapper` | `training.medgemma.soft_tokens.SoftTokenEmbeddingWrapper` | `709f9bb` | Import from the new path. The old name is still re-exported from the Figure-9 module, so existing code keeps working. New argument `num_img_tokens` is the 4th positional parameter; it defaults to 32, matching the old module-global `NUM_IMG_TOKENS`. |
| `train_eval_figure9_llm_variants_200.build_cfg` | `training.stage1.lavis_loader.build_cfg` | `5d572d8` | The old name remains as a lazy delegate. Prefer the new path when you already know you are on the Stage-1 branch. |
| `train_eval_figure9_llm_variants_200.build_stage1_model` | `training.stage1.lavis_loader.build_stage1_model` | `5d572d8` | Same. |
| `train_eval_figure9_llm_variants_200.make_stage1_loader` | `training.stage1.lavis_loader.make_stage1_loader` | `5d572d8` | Same. |
| `train_eval_figure9_llm_variants_200.load_torch_checkpoint` | `training.torch_io.load_torch_checkpoint` | `5d572d8` | Re-exported; no action required. |
| `train_eval_figure9_llm_variants_200.filter_state_dict_for_model` | `training.stage1.lavis_loader.filter_state_dict_for_model` | `5d572d8` | **Not re-exported.** Only the Stage-1 loader used it. Update the import if you called it directly. |
| `train_eval_figure9_llm_variants_200.load_state_dict_materializing_meta` | `training.stage1.lavis_loader.load_state_dict_materializing_meta` | `5d572d8` | **Not re-exported.** Same. |

### Removed arguments

| Old | Removed in | Why | What to do |
|---|---|---|---|
| `build_stage1_records(..., include_stage1_features=False)` | `5d572d8` | The `False` branch still built a LAVIS `Config` and a `MIMIC_CXR_Dataset` loader, so a caller "opting out of Stage-1" stayed fully coupled to it. Every caller in the repo passed `True`. | For native MedGemma records use `run_medgemma_qlora.build_native_records`, which reads the split CSVs through `training/dataio/manifest.py`. Passing `include_stage1_features=True` is now the only behaviour; drop the keyword. |

`build_stage1_records` is now Stage-1-only by contract, and its docstring says
so. It raises the ordinary `TypeError` on the removed keyword rather than a
`DeprecationWarning`, because silently accepting it would mean silently
producing Q-Former records for a caller who asked for native ones.

### Pipeline-mode aliases (unchanged, still deprecated)

Introduced before round 2 and still honoured. `--image-mode` prints a
deprecation line and maps to `--pipeline-mode`:

| Old `--image-mode` | New `--pipeline-mode` |
|---|---|
| `native` | `medgemma_direct` |
| `qformer` | `meta_cxr_qformer` |
| `both` | `both_for_ablation` |

Passing both flags is an error, not a silent precedence rule. Removal is not
scheduled; the aliases cost one dict lookup and existing runbooks use them.

### New modules (nothing to migrate)

| Module | Purpose |
|---|---|
| `training/stage1/lavis_loader.py` | The only place Stage-2 code may import LAVIS. |
| `training/medgemma/soft_tokens.py` | Soft-token substitution. |
| `training/evaluation/perturbations.py` | Counterfactual image perturbations. |
| `training/evaluation/counterfactual.py` | Counterfactual audit evaluator. |
| `training/evaluation/clinical.py` | Optional clinical-metric adapters. |
| `training/trainer/state.py` | RNG capture, run counters, provenance. |
| `training/trainer/checkpointing.py` | Atomic checkpoint save/load. |
| `training/torch_io.py` | Shared checkpoint loader. |

### Rule that new code must follow

**No Stage-2 module may import `model.lavis`, `mhcac`, `biovil_t` or
`vision_encoders` at module scope.** Import
`training.stage1.lavis_loader` from inside the branch that has already
established it needs Stage-1. `tests/test_native_independence.py` enforces
this with a meta-path finder and will fail the build if it is violated.

## Not changed in round 2

Listed so you do not go looking:

- No config key was renamed.
- No CLI flag was removed.
- No directory was renamed. `preporcessing/` keeps its misspelling.
- `inference.py` still runs Vicuna-7B and was not migrated.
- `configs/env_config.yaml` handling is unchanged — still tracked, still needs
  reverting before a commit.
