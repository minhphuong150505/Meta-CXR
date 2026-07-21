# Pending: MedGemma fine-tuning teardown (Phase B)

**Nothing in this document has been executed.** Phase A deleted no files. This
is the audited plan, gated on `docs/gpu_pilot_checklist.md` passing.

## Scope

Delete **MedGemma fine-tuning only**. Stage-1 META-CXR/MHCAC stays fully
trainable — it is the project's research contribution and the verifier/XAI
baseline, and its checkpoints need to be reproducible since the old GCS bucket
was deleted.

Backup tag (pushed): **`before-pretrained-inference-only-4bc10f9`**

## Audit

Classification is from the import graph, not from filenames. Several files whose
names contain `train`, `trainer` or `checkpoint` are **Stage-1 dependencies and
must survive**.

### DELETE — MedGemma fine-tuning only

| File | Lines | Why | Importers | Replacement |
|---|---|---|---|---|
| `training/run_medgemma_qlora.py` | 489 | MedGemma QLoRA fine-tuning CLI | none | `medgemma_inference.run_pretrained_findings` |
| `training/train_eval_figure9_llm_variants_200.py` | 1659 | Stage-2 training god script | none | inference/metric parts to extract first |
| `training/trainer/state.py` | 174 | MedGemma training state (optimizer/scheduler/RNG) | `training/trainer/checkpointing.py`, `tests/test_trainer_resume.py` | `medgemma_inference/progress.py` (inference resume) |
| `training/trainer/checkpointing.py` | 120 | MedGemma training checkpoint writer | `tests/test_trainer_resume.py` | — |
| `tests/test_trainer_resume.py` | — | tests the above only | — | `tests/test_pretrained_findings.py::ResumeInference` |

`train_eval_figure9_llm_variants_200.py` must not be kept as a compatibility
layer. Extract anything valuable (NLG metric wiring, evaluation loop) into a
real module **before** deleting, then delete outright.

### KEEP — despite a training-sounding name

| File | Why it must survive |
|---|---|
| `tests/test_training_core.py` | Tests `model/lavis/common/optims.py` — the **Stage-1** LR scheduler, not MedGemma |
| `training/torch_io.py` | Imported by `training/stage1/lavis_loader.py` |
| `training/stage1/lavis_loader.py` | Stage-1 checkpoint loading |
| `training/dataio/manifest.py` | Split reader + anchor selection; used by the **new** Findings CLI |
| `training/dataio/validate_manifest.py` | Split leakage validation |
| `training/stage2_utils.py` | Provides `stable_fingerprint`, used by `manifest.py` → new CLI. Trim FT-only helpers, do not delete |
| `training/evaluation/clinical.py` | Model-agnostic metric adapters; imported by `model/lavis/data/ReportDataset.py` (Stage 1) |
| `training/evaluation/counterfactual.py` | Model-agnostic audit; required by the new pipeline |
| `training/evaluation/perturbations.py` | Used by counterfactual |
| `training/medgemma/capabilities.py` | Multimodal capability check — inference-time |
| `training/medgemma/soft_tokens.py` | Q-Former soft-token **injection** at inference, Stage-1 dependent |
| `training/run_context.py` | `Stage1Context`; keep while the hybrid verifier path exists |
| `training/pipeline_modes.py` | Now also registers the new external-inference modes |
| `pretraining/**`, `mhcac/**`, `biovil_t/**`, `vision_encoders/**`, `model/lavis/**` | Stage-1, untouched |
| `safety/**` | Pure-Python, model-agnostic |

Once `training/trainer/` is emptied, remove the directory.

### Dependency table

| Package | Fine-tuning use | Inference use | Verdict |
|---|---|---|---|
| `transformers` | yes | **yes** — loads the checkpoint | keep |
| `torch` | yes | **yes** | keep |
| `peft` | LoRA training | **no** — checkpoint ships merged weights, no `adapter_config.json` | **removable** for MedGemma; re-check if a future checkpoint is adapter-only |
| `bitsandbytes` | QLoRA training | **only if** `load_in_4bit: true` | keep as optional `inference` extra |
| `accelerate` | training launch | device placement via `device_map` | keep |
| `datasets` | training data | not used by the new runner | check Stage-1 first |
| `evaluate`, `nltk`, `bert_score`, `pycocoevalcap` | no | yes — NLG metrics | keep under `evaluation` |
| `radgraph`, `chexbert` | — | not implemented in this repo | do not add to docs or tables |

`peft` is the one to be careful with: it is removable **because these specific
checkpoints ship merged**, not because the project stopped fine-tuning. Verify
before removing.

### Docs to rewrite or delete

| Document | Action |
|---|---|
| `docs/STAGE2_PIPELINE_MODES.md` | rewrite — drop MedGemma FT modes, keep Stage-1 verifier modes |
| `docs/medgemma_real_runtime_smoke.md` | delete — describes the FT runtime |
| `docs/round4_baseline.md`, `docs/round5_baseline.md` | delete — FT run baselines |
| `docs/CHECKPOINT_WORKFLOW.md` | rewrite — Stage-1 checkpoints only |
| `docs/migration_guide.md` | supersede with `docs/migration_to_pretrained_inference.md` |
| `docs/README.md`, root `README.md` | rewrite per §18 |
| `docs/cloud/*` | audit — Stage-1 training runbooks stay, Stage-2 FT ones go |
| `requirements-stage2.txt` | rewrite as inference requirements |

No `old_*.md` / `legacy_*.md` / `deprecated_*.md` files. Git history and the tag
preserve them.

## Commit plan (Phase B)

1. `docs: declare external-checkpoint inference direction for MedGemma`
2. `refactor: extract evaluation from the figure-9 script`
3. `refactor: remove medgemma fine-tuning entrypoints`
4. `refactor: remove medgemma training-only implementation`
5. `refactor: remove obsolete medgemma training configuration`
6. `test: replace medgemma training tests with inference invariants`
7. `docs: rewrite README around external inference + stage-1 verifier`
8. `chore: drop training-only dependencies and CI jobs`
9. `security: update artifact and model-cache ignores`
10. `docs: add migration to pretrained inference`

After each: `pytest -q` on the affected tests, `compileall`, `ruff check` on
changed files only. Do not reformat the repository.

## Do not start until

Every box in `docs/gpu_pilot_checklist.md` §3 is ticked and the projected
full-run cost is explicitly approved. The replacement must be proven to work
before the thing it replaces is deleted.
