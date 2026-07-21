# Test report — refactor round 2

Every number here comes from a command actually run in this environment. Where
something could not be executed, it is listed as not executed rather than
inferred.

## Environment

| Field | Value |
|---|---|
| OS | Linux 6.17.0-40-generic |
| Python | 3.12.3 |
| Interpreter | `/home/phuong/venv/bin/python` |
| Branch | `refactor/clean-medgemma-xai-pipeline` |
| Base SHA | `ee3ba2f` |
| Final SHA | see `git log`; last code commit `103fb95` |
| Working tree | clean at each commit boundary |

### Installed

`torch 2.12.0+cpu`, `pandas 3.0.3`, `numpy 2.5.1`, `PIL 12.3.0`,
`matplotlib 3.11.1`, `omegaconf 2.3.0`, `pytest`, `ruff 0.15.22`.

`ruff` was installed during this pass. It is a self-contained binary wheel with
no Python dependencies, so it cannot perturb the torch pin.

### Not installed

`transformers`, `peft`, `accelerate`, `bitsandbytes`, `torchvision`, `timm`,
`nltk`, `bert_score`, `pycocoevalcap`, `mypy`.

Installing any of the first six pulls a multi-GB CUDA stack that would upgrade
torch, which `CLAUDE.md` forbids for this venv.

## Commands run

```bash
/home/phuong/venv/bin/python -m compileall -q .
/home/phuong/venv/bin/python -m pytest
/home/phuong/venv/bin/ruff check .
/home/phuong/venv/bin/ruff format --check .
```

## Results

| Command | Baseline (`ee3ba2f`) | Final |
|---|---|---|
| `compileall -q .` | exit 0 | **exit 0** |
| `pytest` | 141 passed, 0 failed, 0 skipped, 2.6 s | **229 passed, 0 failed, 0 skipped, 3.3 s** |
| `ruff check .` | 427 errors | **427 errors** |
| `ruff format --check .` | 59 would reformat | 59 would reformat |
| `mypy` | not installed | not installed |

Zero skipped tests, both before and after.

### Test count by file

| File | Baseline | Final | Δ |
|---|---:|---:|---:|
| `test_safety_pipeline.py` | 31 | 31 | — |
| `test_counterfactual.py` | — | 31 | **+31** |
| `test_shared_visual_tokens.py` | 20 | 20 | — |
| `test_trainer_resume.py` | — | 20 | **+20** |
| `test_manifest.py` | 17 | 17 | — |
| `test_clinical_metrics.py` | — | 17 | **+17** |
| `test_soft_token_injection.py` | — | 14 | **+14** |
| `test_mimic_data_pipeline.py` | 13 | 13 | — |
| `test_stage2_utils.py` | 13 | 13 | — |
| `test_run_context.py` | 12 | 12 | — |
| `test_section_metrics.py` | 11 | 11 | — |
| `test_pipeline_modes.py` | 8 | 8 | — |
| `test_native_independence.py` | — | 6 | **+6** |
| `test_multiview_losses.py` | 5 | 5 | — |
| `test_view_fusion.py` | 5 | 5 | — |
| `test_stage1_objectives.py` | 4 | 4 | — |
| `test_training_core.py` | 2 | 2 | — |
| **Total** | **141** | **229** | **+88** |

No existing test was deleted, renamed or weakened.

### Lint

427 → 427. The count is unchanged because the 88 new tests and 8 new modules
are all clean, and one orphaned import that this refactor created
(`validate_soft_token_batch` in the Figure-9 module, after the injector moved
out) was removed.

```
ruff check training/stage1 training/medgemma training/evaluation \
           training/trainer training/torch_io.py     → All checks passed
```

The 5 lint errors remaining under `tests/` are in round-1 files
(`test_mimic_data_pipeline.py`, `test_multiview_losses.py`,
`test_view_fusion.py`) and were not touched. The other 422 sit in
`mhcac/`, `vision_encoders/`, `biovil_t/` and `utils/`, 218 of them being
`W293` blank-line whitespace in frozen encoder code.

## What the tests actually cover

### Executed against real objects

| Area | Evidence |
|---|---|
| Soft-token injection | 14 tests on real `nn.Embedding` and tensors: per-sample distinctness, batch mismatch, missing/extra/partial placeholders, hidden-size mismatch, wrong rank, text-only passthrough, gradient flow to projector *and* base table, fp32→fp16 casting. |
| Resume correctness | Toy model trained 12 steps uninterrupted vs 6 + checkpoint + resume + 6, asserted bit-identical (`rtol=0, atol=0`) with identical scheduler LR. A negative control confirms the assertion is strict. |
| RNG capture | All four streams (Python, NumPy, torch CPU, DataLoader generator) round-tripped through save/restore. |
| Counterfactual audit | 31 tests with deterministic fake backends, including an image-dependent generator (passes) and a language-prior generator (correctly flagged). |
| Perturbations | Histogram/value-multiset preservation, determinism under seed, donor never being the sample itself. |
| Clinical adapters | Missing dependency raises with install command; installed-but-unwired raises `NotImplementedError`; no code path returns a number. |
| Privacy | Recursive detection of 6 forbidden identifier keys nested inside lists inside dicts. |
| Stage-1 independence | Meta-path finder raising on any `model.lavis`/`mhcac`/`biovil_t`/`vision_encoders` import, with a positive control proving the guard is not vacuous. |

### Static or mocked only — NOT executed

| Area | Why |
|---|---|
| `VariantLLM` construction, NF4 quantization, LoRA attachment | needs `transformers`, `peft`, `bitsandbytes` |
| `train_fine` training loop | same, plus MedGemma weights |
| `generate` | same |
| `compute_nlg` / `compute_sectioned_nlg` | needs `nltk`, `bert_score`, `pycocoevalcap` |
| Stage-1 model build, `MIMIC_CXR_Dataset` iteration | needs `torchvision`, `timm`, LAVIS stack, MIMIC-CXR images |
| GCS upload and bucket privacy verification | needs `gcloud` and credentials |
| Anything end-to-end on GPU | no GPU in this environment |

The Stage-1 independence test proves the *import graph* is correct. It does not
prove a native training run succeeds — that requires MedGemma weights and a GPU.

## Dependencies a full validation would still need

| Requirement | Needed for |
|---|---|
| NVIDIA GPU (L4) | any training or generation |
| `requirements-stage2.txt` env | `VariantLLM`, trainer, NLG metrics |
| MedGemma 1.5 4B-it weights + HF token | Stage-2 runs |
| MIMIC-CXR-JPG + credentialed GCS access | any data-dependent test |
| Stage-1 `checkpoint_best.pth` | `meta_cxr_qformer` mode only |
| RadGraph / CheXbert weights | clinical metrics — **never run here** |

## Statements this report does not make

- **Two-GPU DDP has not been run.** No distributed code was added in round 2,
  and no claim about multi-GPU behaviour is made.
- **No RadGraph, CheXbert, CheXpert-labeler or RadCliq score exists.** No such
  model has been executed in this repository, at any point.
- **No NLG benchmark number was produced.** The metric code was not executed.
- **The trainer and checkpoint manager are not yet wired into `train_fine`.**
  They are tested standalone. The swap is deferred because it cannot be
  executed here, and moving a 177-line untestable loop blind is how a refactor
  introduces the bug it was meant to prevent.
