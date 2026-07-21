# Phase A — Findings-first inference over an external MedGemma checkpoint

Status: **code complete, never run on a GPU.** Everything below is verified by
CPU tests with a fake model. Nothing here is evidence that MedGemma runs.

## What this phase is

Report generation now uses a **third-party, already-fine-tuned** MedGemma
checkpoint. This project does not fine-tune MedGemma.

This is *not* a repository-wide move to inference-only. The project has two
tiers:

| Tier | Component | Trainable by this project? |
|---|---|---|
| A. Report generation | `erjui/medgemma-4b-srrg-findings` (external) | **No** — inference only |
| B. Research verifier | META-CXR / MHCAC Stage 1 | **Yes** — still trainable, unchanged |

Stage 1 remains the project's own research contribution and stays fully
trainable. `torch.optim`, `.backward()` and `model.train()` are legitimate under
`pretraining/`, `mhcac/`, `model/lavis/` and are *not* banned there.

## Model provenance — state it accurately

- `erjui/medgemma-4b-srrg-findings` and `erjui/medgemma-4b-srrg-impression`
- Fine-tuned by a third party (erjui) from `google/medgemma-4b-it`
- Trained on the `erjui/csrrg_ift_dataset`, which the model card describes as
  derived from **MIMIC-CXR and CheXpert+**
- **Not** fine-tuned by this repository
- **Not** trained on this repository's splits

Because the training mixture includes MIMIC-CXR, our MIMIC-CXR test split may
overlap the checkpoint's training data. Any score obtained here must be
reported as an **external-baseline evaluation with possible train/test
contamination**, not as a clean held-out result. This is a limitation to state,
not one to work around.

### Distribution format (verified against the HF API, 2026-07-21)

The Findings repo ships **merged weights** — `model-0000{1,2}-of-00002.safetensors`
plus `model.safetensors.index.json` — and has **no `adapter_config.json`**.

Consequences:
- **PEFT is not required** to load it. The `lora` tag on the model card
  describes how the authors trained it, not how it is distributed.
- `bitsandbytes` is needed **only** if you set `load_in_4bit: true`.

Config: `architectures: ["Gemma3ForConditionalGeneration"]`, `model_type: gemma3`,
SigLIP vision tower at 896×896, 256 image tokens, published in bfloat16,
`transformers_version: 4.51.3`. Loading needs **transformers ≥ 4.50**.

The processor is loaded from the checkpoint repo itself, not from
`google/medgemma-4b-it`: the repo ships its own tokenizer, `added_tokens.json`
and `preprocessor_config.json`, and the base repo is gated. This keeps the
processor consistent with the weights and avoids a gated dependency.

## Layout

```
model/pretrained_medgemma/     external checkpoint: loader, reporter, schema, errors
medgemma_inference/            config, runner, progress/resume, writer, CLI
runtime/                       budget controller, device/dtype resolution
configs/experiments/           pretrained_medgemma_findings_first.yaml
```

`src/meta_cxr/` was **not** adopted. The repo uses flat top-level packages and
repackaging it would touch every import for no functional gain. It stays a
possible long-term refactor, not a current one.

## Commands

```bash
# 10-study smoke run
python -m medgemma_inference.run_pretrained_findings \
  --config configs/experiments/pretrained_medgemma_findings_first.yaml \
  --split validation --max-samples 10

# 100-study pilot, projecting cost onto the whole split
python -m medgemma_inference.run_pretrained_findings \
  --config configs/experiments/pretrained_medgemma_findings_first.yaml \
  --split validation --max-samples 100 --estimate-full-cost

# full split — refuses to start without --confirm-full-run
python -m medgemma_inference.run_pretrained_findings \
  --config configs/experiments/pretrained_medgemma_findings_first.yaml \
  --split test --max-samples all --confirm-full-run
```

Re-running the same command resumes: completed `sample_key`s are skipped, and a
fully-resumed run loads no model at all.

## Impression is disabled

Phase 2 is blocked in three independent places:

1. `configs/…yaml` sets `models.impression.enabled: false` and
   `evaluation.run_impression: false`.
2. Config validation raises `ConfigError` if either is true.
3. `assert_impression_disabled()` raises `ImpressionPhaseDisabledError` in the
   runner **before any model is loaded**, and `PretrainedImpressionReporter`
   raises on construction.

There is no flag that bypasses this. `impression_reporter.py` imports neither
torch nor transformers and contains no `from_pretrained` call — asserted by an
AST test, so the guarantee cannot rot.

Only one model is ever resident in VRAM: the Impression checkpoint is never
downloaded, constructed, or allocated.

## Privacy

Prediction records carry a salted `sample_key` digest and nothing else
identifying. `subject_id`, `study_id`, `dicom_id`, `image_path` and the
reference report are rejected by `assert_publishable()` at write time, and the
config refuses to enable them. This follows the PhysioNet DUA: derivatives
carrying identifiers or report text are restricted data.

## What is NOT verified

- That the checkpoint loads. No GPU, no `transformers` on the dev box.
- That generation produces clinically sensible FINDINGS.
- Any throughput or cost figure. `cost_estimate_findings.json` is written from
  measured wall-clock only; there are no placeholder numbers anywhere.

See `docs/gpu_pilot_checklist.md`.
