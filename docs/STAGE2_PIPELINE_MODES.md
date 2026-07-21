# Stage-2 pipeline modes

Status: code-complete on CPU, **never executed on a GPU**. Every shape and command
below is derived from the source, not from an observed training run.

## The three architectures

### 1. `medgemma_direct` — the default

```
CXR JPG
  → AutoProcessor (MedGemma)
  → native MedGemma image tower
  → native multimodal projector
  → Gemma decoder (+ QLoRA NF4 on language layers only)
  → FINDINGS + IMPRESSION
```

No Stage-1 checkpoint, no Stage-1 config, no Q-Former, no MHCAC, no structured
abnormality text in the prompt. The image is the only clinical evidence at
inference. Records come from `training/dataio/manifest.py`, which reads the split
CSVs directly and never imports LAVIS or the Stage-1 model.

### 2. `meta_cxr_qformer` — hybrid ablation

```
CXR JPG
  → BioViL-T [B,P,1408] ─┐
  → PubMedCLIP [B,P,768] ─┼→ per-encoder FC → concat(token axis) → [B,ΣP,1408]
  → SwinV2 [B,P,1024] ────┘
  → Q-Former (32 query tokens)          → [B,32,768]
  → trainable img_proj (fp32 Linear)    → [B,32,hidden]
  → substituted at 32 <qformer_soft_token> positions
  → Gemma decoder (+ QLoRA)
```

Requires a Stage-1 checkpoint. **This is not native MedGemma** — the mode name
now says so. `meta_cxr_qformer_with_mhcac_prompt` additionally injects MHCAC
P/N/U findings into the prompt as text.

### 3. `both_for_ablation`

Runs `medgemma_direct` first, then `meta_cxr_qformer`, so a crash in the
ablation still leaves the primary result on disk. Requires Stage-1 because the
hybrid does.

## Section modes

| Mode | Target format |
|---|---|
| `findings_only` | bare FINDINGS text (legacy behaviour) |
| `impression_only` | bare IMPRESSION text |
| `findings_and_impression` (**default**) | `FINDINGS: …\n\nIMPRESSION: …` |

`findings_and_impression` requires **both** sections to be valid; a row with only
one is dropped rather than trained with an empty section. `split_generated_report`
treats an unheadered generation as FINDINGS with an empty IMPRESSION, so an
omitted IMPRESSION scores as a miss instead of being silently duplicated.

`max_new_tokens` auto-sizes to 512 for `findings_and_impression`, 256 otherwise.

### Constraint: Q-Former modes are FINDINGS-only

`ReportDataset.text_output` emits FINDINGS only, so any Stage-1 Q-Former mode
must use `--section-mode findings_only`. The run exits with an explanatory error
rather than silently training the hybrid on a different target than the primary
pipeline.

## Migration — data contract change

`preporcessing/preprocess_mimic_cxr.py` now emits four new columns:

- `impression_clean`, `impression_valid`, `impression_token_count`

**Manifests built before this change cannot serve `impression_only` or
`findings_and_impression`.** `assert_columns` fails with an explicit message
naming the missing columns. To use the default section mode you must re-run
preprocessing and re-upload the split CSVs:

```bash
python preporcessing/preprocess_mimic_cxr.py \
    --raw-dir ~/data/mimic-cxr-raw \
    --reports-root ../Report/mimic-cxr-reports/files \
    --output-dir ~/data/mimic-cxr-processed/full_allviews
```

IMPRESSION is parsed from an explicit `IMPRESSION:`/`CONCLUSION:` tag only — it
is never recovered from the narrative body, so an empty value means the report
genuinely had no impression section. FINDINGS and IMPRESSION get **separate**
train-derived length bounds because their length distributions differ.

## Commands

```bash
# Validate the manifests training will actually consume (fail-fast on leakage)
python -m training.dataio.validate_manifest --section-mode findings_and_impression
python -m training.dataio.validate_manifest --vis-root /mnt/mimic --image-sample 500

# Default: native MedGemma, findings + impression, single GPU
python training/run_medgemma_qlora.py \
    --output-dir training/outputs/medgemma_direct

# Hybrid ablation (needs Stage-1; FINDINGS only)
python training/run_medgemma_qlora.py \
    --pipeline-mode meta_cxr_qformer \
    --section-mode findings_only \
    --checkpoint-root <dir>

# Both, for the ablation table
python training/run_medgemma_qlora.py \
    --pipeline-mode both_for_ablation --section-mode findings_only \
    --checkpoint-root <dir>

# Resume (adapter dir is named after the pipeline mode)
python training/run_medgemma_qlora.py \
    --resume-from training/outputs/.../adapters/medgemma_qlora_medgemma_direct/checkpoints/last

# Tests
python -m pytest tests/ training/test_stage2_utils.py    # 67 CPU tests
```

The deprecated `--image-mode {native,qformer,both}` still works and prints the
`--pipeline-mode` it maps to.

## Single-GPU only

`device_map` pins to `torch.cuda.current_device()`, so `CUDA_VISIBLE_DEVICES`
selects the GPU. There is **no DDP, FSDP or DistributedSampler** in this
repository — multi-GPU is not supported and no multi-GPU claim should be made.

## Known gaps

- Checkpoint selection is validation **cross-entropy**, not a clinical metric.
- No RadGraph / CheXbert / RadCliQ / GREEN implementation exists here.
- No hallucination, grounding, uncertainty, abstention or counterfactual audit.
- `threshold.json` carries no provenance and is never loaded implicitly.
- `run_medgemma_qlora.py` still imports the Figure-9 module for `VariantLLM` and
  the evaluation helpers, so LAVIS/torch/transformers load even in
  `medgemma_direct`. The *data* path and Stage-1 *requirement* are decoupled;
  extracting `VariantLLM` into `training/trainers/` is the next step.
