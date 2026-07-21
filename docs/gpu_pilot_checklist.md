# GPU pilot checklist — RTX 3090

Run this before anything is deleted. The Phase-B teardown is gated on it.

## 0. Environment

The dev box has no GPU and no `transformers`; every Phase-A test uses a fake
model. This checklist is the first time real MedGemma code runs.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"   # need >= 4.50
nvidia-smi --query-gpu=name,memory.total --format=csv
```

Gemma3 multimodal support landed in transformers 4.50; the checkpoint declares
4.51.3. An older transformers will fail in `_model_class()` with an explicit
message rather than silently loading something text-only.

### VRAM

24 GB on a 3090. A 4B model in bfloat16 is ~8 GB of weights, plus the SigLIP
tower and KV cache for a 256-image-token prompt. bf16 should fit comfortably.
Set `load_in_4bit: true` **only** if it does not — that needs `bitsandbytes`, and
`check_4bit_available()` will refuse rather than silently running unquantised.

The 3090 is Ampere, so `torch.cuda.is_bf16_supported()` is true and `dtype: auto`
resolves to bfloat16 — matching the checkpoint's published dtype.

## 1. Model access

Both repos are public and ungated (`"gated": false`), so no HF token is needed.
The base `google/medgemma-4b-it` **is** gated, which is why the processor is
loaded from the checkpoint repo instead.

First load downloads ~8 GB. Set `HF_HOME` to a disk with room.

## 2. Ordered pilot

Do not skip ahead. Each step gates the next.

| Step | Command | Gate |
|---|---|---|
| 1 | `--split validation --max-samples 1` | loads, generates non-empty FINDINGS |
| 2 | `--split validation --max-samples 10` | 10/10 succeed, no `unexpected_impression_generated` |
| 3 | `--split validation --max-samples 100 --estimate-full-cost` | cost estimate written |
| 4 | `--split validation --max-samples 500 --estimate-full-cost` | rate stable vs. step 3 |
| 5 | review `cost_estimate_findings.json` | projected full-split cost acceptable |
| 6 | full run, `--confirm-full-run` | only after 5 |

## 3. Acceptance gates for starting Phase B

All must hold:

- [ ] `config.json` and processor load from `erjui/medgemma-4b-srrg-findings`
- [ ] The multimodal class resolves; `NotMultimodalError` is **not** raised
- [ ] A real image reaches `generate()` — the vision tower is genuinely used
- [ ] ≥ 10 pilot samples complete successfully
- [ ] FINDINGS output is non-empty and clinically plausible on inspection
- [ ] No `unexpected_impression_generated` warnings (or an understood rate)
- [ ] Resume verified: kill mid-run, re-run, no sample is regenerated
- [ ] Budget stop verified: a low `budget_limit_usd` stops cleanly with valid JSONL
- [ ] `cost_estimate_findings.json` holds real measured numbers
- [ ] Projected full-run cost reviewed and **explicitly approved**

## 4. Sanity check the output

The model card documents this shape:

```
FINDINGS:
Lungs and Airways:
- No pleural effusion or pneumothorax detected
Cardiovascular:
- Mild left ventricular enlargement
```

The postprocessor strips the `FINDINGS:` header. Section sub-headers are kept —
they are part of the structured-report format this checkpoint was trained for.

Watch for:
- **An IMPRESSION section.** Dropped and warned, never silently kept.
- **Empty output** → `empty_findings` warning; check the chat template applied.
- **Prompt echo.** Should be impossible: only tokens past `prompt_length` are
  decoded. If it appears, the processor's template disagrees with the model's.

## 5. If it fails

Every failure mode raises rather than degrading:

| Symptom | Error | Meaning |
|---|---|---|
| no vision tower | `NotMultimodalError` | wrong class or wrong checkpoint |
| no image processor | `NotMultimodalError` | processor mismatch |
| transformers too old | `FindingsModelLoadError` | upgrade; do not downgrade the class |
| CUDA requested, absent | `RuntimeError` | no silent CPU fallback |
| 4-bit without bitsandbytes | `QuantizationUnavailable` | install it or use bf16 |

Do not "fix" any of these by loosening the check. A text-only fallback would
produce reports that never saw the X-ray — the single most expensive failure
available here.
