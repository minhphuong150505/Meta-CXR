# Stage-2 Prompt Audit (as-shipped, before optimisation)

Scope: every place a string reaches the Stage-2 LLM (MedGemma / Vicuna). All line
references are to the tree on branch `feature/optimize-stage2-prompts` at audit time.
This document describes the **existing** behaviour; the new builder is documented in
`stage2_prompt_design.md`.

## Where prompts are built

| Concern | Location |
|---|---|
| Class assignment P/N/U | `training/train_eval_figure9_llm_variants_200.py:339` `classify_with_thresholds` |
| Structured-findings string | `…:354` `format_findings` |
| Soft-token block (`<qformer_soft_token>`×32) | `…:362` `image_block` |
| Guided instruction (styles `fine`/default) | `…:366` `build_instruction` |
| Native (image-only) instruction | `…:389` `build_native_instruction` → `training/stage2_utils.py:236` `native_findings_instruction` |
| Assembled guided prompt | `…:394` `build_prompt` |
| MedGemma chat assembly + masking | `VariantLLM._chat_texts:861`, `_native_messages:896`, `_native_chat_inputs:919`, `encode_train_example:937` |
| Inference | `VariantLLM.generate:1227` |
| Legacy Vicuna JSON prompter (inference.py only) | `utils/prompter.py` |

Constants: `NUM_IMG_TOKENS = 32`, `ABNORMALITIES_14` (No Finding + 13 findings incl.
Support Devices), `CLASS_MAP = {negative:0, positive:1, uncertain:2}`.

## 1. Current prompt — Q-Former guided mode (`meta_cxr_qformer`, `image_mode=qformer`)

Training uses `prompt_style="fine"` (`collate_train:980`). The user message text is:

```
Image information: <qformer_soft_token> <qformer_soft_token> … (×32).

Abnormality information: Positive findings: <pos…>. Negative findings: <neg…>. Uncertain findings: <unc…>

Act as an expert radiologist. Write only the Findings section of a chest X-ray report
as one concise clinical paragraph. Do not invent facts, add an Impression section, or
repeat the structured findings.
```

This whole string is one `user` turn; `processor.apply_chat_template` wraps it with the
MedGemma turn markers. The `<qformer_soft_token>` placeholders are replaced at forward
time by projected Q-Former vectors (`SoftTokenEmbeddingWrapper`).

The default (non-`fine`) style (`build_instruction:375`) additionally hard-instructs:
"Do not invent findings. **Only describe abnormalities explicitly provided in the
'Abnormality information'.**"

## 2. Current prompt — native mode (`medgemma_direct`, `image_mode=native`, the default)

User message = `[{"type":"image", "image":<PIL>}, {"type":"text","text": <native instruction>}]`:

```
Act as an expert radiologist. Based only on the provided chest radiograph, write only
the Findings section as one concise clinical paragraph. Describe visible observations
precisely, use cautious language when appropriate, and do not invent facts, add an
Impression section, use bullet points, or discuss the task itself.
```

No structured labels, no soft tokens — genuinely image-only (correct for a native
baseline).

## 3. Data placed into the user message
- Q-Former mode: 32 soft-token placeholders + Stage-1 P/N/U class lists + fixed
  instruction.
- Native mode: the real image + fixed instruction.
- Neither mode places the ground-truth CheXpert label nor the target FINDINGS text into
  the user message.

## 4. Data only in the assistant target
- The reference FINDINGS (`record["ref"]`) is the assistant turn only
  (`_chat_texts:877-879`, `_native_messages:910-916`). No target leakage into the prompt.

## 5. Soft-token insertion
`image_block` emits exactly `NUM_IMG_TOKENS` copies of the single special token
`<qformer_soft_token>`. At forward time `SoftTokenEmbeddingWrapper` (`medgemma/soft_tokens.py`)
substitutes the embedding at each placeholder with a projected Q-Former vector and
`validate_soft_token_batch` fails closed if the count ≠ 32 or the batch/embedding sizes
disagree.

## 6. Number of Q-Former tokens
32 (`NUM_IMG_TOKENS`), matching the 32 Q-Former query tokens.

## 7. `apply_chat_template` usage
- Native: `apply_chat_template(tokenize=True, return_dict=True, return_tensors="pt",
  truncation, max_length, add_generation_prompt=<True for prompt / False for full>)`.
- Q-Former: `apply_chat_template(tokenize=False, add_generation_prompt=<True/False>)`
  → strings, then `tokenizer(...)`.
- Inference (`generate`) reuses the same functions with `add_generation_prompt=True`.

## 8. Loss masking over the user prompt
`encode_train_example:970` → `masked_label_ids(full_ids, prompt_ids)`: sets the
prompt-prefix tokens (which include the 32 soft-token positions) to `-100`, raises if
`full` does not start with `prompt` (BOS/template bug guard), and raises if the target is
completely truncated. Soft-token labels are therefore `-100`; the target is trained.

## 9. Positive / negative / uncertain handling
`classify_with_thresholds:339` assigns each of the 13 findings to exactly one class
(threshold-with-argmax fallback; `No Finding` skipped). `format_findings:354` renders all
three lists. Uncertain findings are listed under "Uncertain findings:" with no cautious
qualifier and framed as fact. Uncertain is **not** auto-promoted to positive (good).

## 10. Normal-study handling
None. A normal study yields ~13 negatives → "Negative findings: Enlarged
Cardiomediastinum, Cardiomegaly, Lung Opacity, …" (a 13-item dump). No compact/critical
summary policy exists.

## 11. View metadata
Absent from the prompt. Native records (`dataio/manifest.py:build_records`) carry only
`index, sample_key, ref, image_path`. The split CSVs *do* carry `ViewPosition`
(`select_anchor_rows` uses it) but it is dropped before the record is built.

## 12. Prior / temporal context
Absent. No `prior_available` flag, no prior image/report, and nothing forbids temporal
phrasing in the output.

## 13. Train vs inference prompt parity
Holds today: `encode_train_example` and `generate` both route through
`_chat_texts` / `_native_chat_inputs`, so the user turn is identical and only the
assistant turn (present in train, absent in inference) differs. Weakness: the logic is
duplicated across ≥5 functions, so parity is a convention, not an enforced invariant.

## 14. Target-leakage risk — **LOW (today)**
Target FINDINGS and ground-truth labels are assistant-only. The structured cues are
Stage-1 *predictions* of the current sample (auxiliary), not its gold label. No leakage.

## 15. Temporal-hallucination risk — **HIGH**
MIMIC FINDINGS frequently contain "unchanged", "stable", "compared to prior", "new since
…". With no prior in the input and no guard, the model is trained to emit interval-change
language it cannot ground → confident temporal hallucination at inference. Quantified in
`stage2_temporal_target_audit.md` (counts require a data run).

## 16. Stage-1 error-propagation risk — **HIGH**
"Abnormality information: …" presents Stage-1 output as established fact, and the default
style commands "Only describe abnormalities explicitly provided in the 'Abnormality
information'." A Stage-1 false negative is thereby instructed *out* of the report even
when the image shows the finding; a false positive is instructed *in*. The prompt gives
no primacy to the visual evidence.

## 17. Prompt-length / truncation risk — **MED**
`max_length=768`. The Q-Former prompt is short, but the uncapped negative dump inflates
it. `encode_train_example` raises only when the target is **fully** truncated; a partially
truncated target is trained silently. No length statistics are collected. Quantified by
`scripts/prompt_length_statistics.py` (needs a tokenizer+data run).

## 18. Negative-list-dominates-report risk — **HIGH**
The uncapped negative list is the largest structured block for the common (mostly-normal)
study, biasing the model toward long "no X, no Y, no Z…" enumerations rather than a
concise clinical paragraph.

## Extra finding — `uses_mhcac_prompt` is dead
`pipeline_modes.py` distinguishes `meta_cxr_qformer` (meant to be visual-only) from
`meta_cxr_qformer_with_mhcac_prompt`, but `uses_mhcac_prompt` is never read: both build
the identical guided prompt via `build_prompt`, so the "visual-only" Q-Former variant
actually **leaks structured labels**. The new builder makes the five visual modes
explicit (`native_anchor_only`, `native_anchor_guided`, `native_multiview`,
`qformer_visual_only`, `qformer_guided`) so this can no longer happen silently.

## Severity summary
| # | Issue | Severity |
|---|---|---|
| 16 | Stage-1 predictions framed as ground truth; visual evidence not primary | HIGH |
| 18 | Uncapped negative list dominates the report | HIGH |
| 15 | No prior/temporal guard; temporal hallucination trained in | HIGH |
| — | `uses_mhcac_prompt` dead → visual-only mode leaks labels | MED |
| 11 | No view metadata in prompt | MED |
| 6 | No prompt version/hash/config; not reproducible | MED |
| 17 | Silent partial target truncation; no length stats | MED |
| 9 | Uncertain findings framed as fact | LOW |
| 7 | `<qformer_soft_token>` not suppressed at generation | LOW |
