# Stage-2 Prompt Design (v2)

The v2 prompt builder (`stage2/prompts/`) is one shared, versioned, torch-free entry
point for both training and inference. It replaces prompt logic that was scattered
across `format_findings`, `build_instruction`, `build_prompt` and
`build_native_instruction`. See `stage2_prompt_audit.md` for what it fixes.

## Data flow

```
Stage-1 prediction (logits → P/N/U per finding)  ─┐
Q-Former embeddings (32 tokens)                   ─┤
view / prior / indication / technique metadata    ─┤
                                                   ▼
                          context_from_record → PromptContext
                                                   │
                                          PromptConfig (YAML, validated)
                                                   ▼
                                PromptBuilder.build_user_messages
                                (policy selection: normal/negative/uncertainty)
                                                   ▼
                       ordered PromptParts  (image | soft_tokens | text)
                                                   ▼
        VariantLLM: apply_chat_template → tokenize → mask prompt prefix (-100)
                                                   ▼
                                   MedGemma generation (soft token suppressed)
                                                   ▼
                                            evaluator
```

## Core principles the prompt encodes
1. **Visual evidence is primary**; Stage-1 predictions are *auxiliary cues that may be
   wrong* ("Auxiliary Stage-1 predictions, which may be imperfect:").
2. The model may **omit** an unsupported prediction and must **not** drop a visible
   finding merely because it was predicted absent.
3. **Uncertain ≠ positive.** Uncertain findings render under "Possible or uncertain"
   and are never promoted.
4. **No prior → no temporal language.** The `NO_PRIOR_GUARD` line is added whenever no
   prior/comparison is available.
5. **FINDINGS only** — no Impression, recommendations, bullets or task talk.
6. Laterality/location/severity are stated only when supported, never inferred from a
   finding name.

## Visual modes (fixes the dead `uses_mhcac_prompt`)
| Mode | soft tokens | native image | structured cues | aux image | image_mode |
|---|---|---|---|---|---|
| `native_anchor_only` | – | ✓ | – | – | native |
| `native_anchor_guided` | – | ✓ | ✓ | – | native |
| `native_multiview` | – | ✓ | ✓ | ✓ | native |
| `qformer_visual_only` | ✓ | – | – | – | qformer |
| `qformer_guided` | ✓ | – | ✓ | – | qformer |

`qformer_visual_only` genuinely carries no Stage-1 labels — the "visual only"
experiment can no longer leak them.

## Policies
- **normal_policy**: `compact_summary` (default), `critical_negatives`,
  `all_negatives`, `no_structured_normal_statement`. A structurally-normal prediction
  (no positive/uncertain) never dumps 13 negatives; the compact statement is used, and
  it is framed as a *prediction*, so the image can still override it.
- **negative_policy**: `none`, `all`, `critical_only` (default), `top_k_confident`,
  `thresholded`, `normal_summary`, capped by `max_negative_findings`.
- **uncertainty_policy**: `explicit_possible` (default), `probability_bins` (requires
  validation-calibrated probabilities — the builder raises without them),
  `omit_low_confidence`, `preserve_three_class`.
- **temporal.target_policy**: `keep` (default) / `remove_temporal_clauses` /
  `exclude_sample` / `require_prior_context` (see `stage2_temporal_target_audit.md`).

Critical negatives default to Pneumothorax, Pleural Effusion, Consolidation, Edema —
all in the classifier ontology (`ABNORMALITIES_14`); unknown names are filtered, never
crash.

## Final Q-Former guided template (rendered)
```
Visual study features:

<qformer_soft_token> × 32

Auxiliary Stage-1 predictions, which may be imperfect:
- Present: <positive or none>
- Possible or uncertain: <uncertain or none>
- Clinically relevant absent: <selected negatives>          # omitted if none

Study context:
- Views: <PA (anchor), lateral (auxiliary)>                 # only shown lines
- Prior comparison available: <yes|no>
- Indication: <...>                                         # omitted if absent
- Technique: <...>                                          # omitted if absent

Generate only the FINDINGS section for the current chest radiograph study. Use the
visual study features as the primary evidence. Treat the structured predictions only
as auxiliary cues that may be wrong. … Express possible or uncertain findings
cautiously. Do not convert an uncertain prediction into a definite finding without
visual support. If prior comparison is unavailable, do not state or imply that a
finding is new, improved, worsened, stable or unchanged. Do not mention prediction
labels, confidence categories, model outputs, the prompt, an Impression section,
recommendations or the task itself. Return one concise clinical paragraph of 1 to 4
sentences.
```

## Final native image-only template (`native_anchor_only`, rendered)
```
[image]

Generate only the FINDINGS section for the current chest radiograph study. Use only
the provided image or images and available study context. Describe supported positive
findings and clinically relevant negative findings. Include laterality, location,
severity, extent and support-device position only when visible or supported. Use
cautious wording for equivocal findings. If prior comparison is unavailable, do not
state or imply that a finding is new, improved, worsened, stable or unchanged. Do not
output an Impression section, recommendations, bullet points, patient history,
prediction labels or discussion of the task. Return one concise clinical paragraph of
1 to 4 sentences.
```

## Train/inference parity & masking
`VariantLLM.encode_train_example` (train) and `VariantLLM.generate` (inference) both
call `_render_prompt_text` → `PromptBuilder`, so the user turn is identical; only the
assistant turn differs. `masked_label_ids` masks the whole prompt prefix (including the
32 soft-token positions) to `-100` and fails closed if the full sequence does not start
with the prompt. The soft-token placeholder is a single registered special token; the
builder emits exactly 32 and `validate_soft_token_batch` fails closed otherwise. The
special token is added to `bad_words_ids` at generation.

## Reproducibility
Each run records, in the adapter `meta.json`/`manifest.json`, the prompt `version`,
`visual_mode`, the four policies, `config_hash`, `template_hash`, `num_img_tokens`,
tokenizer/processor id and model id. Rendered per-sample prompts (which contain
findings text) are written only by the debug exporter, never into uploaded artifacts.

## Backward compatibility
`prompt_config=None` (the default) preserves the exact legacy prompt strings and does
not alter existing adapters. v2 is opt-in via `--prompt-config`.

## Not yet verified
The builder and its parity/masking invariants are covered by CPU unit tests. The
`VariantLLM` wiring (chat-template rendering, generation, suppression) has **not** run
on a GPU with MedGemma weights; no model metric in this repo comes from a v2 run.
