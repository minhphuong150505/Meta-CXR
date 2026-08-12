> Source: `training/train_eval_figure9_llm_variants_200.py:595-1372`
> Status: ✅ ACTIVE

# `class VariantLLM`

## Located in

[`train_eval_figure9…py`](../../train_eval_figure9_llm_variants_200.py.doc.md)

## Responsibility
**Toàn bộ cơ chế Stage 2 trong một class** (~660 dòng, ~25 method): load model,
QLoRA, prompt, collate, soft token, train loop, generate, checkpoint.

⚠ Docs nội bộ ghi class này đang **chờ được tách nhỏ**
(`docs/refactor_hotspots.md:11`). Phạm vi documentation:
[D-006](../../../_meta/DECISIONS.md#d-006--độ-sâu-documentation-cho-động-cơ-stage-2).

## Constructor
[`__init__`](__init__.md) (`:596`) — load MedGemma 4-bit NF4, gắn LoRA, dựng soft
token wrapper nếu mode Q-Former.

## State
| Attribute | Ghi chú |
|---|---|
| model MedGemma + LoRA | 4-bit NF4, double quant, compute bfloat16 |
| processor / tokenizer | |
| `img_proj` | Chỉ mode Q-Former; chiếu 768 → hidden Gemma |
| soft token wrapper | Chỉ mode Q-Former |
| `prompt_config` | `PromptConfig` hoặc `None` (→ prompt legacy) |

## Lifecycle
```text
__init__ → assert_vision_tower_frozen → parameter_report
   ↓
train_fine(train, val)  ← chọn checkpoint theo validation CE
   ↓
generate(test)  ← ĐÚNG MỘT LẦN
   ↓
save checkpoints/last mỗi epoch + promote best adapter + _prompt_metadata()
```

## Public methods có trang riêng
[`train_fine`](train_fine.md) ★ · [`generate`](generate.md) ★ ·
[`encode_train_example`](encode_train_example.md) · [`collate_train`](collate_train.md) ·
[`_forward_batch`](_forward_batch.md) · [`evaluate_loss`](evaluate_loss.md) ·
[`assert_vision_tower_frozen`](assert_vision_tower_frozen.md)

## Trivial helpers
| Method | Vai trò |
|---|---|
| `_language_lora_targets` (`:760`) | Chọn module cho LoRA |
| `_align_output_head_dtype` (`:778`) | Đồng bộ dtype head |
| `parameter_report` (`:789`) | Đếm tham số train được |
| `_prompt_metadata` (`:839`) | version + config hash + template hash |
| `save_adapter` (`:864`) | Ghi adapter + img_proj |
| `load_img_proj_if_present` (`:907`) | Nạp lại projection |
| `_render_prompt_text` (`:914`) | → `PromptBuilder` |
| `_chat_texts` (`:930`) | Cặp prompt/target |
| `_load_rgb` (`:964`) | Đọc ảnh |
| `_native_messages` (`:971`) / `_native_chat_inputs` (`:999`) | Đường native |

## Dependencies
`transformers`, `peft`, `bitsandbytes`, `stage2.prompts`, `training.medgemma.*`,
`training.stage2_utils`

## Callers
`run_medgemma_qlora.train_mode`

## Run identity

Các hàm build/evaluate nhận `Stage1Context` tường minh. Class không đọc global
`RUN_NAME`, `THRESHOLDS` hay `STAGE1_*_OVERRIDE`.
