> Source: `training/train_eval_figure9_llm_variants_200.py:596-759`
> Status: ✅ ACTIVE

# `VariantLLM.__init__(...)`

## Located in

[`training/train_eval_figure9_llm_variants_200.py`](../../train_eval_figure9_llm_variants_200.py.doc.md)

## Purpose
Load Vicuna hoặc MedGemma, tùy chọn quantize MedGemma 4-bit, gắn/đọc LoRA và dựng
projector cho đường Q-Former.

## Execution flow
```text
hf_kwargs() / hf_token()                 ← MedGemma là gated model, cần HF_TOKEN
   ↓
IF quantize_4bit: BitsAndBytesConfig(NF4, double quant, preferred_dtype)
   ↓
load model + processor  (MEDGEMMA_MODEL_ID)
   ↓
validate_multimodal_capability(...)      ← trừ text_only_language_prior_ablation
   ↓
LoraConfig(r=lora_rank, alpha=lora_alpha, target_modules=_language_lora_targets())
get_peft_model(...)
   ↓
_align_output_head_dtype()
   ↓
IF mode Q-Former:
   img_proj = Linear(768 → hidden_gemma)
   đăng ký <qformer_soft_token>
```

`SoftTokenEmbeddingWrapper` **không** được cài cố định trong constructor. Nó chỉ
bọc embedding trong thời gian một `_forward_batch()` hoặc `generate()`, rồi
embedding gốc được phục hồi bằng `finally`.

## ★ LoRA target khác nhau giữa hai LLM
| | target_modules |
|---|---|
| MedGemma (Stage 2) | Danh sách **full module name** thuộc language tower, lọc bởi `language_lora_target_names` |
| Vicuna | Các suffix `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

## ★ `_language_lora_targets` — chỉ LoRA phần ngôn ngữ
Không gắn LoRA vào vision tower. Nếu không tìm được language target, code raise
thay vì fallback sang `all-linear`. `train_fine()` gọi
`assert_vision_tower_frozen()` trước vòng epoch.

## Config dependencies
`MEDGEMMA_MODEL_ID` · `HF_TOKEN` · `--pipeline-mode` · `--prompt-config`

Constructor mặc định `lora_rank=8`, `lora_alpha=16`; entrypoint chính
`run_medgemma_qlora.py` truyền CLI mặc định `16`/`32`.

## Side effects
Tải checkpoint lần đầu · Có thể cấp phát GPU/CPU tùy máy và `quantize_4bit`

## Error handling
Thiếu HF token cho gated model → raise từ `transformers` ·
`requires_multimodal` không thỏa → `MultimodalModelLoadError`

## Tests
`tests/test_multimodal_capability.py`

## Modification risk
Đổi `num_img_tokens` phải khớp `num_query_token` của Stage 1 (32).
