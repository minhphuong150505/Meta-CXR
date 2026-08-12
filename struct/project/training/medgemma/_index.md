> Source: `training/medgemma/`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `training/medgemma/`

## Purpose

Hai thứ đặc thù MedGemma: tiêm soft token (mode Q-Former) và kiểm tra model có
thật sự đa phương thức hay không.

## Parent

[`training/`](../_index.md)

## Children

| File | LOC | Doc | Status |
|---|---|---|---|
| `soft_tokens.py` | — | [📄](soft_tokens.py.doc.md) | 🟡 `SoftTokenEmbeddingWrapper`, `soft_token_bad_words_ids` |
| `capabilities.py` | 221 | [📄](capabilities.py.doc.md) | ✅ Kiểm tra khả năng đa phương thức |

## ⚠ `soft_tokens.py` — chỗ dễ sai nhất repository

Vector Q-Former đã chiếu **THAY THẾ** embedding tại vị trí `<qformer_soft_token>`
— **không cộng vào**.

Nếu index theo hàng sai:

```text
loss vẫn giảm bình thường  ✓
mỗi study được mô tả bằng ảnh của study khác  ✗
KHÔNG có tín hiệu nào báo lỗi
```

Vì thế module này validate shape **theo từng hàng** và **fail-closed** thay vì
đoán. Đừng nới lỏng các kiểm tra đó để "cho nó chạy".

Soft token cũng được đưa vào `bad_words_ids` khi generate, để model không tự sinh
ra ký hiệu đó.

## `capabilities.py` — vì sao cần

Một model text-only vẫn nhận prompt và vẫn sinh ra báo cáo trông hợp lý — **nhưng
chưa từng nhìn thấy một pixel nào**. Đó là thất bại đắt nhất project có thể mắc.

Module này kiểm tra khả năng đa phương thức và **fail-closed**: mọi mode trừ
`text_only_language_prior_ablation` đều có `requires_multimodal=True` và sẽ dừng
nếu model hóa ra là text-only.

## Main responsibilities

1. Tiêm soft token đúng vị trí, đúng hàng.
2. Chặn model tự sinh soft token.
3. Xác nhận model thật sự đa phương thức trước khi train/generate.

## Entry points

Không có. Thư viện.

## Dependencies

`torch`, `transformers` · `training/stage2_utils.validate_soft_token_batch`
(dual-import shim `:20-22`)

## Used by

`training/train_eval_figure9_llm_variants_200.py:96-108` (dual-import) ·
`tests/test_soft_token_injection.py` · `tests/test_multimodal_capability.py` ·
`tests/test_stage2_prompts.py:42`

## Execution flow

```text
Q-Former output [B,32,768]
   ↓ img_proj
[B,32,hidden_gemma]
   ↓
SoftTokenEmbeddingWrapper.forward(input_ids)
   ├─ tìm vị trí <qformer_soft_token> trong từng hàng
   ├─ validate_soft_token_batch()   ← fail-closed
   └─ THAY THẾ embedding tại các vị trí đó
   ↓
Gemma decoder
```

## Important configurations

Chỉ hoạt động khi `--pipeline-mode meta_cxr_qformer` hoặc
`meta_cxr_qformer_with_mhcac_prompt`. `num_img_tokens` mặc định `32`, khớp
`num_query_token` của Stage 1.

## Status

```text
✅ ACTIVE — capabilities.py
🟡 CONDITIONAL — soft_tokens.py (chỉ mode Q-Former)
```

## Notes

- `SoftTokenEmbeddingWrapper` từng nằm trong `fig9`; đã tách ra
  (`docs/migration_guide.md:11`). Tên cũ vẫn được re-export nên code cũ chạy được.
  `num_img_tokens` là tham số vị trí **thứ 4**, mặc định 32.

- **Số soft token phải khớp `num_query_token` của Stage 1.** Đổi một bên mà không
  đổi bên kia → shape mismatch, hoặc tệ hơn, im lặng sai.

## Related documentation

[ARCHITECTURE.md §3.1](../../_meta/ARCHITECTURE.md#31-soft-token--chỗ-dễ-sai-nhất-repo) ·
[DATA_FLOW.md §4.2](../../_meta/DATA_FLOW.md#42-đường-q-former-meta_cxr_qformer) ·
[GLOSSARY: Soft token](../../_meta/GLOSSARY.md#soft-token)

← [`training/`](../_index.md) · [HOME](../../../HOME.md)
