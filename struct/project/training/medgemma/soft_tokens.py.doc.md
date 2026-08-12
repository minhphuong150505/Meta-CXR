> Source: `training/medgemma/soft_tokens.py` (102 dòng)
> Status: 🟡 CONDITIONAL — ⚠ chỗ dễ sai nhất repo
> Last verified against source: 2026-08-12

# `training/medgemma/soft_tokens.py`

## Purpose

Tiêm output Q-Former vào MedGemma dưới dạng **soft token**, và chặn model tự sinh
ra ký hiệu soft token.

## ⚠ Đọc trước: lỗi ở đây hoàn toàn im lặng

Vector Q-Former đã chiếu **THAY THẾ** embedding tại vị trí `<qformer_soft_token>`
— **không cộng vào**.

Nếu index theo hàng sai:

```text
loss vẫn giảm bình thường               ✓
mỗi study mô tả bằng ảnh study khác     ✗
không tín hiệu nào báo                  ✗
```

Vì thế module này validate shape **theo từng hàng** và **fail-closed**.
**Đừng nới lỏng kiểm tra đó để "cho nó chạy".**

## Status

```text
🟡 CONDITIONAL — chỉ mode meta_cxr_qformer*
```

## Main items

| Tên | Dòng | Vai trò |
|---|---|---|
| `soft_token_bad_words_ids(img_token_id)` | 27 | → `bad_words_ids` khi generate; `None` nếu không có token |
| `SoftTokenEmbeddingWrapper(nn.Module)` | 44 | ★ Bọc embedding layer |
| `.__init__(...)` | 51 | ⚠ `num_img_tokens` là tham số **vị trí thứ 4**, mặc định 32 |
| `.weight` | 65 | Property ủy quyền |
| `.forward(input_ids)` | 68 | ★ Thay thế tại vị trí soft token |

## Execution flow

```text
Q-Former [B,32,768] ─► img_proj ─► [B,32,hidden_gemma]
                                        │
input_ids ─► embedding gốc ─► [B,L,hidden_gemma]
                                        │
              tìm vị trí <qformer_soft_token> theo TỪNG HÀNG
                                        │
              validate_soft_token_batch()   ← fail-closed
                                        │
              THAY THẾ (không cộng) tại các vị trí đó
                                        ▼
                                Gemma decoder
```

## Calls / Called by

Gọi: `torch.nn`, `training.stage2_utils.validate_soft_token_batch` (`:20`/`:22` dual-import).
Được gọi: `fig9:100,108`; `tests/test_soft_token_injection.py`, `test_stage2_prompts.py:42`.

## Side effects

Không mutate ngoài; bọc module embedding của model.

## Error / edge cases

Số vị trí soft token ≠ `num_img_tokens` → raise · Batch size không khớp → raise ·
`img_token_id is None` → `soft_token_bad_words_ids` trả `None`

## Related tests

`tests/test_soft_token_injection.py` — **THAY THẾ chứ không cộng**, index theo hàng

## Developer notes

1. **Số soft token phải khớp `num_query_token` Stage 1** (mặc định 32). Đổi một
   bên mà không đổi bên kia → shape mismatch, hoặc tệ hơn, im lặng sai.
2. Từng nằm trong `fig9`; đã tách (`docs/migration_guide.md:11`). Tên cũ còn re-export.
3. `soft_token_bad_words_ids` phải được dùng ở **mọi** lời gọi generate của mode
   Q-Former, nếu không model có thể tự sinh ký hiệu đó vào báo cáo.

← [`medgemma/`](_index.md) · [HOME](../../../HOME.md)
