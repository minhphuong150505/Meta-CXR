> Source: `training/stage2_utils.py` (298 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `training/stage2_utils.py`

## Purpose

Helper dùng chung của Stage 2 — và một số chốt chặn an toàn quan trọng.

## Status

```text
✅ ACTIVE
```

## Main functions

| Hàm | Dòng | Vai trò |
|---|---|---|
| `stable_fingerprint(payload, length=16)` | 21 | ★ Vân tay ổn định; dùng cả ở P8 |
| `file_identity(path)` | 29 | Danh tính file (size/mtime/hash) |
| `select_threshold_class(...)` | 42 | Chọn lớp theo threshold |
| `safe_prediction_row(...)` | 68 | ★ Lọc trường nhạy cảm khỏi hàng prediction |
| `contains_sensitive_eval_fields(row)` | 80 | ★ Phát hiện trường nhạy cảm |
| `adapter_is_complete(path, image_mode)` | 84 | Adapter đủ file chưa |
| `masked_label_ids(...)` | 114 | ★ Mask prompt prefix khỏi training label |
| `validate_soft_token_batch(...)` | 138 | ★ **Fail-closed** cho soft token |
| `accumulation_window_size(...)` | 159 | ★ Kích thước cửa sổ accumulation (đuôi) |
| `language_lora_target_names(...)` | 171 | Chọn module LoRA |
| `section_omission_rate(...)` | 199 | Tỉ lệ bỏ sót section |
| `prefix_metric_keys(...)` | 231 | Thêm tiền tố key metric |
| `native_findings_instruction()` | 236 | Chỉ dẫn cho đường native |
| `private_bucket_violations(...)` | 251 | ★ Phát hiện bucket không riêng tư |

## Bốn hàm đáng chú ý

### `masked_label_ids` — prompt không được học lại
Đặt `-100` cho mọi token thuộc prompt prefix. Không có nó, model học sinh lại
prompt thay vì học sinh báo cáo.

### `validate_soft_token_batch` — chống lỗi im lặng
Kiểm shape theo **từng hàng**. Nếu index sai, loss vẫn giảm nhưng mỗi study được
mô tả bằng ảnh của study khác — không tín hiệu nào báo. Hàm này fail-closed.

### `accumulation_window_size` — đuôi accumulation
Batch cuối của epoch thường không đủ `grad_accum` bước. Chia loss cho đúng số
microbatch thực có, không cho hằng số. Có test riêng
(`tests/test_training_core.py`).

### `contains_sensitive_eval_fields` / `safe_prediction_row` — chốt dữ liệu
`SENSITIVE_EVAL_KEYS` được `training/evaluation/counterfactual.py:36` mirror lại.
Hai nơi phải khớp nhau.

## Calls / Called by

Gọi: stdlib + `hashlib`, `pathlib`.
Được gọi: `fig9:63,79`; `run_medgemma_qlora.py` (gián tiếp); `dataio/manifest.py:21`;
`medgemma/soft_tokens.py:20`; `medgemma_inference/run_pretrained_findings.py:39`;
`tests/test_stage2_utils.py`, `test_stage2_prompts.py:41`, `test_section_metrics.py:21`

## Side effects

Không (`file_identity` đọc metadata file).

## Related tests

`tests/test_stage2_utils.py` · `tests/test_training_core.py` (đuôi accumulation) ·
`tests/test_section_metrics.py`

## Developer notes

1. Mang **dual-import shim** — file này được import từ cả hai đường.
2. `SENSITIVE_EVAL_KEYS` phải khớp với bản mirror ở `evaluation/counterfactual.py:36`.
3. `validate_soft_token_batch` **đừng nới lỏng** để "cho nó chạy".

← [training/](_index.md) · [HOME](../../HOME.md)
