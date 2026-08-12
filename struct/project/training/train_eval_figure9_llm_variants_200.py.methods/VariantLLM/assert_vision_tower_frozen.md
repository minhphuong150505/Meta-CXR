> Source: `training/train_eval_figure9_llm_variants_200.py:825-838`
> Status: ✅ ACTIVE

# `VariantLLM.assert_vision_tower_frozen()`

## Located in

[`training/train_eval_figure9_llm_variants_200.py`](../../train_eval_figure9_llm_variants_200.py.doc.md)

## Purpose
Xác nhận vision tower của MedGemma **đóng băng** — raise nếu không.

## ★ Vì sao là assert chứ không phải setter
Nếu chỉ đặt `requires_grad = False`, một thay đổi ở `LoraConfig` (ví dụ
`target_modules="all-linear"` bắt luôn module vision) có thể **âm thầm** làm vision
tower học lại.

Code hiện tại **từ chối** fallback sang `target_modules="all-linear"`, vì lựa chọn
đó sẽ bọc cả vision tower. `_language_lora_targets` (`:760`) trả full module name
chỉ thuộc phần ngôn ngữ; hàm này là lớp kiểm chứng thứ hai trước training.

## Signature
```python
def assert_vision_tower_frozen(self) -> None
```

## Execution flow
```text
duyệt tham số vision tower
   ↓
bất kỳ param nào requires_grad=True → raise
```

## Side effects
Không.

## Error handling
Raise nếu phát hiện tham số vision train được.

## Called by
`VariantLLM.train_fine` — sau khi dựng/khôi phục optimizer, trước vòng epoch.

## Tests
`tests/test_multimodal_capability.py` (liên quan)

## Modification risk
⚠ **Đừng bỏ hoặc bắt exception này.** Vision tower học lại sẽ (a) làm hỏng so sánh
với baseline, (b) tăng bộ nhớ, (c) không có tín hiệu nào khác báo.
