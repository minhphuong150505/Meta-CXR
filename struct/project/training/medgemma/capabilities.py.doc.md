> Source: `training/medgemma/capabilities.py` (221 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `training/medgemma/capabilities.py`

## Purpose

Xác nhận model **thật sự đa phương thức** trước khi train hay generate, và
fail-closed nếu không.

## Why it exists

Một model text-only vẫn nhận prompt và vẫn sinh ra báo cáo trông hợp lý — **nhưng
chưa từng nhìn thấy một pixel nào**. Đó là thất bại đắt nhất project có thể mắc:
kết quả trông tốt, kết luận sai hoàn toàn.

## Status

```text
✅ ACTIVE
```

## Main items

| Tên | Dòng | Vai trò |
|---|---|---|
| `MultimodalModelLoadError` | 51 | Exception |
| `CapabilityReport` | 97 | Kết quả kiểm; `.as_metadata()` ghi vào artifact |
| `MultimodalCapabilityValidator` | 162 | `.inspect(model, processor)`, `.require_multimodal(...)` |
| `validate_multimodal_capability(...)` | 206 | Hàm tiện dụng |

### Bốn kiểm tra độc lập

| Hàm | Dòng | Kiểm gì |
|---|---|---|
| `_has_image_processor(processor)` | 113 | Processor có xử lý ảnh |
| `_has_vision_config(model)` | 131 | Config có phần vision |
| `_has_multimodal_modules(model)` | 138 | Có module đa phương thức |
| `_forward_accepts_pixels(model)` | 142 | `forward` nhận `pixel_values` |
| `_is_pure_causal_lm(model)` | 157 | ⚠ Cờ đỏ: chỉ là causal LM |

Bốn góc nhìn khác nhau — một model có thể lừa được một kiểm tra, khó lừa cả bốn.

## Liên hệ với `pipeline_modes`

`PipelineMode.requires_multimodal` quyết định có gọi `require_multimodal` không.
**Chỉ** `text_only_language_prior_ablation` đặt `False`.

## Calls / Called by

Gọi: `inspect`, `typing`; không import transformers ở module scope.
Được gọi: `fig9:96,104`; `tests/test_multimodal_capability.py:26,326`.

## Side effects

Chỉ đọc thuộc tính model. `CapabilityReport.as_metadata()` được ghi vào metadata run.

## Error / edge cases

Model text-only + mode yêu cầu multimodal → `MultimodalModelLoadError` **raise**,
không hạ cấp.

## Related tests

`tests/test_multimodal_capability.py` (330 dòng)

## Developer notes

1. **Đừng bắt exception này rồi tiếp tục.** Fail-closed là toàn bộ giá trị của module.
2. `as_metadata()` nên luôn vào artifact — sau này nó là bằng chứng model đã nhìn ảnh.

← [`medgemma/`](_index.md) · [HOME](../../../HOME.md)
