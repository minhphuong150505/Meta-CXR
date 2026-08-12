> Source: `stage2/prompts/policies.py` (205 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `stage2/prompts/policies.py`

## Purpose
Chính sách nội dung prompt: bao nhiêu negative, diễn đạt uncertain thế nào, xử lý
ngôn ngữ so sánh thời gian ra sao.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `NormalPolicy` (Enum) | 18 | Khi không có positive/uncertain |
| `NegativePolicy` (Enum) | 25 | Bao nhiêu negative đưa vào |
| `UncertaintyPolicy` (Enum) | 34 | Cách diễn đạt uncertain |
| `TemporalTargetPolicy` (Enum) | 41 | ★ Xử lý target có so sánh thời gian |
| `select_negative_findings(...)` | 100 | Chọn + giới hạn negative |
| `render_uncertain(...)` | 138 | Ngôn ngữ thận trọng |
| `apply_temporal_target_policy(...)` | 174 | ★ |
| `contains_temporal_language(text)` | 92 | ★ Phát hiện "unchanged", "compared to" … |
| `confidence_bin(probability)` | 56 | Xác suất → nhãn tin cậy |
| `_cap(names, limit)` | 96 | |
| `_strip_temporal_sentences(text)` | 199 | |

## ⚠ Temporal: guard trong prompt ≠ dữ liệu sạch
`temporal_target_policy` mặc định là **`keep`**. Prompt có guard cấm so sánh thời
gian, nhưng **dữ liệu train vẫn chứa target có so sánh** vì split hiện chưa mang
prior linkage đầy đủ.

`scripts/audit_temporal_targets.py` đo mức độ vấn đề để chọn policy **bằng bằng
chứng thay vì đoán**.

## Calls / Called by
Gọi: `enum`, `re`. Stdlib.
Được gọi: `builder.py`; `stage2/prompts/__init__` re-export `contains_temporal_language`
→ `scripts/audit_temporal_targets.py:28`; `tests/test_stage2_prompts.py:33`.

## Side effects
Không.

## Related tests
`tests/test_stage2_prompts.py -k negative_policy`

## Developer notes
Chín config `configs/prompt_ablation/P1..P9.yaml` chính là để đo tác động của các
policy này. Đổi mặc định nên có bằng chứng từ ablation.

## Source relationships

- **Parent:** [`_index.md`](_index.md)

← [HOME](../../../HOME.md)
