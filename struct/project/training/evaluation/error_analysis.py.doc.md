> Source: `training/evaluation/error_analysis.py` (356 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `error_analysis.py`

## Purpose
Phân tích lỗi từng mẫu: hallucination thời gian, lặp lại, sai bên trái/phải, sai
mức độ, thiết bị.

## Why it exists
Một điểm BLEU không cho biết model sai **kiểu gì**. File này phân loại lỗi thành
các nhóm hành động được.

## Status
```text
✅ ACTIVE — nơi DUY NHẤT ngoài test dùng `safety/`
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `analyse_sample(...)` | 230 | ★ Một mẫu |
| `summarise_errors(reports)` | 313 | Tổng hợp |
| `SampleErrorReport` | 189 | |
| `detect_temporal_hallucination(...)` | 142 | ★ "so với phim trước" khi không có prior |
| `repetition_ratio(text)` | 163 | Model lặp câu |
| `_laterality(text)` | 173 | trái/phải |
| `_severity(text)` | 178 | mức độ |
| `_devices(text)` | 183 | ống, dây, máy |
| `_mentions(text)` | 115 | |
| `_length_stats(values)` | 346 | |

## Calls / Called by
Gọi: `evaluation.generation_metrics.normalize/tokenize` (`:36`),
**`safety.claims.*` (`:30`)**.
Được gọi: `scripts/evaluate_stage2.py:40`; `tests/test_generation_metrics.py:14`.

## Side effects
Không.

## Related tests
`tests/test_generation_metrics.py`

## Developer notes
1. ⚠ **Temporal hallucination là lỗi nguy hiểm nhất** ở đây: model viết "không đổi
   so với phim trước" trong khi không có phim trước. Nó nghe rất thuyết phục.
   `scripts/audit_temporal_targets.py` đo mức độ vấn đề ở phía dữ liệu.
2. Đây là caller production duy nhất của `safety/`. Phần còn lại của `safety/` chưa nối.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md)

← [HOME](../../../HOME.md)
