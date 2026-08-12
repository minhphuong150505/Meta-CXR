> Source: `training/evaluation/subgroup_analysis.py` (215 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `subgroup_analysis.py`

## Purpose
Chấm điểm theo nhóm con: theo view, theo nhãn, theo độ dài báo cáo.

## Why it exists
Một model có thể tốt trung bình nhưng hỏng ở nhóm cụ thể (ví dụ chỉ ảnh lateral,
hoặc báo cáo rất dài). Trung bình che giấu điều đó.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `evaluate_subgroups(...)` | 160 | ★ |
| `view_subgroups(...)` | 75 | Theo `ViewPosition` |
| `label_subgroups(...)` | 113 | Theo nhãn |
| `length_subgroups(...)` | 134 | Theo độ dài |
| `subgroup_table(results, columns)` | 193 | Bảng |
| `Subgroup` / `SubgroupResult` | 35, 52 | |

## Calls / Called by
Gọi: `numpy`.
Được gọi: `scripts/evaluate_stage1.py:47`, `evaluate_stage2.py:54`.

## Side effects
Không.

## Related tests
`tests/test_evaluation_integration.py`

## Developer notes
Nhóm quá nhỏ cho khoảng tin cậy rất rộng — đọc kèm CI từ `bootstrap.py`, đừng đọc
điểm trần trụi.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md)

← [HOME](../../../HOME.md)
