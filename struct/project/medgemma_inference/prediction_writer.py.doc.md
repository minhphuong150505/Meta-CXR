> Source: `medgemma_inference/prediction_writer.py` (109 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `medgemma_inference/prediction_writer.py`

## Purpose
Ghi JSONL **append-only, crash-safe**, và chặn dữ liệu nhạy cảm lọt ra.

## ★ Crash-safe: flush + fsync từng dòng
Docstring `:3` ghi rõ: mỗi record được flush và fsync ngay khi ghi. Process bị kill
để lại file mà **mọi dòng hoàn chỉnh đều hợp lệ**; dòng dở dang bị cắt khi resume
thay vì để nó phá reader tiếp theo.

Với một run inference dài trên GPU thuê, mất toàn bộ kết quả vì crash ở phút 200
là rất đắt.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| [`PredictionWriter`](prediction_writer.py.methods/PredictionWriter/_index.md) | 71 | ★ |
| `assert_publishable(record)` | 29 | ★ **Chặn PHI** trước khi ghi |
| `read_completed_keys(path)` | 40 | ★ Đọc key đã xong, cắt dòng hỏng |
| `PrivacyViolation` | 25 | |

## `assert_publishable` — chốt chặn
Record đầu ra **không được mang** `subject_id`, `study_id`, `dicom_id`, đường dẫn
hay reference text. Hàm này raise `PrivacyViolation` nếu thấy.

## Calls / Called by
Gọi: `json`, `os` (`fsync`).
Được gọi: `runner.py`; `tests/test_pretrained_findings.py`.

## Side effects
Ghi + fsync file.

## Error / edge cases
`PrivacyViolation` khi record mang định danh · Dòng JSON hỏng ở cuối file → bị cắt

## Related tests
`tests/test_pretrained_findings.py`

## Developer notes
`.jsonl` bị `.gitignore` chặn. Đừng commit, đừng đưa nội dung vào `struct/`.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
