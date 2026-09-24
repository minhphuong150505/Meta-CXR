> Source: `model/lavis/data/chexpert_labels.py`
> Status: ✅ ACTIVE — gọi bởi `MIMIC_CXR_Dataset.__init__`
> Last verified against source: 2026-09-24

# `model/lavis/data/chexpert_labels.py`

← [ReportDataset.py](ReportDataset.py.doc.md) · [DECISIONS D-018](../../../_meta/DECISIONS.md#d-018--ô-chexpert-trống-là-âm-tính-đảo-ngược-quyết-định-2026-08-14)

## Purpose

Ánh xạ export CheXpert thành chỉ số lớp cho Stage 1 và dựng các cờ hợp lệ theo
dòng. Chỉ import numpy + pandas, để suite CPU ghim được đúng logic mà
`ReportDataset` chạy mà không cần torchvision/transformers/LAVIS.

⚠ Thư mục `model/lavis/data/` bị `.gitignore` — file này phải được thêm bằng
`git add -f`.

## API

| Tên | Vai trò |
|---|---|
| `IGNORE_LABEL = -100` | sentinel "không có nhãn", vừa int8 |
| `BLANK_LABEL_POLICIES = ("negative", "ignore")` | giá trị hợp lệ của `model.mhcac.blank_label_policy` |
| `DEFAULT_BLANK_LABEL_POLICY = "ignore"` | khi config **không có** khoá — giữ config cũ tái lập được |
| `resolve_blank_label_policy(value)` | `None` → mặc định; giá trị lạ → `ValueError` |
| `map_chexpert_labels(frame, cols, policy)` | `1→1, 0→0, -1→2`; ô trống → `0` (`negative`) hoặc `-100` (`ignore`); dòng trống cả `cols` → `-100` dưới cả hai |
| `prepare_chexpert_labels(chexpert, cols, *, blank_policy, excluded_labels)` | thêm `_has_chexpert_label_raw`, `_mention_<label>` (**trước** fill), ánh xạ nhãn, áp `excluded_labels` (**sau** fill), `_has_usable_label` |
| `attach_chexpert_labels(annotation, prepared, cols, *, label_key, processed_has_label)` | left-join `many_to_one`; `classification_valid`, `mention_valid`; study không khớp → `-100` mọi nhãn, mention 0 |

## Invariants (ghim bởi `tests/test_blank_label_masking.py`)

- Study không có thông tin CheXpert (không bản ghi, hoặc bản ghi trống toàn bộ)
  không bao giờ thành 14 số 0 — dưới cả hai policy.
- Mention target giống hệt nhau dưới hai policy.
- `processed_has_label=True` mà nguồn không có nhãn nào → `ValueError` (bảo vệ join).
- `preporcessing/preprocess_mimic_cxr.py::clean_chexpert` cho kết quả trùng khớp.

## Callers

`MIMIC_CXR_Dataset.__init__` · `scripts/count_chexpert_blank_policy.py` (nạp theo
đường dẫn file) · `tests/test_blank_label_masking.py`.
