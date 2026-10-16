> Source: `model/lavis/data/external_eval.py`
> Status: ⚠ CONDITIONAL — chỉ khi config có `datasets.mimic_cxr.external_eval` (Bảng 4)
> Last verified against source: 2026-10-16

# `model/lavis/data/external_eval.py`

← [ReportDataset.py](ReportDataset.py.doc.md) · [chexpert_labels.py](chexpert_labels.py.doc.md)

## Purpose

Bảng 4 của bài báo: chấm Stage 1 (chỉ train trên MIMIC-CXR) trên **CheXpert
validation** (200 study / 234 ảnh, nhãn 0/1 do bác sĩ). Thay vì một lớp dataset
thứ hai, module đổi `valid.csv` của CheXpert thành đúng hai bảng mà
`MIMIC_CXR_Dataset` đã đọc, nên tiền xử lý ảnh, chọn view và collate trùng hệt
đường MIMIC. Chỉ import pandas — suite CPU ghim được.

⚠ Thư mục `model/lavis/data/` bị `.gitignore` — thêm bằng `git add -f`.
⚠ **Chưa chạy trên GPU và chưa thử trên `valid.csv` thật** (2026-10-16).

## Config

```yaml
datasets:
  mimic_cxr:
    external_eval:
      kind: chexpert_val
      valid_csv: <dir>/CheXpert-v1.0-small/valid.csv
      vis_root: <dir>          # thư mục CHỨA CheXpert-v1.0-small/
      splits: [test]           # split đánh giá được thay; không bao giờ train
```

## API

| Tên | Vai trò |
|---|---|
| `ExternalEval` | dataclass `kind, valid_csv, vis_root, splits` |
| `resolve_external_eval(raw, split)` | `None` khi khoá vắng hoặc `split` không nằm trong `splits`; khoá lạ/thiếu, `kind` lạ, `splits` rỗng hoặc chứa `train` → lỗi |
| `chexpert_val_tables(valid, label_cols)` | → `(reports, labels)`: một dòng mỗi ảnh (schema MIMIC, `image_path` tương đối, `findings_clean=""`, `target_valid=False`, `has_chexpert_label=True`, `ViewPosition` AP/PA/LATERAL/UNKNOWN) và một dòng mỗi study (14 cột 0/1) |
| `load_chexpert_val(external, label_cols)` | đọc CSV, gọi hàm trên, yêu cầu mọi ảnh tồn tại dưới `vis_root` |

## Invariants (ghim bởi `tests/test_external_eval_chexpert.py`, 15 test)

- Đường dẫn phải khớp `.../valid/patientN/studyM/viewK_frontal|lateral.jpg` —
  chặn nhầm `train.csv` (nhãn labeler, có -1).
- Nhãn chỉ 0/1; -1 hay ô trống → `ValueError` (Bảng 4 cần nhãn bác sĩ).
- Các view của một study phải cùng nhãn; bảng nhãn duy nhất theo
  `(subject_id, study_id)`.
- Frontal làm anchor, lateral làm aux qua `build_study_index` hiện có.
- Nhãn đi qua `prepare_chexpert_labels` / `attach_chexpert_labels` thành 0/1,
  `classification_valid` đúng mọi dòng.

## Callers

`MIMIC_CXR_Dataset.__init__` · `tests/test_external_eval_chexpert.py`.
Kết quả chấm bằng `scripts/evaluate_chexpert_crossdomain.py`. Lệnh:
`docs/handoff/PLAN-2026-10-16-chexpert-table4.md`.
