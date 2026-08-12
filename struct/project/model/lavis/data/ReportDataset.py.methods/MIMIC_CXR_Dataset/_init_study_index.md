> Source: `model/lavis/data/ReportDataset.py:477-524`
> Status: ✅ ACTIVE

# `MIMIC_CXR_Dataset._init_study_index(cfg, truncate=None)`

## Located in

[`ReportDataset.py`](../../ReportDataset.py.doc.md)

## Purpose
Gộp nhiều hàng ảnh của cùng một study thành **một** mẫu, chọn anchor, và liệt kê
auxiliary view.

## ★ Vì sao study-level sampling
Coi mỗi ảnh là một mẫu độc lập sẽ:
- **lặp lại cùng một báo cáo** nhiều lần (một study, nhiều ảnh, một report)
- **đánh trọng số cao hơn** cho study nhiều view

Sampling theo study sửa cả hai.

## Signature
```python
def _init_study_index(self, cfg, truncate=None) -> None
```
Ghi `self.studies` — list dict `{anchor, aux, anchor_view_id, aux_view_ids}`.

## Config dependencies
```yaml
model:
  data:
    study_sampling: true
    anchor_priority: [PA, AP, lateral]
    max_aux_views: 1
```
⚠ **`data:` phải nằm trong `model:`** — `Config` chỉ merge `run`/`model`/`datasets`.

## Execution flow
```text
đọc model.data.{study_sampling, anchor_priority, max_aux_views}
   ↓
study_sampling=false → mỗi hàng là một study (aux rỗng)
   ↓
mimic_cxr_utils.build_study_index(annotation, anchor_priority, max_aux_views, ...)
   ├─ nhóm theo study_id
   ├─ chọn anchor theo thứ tự ưu tiên ViewPosition
   └─ lấy tối đa max_aux_views hàng còn lại làm aux
   ↓
truncate → cắt danh sách study
```

## Dependencies
`model/lavis/data/mimic_cxr_utils.py`: `build_study_index` (`:23`), `view_id` (`:13`)

## Side effects
Ghi `self.studies`. Không đọc ảnh.

## Config dependencies
`run.truncate_train` / `truncate_val` / `truncate_test` (qua tham số `truncate`)

## Tests
`tests/test_mimic_data_pipeline.py` — nạp `mimic_cxr_utils.py` **theo path** để
test được mà không import LAVIS.

## Modification risk
Đổi `anchor_priority` đổi ảnh nào là tín hiệu chính → **kết quả không so sánh được**
với checkpoint cũ. Đổi `max_aux_views` đổi `N_max` và chi phí encode aux.
