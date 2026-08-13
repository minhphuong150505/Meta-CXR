> Source: `model/lavis/data/ReportDataset.py:843-951`
> Status: ✅ ACTIVE

# `MIMIC_CXR_Dataset.__getitem__(index)`

## Located in

[`ReportDataset.py`](../../ReportDataset.py.doc.md)

## Purpose
Một **study** (không phải một ảnh) → một sample dict.

## Signature
```python
def __getitem__(self, index) -> dict
```
`index` chạy trên `self.studies`, nên `__len__` = **số study**.

## Returns
```python
{"text_output": str, "image_id": int,
 "classification_labels": [14] long, "classification_mask": [] bool,
 "generation_mask": [] bool, "dicom_id": str, "image_path": str,
 "image": [3,448,448]  HOẶC  "<enc>_feat",
 # chỉ khi model.explanation.mask_cache_dir được cấu hình:
 "explanation_mask": [112,112] float32 (0/1),
 "explanation_mask_valid": [] bool,
 "explanation_mask_source": int,  # 0=lung anatomical prior, 1=MS-CXR bbox
 # chỉ khi multi_view:
 "anchor_view_id": int, "aux_view_ids": list[int], "aux_image": list[[3,448,448]]}
```
⚠ `aux_view_ids` và `aux_image` là **ragged** — `collater` pad chúng.

## Execution flow
```text
study = self.studies[index]
ann   = self.annotation.iloc[study["anchor"]]
   ↓
IF mask cache:
   _read_explanation_mask(anchor ID) → uint8 mask, valid, source
   _row_visual(ann, mask)             → image + cùng affine mask
ELSE:
   _row_visual(ann)                   → đường cũ, không có key explanation
caption = ann["findings"].strip()
chexpert_labels = ann[chexpert_cols].astype(float)
   ↓
sample = {... classification_mask=ann["classification_valid"],
              generation_mask=ann["target_valid"] ...}
sample.update(anchor_visual)
IF mask cache: thêm float mask [112,112], bool valid, int source
   ↓
IF multi_view:
   aux_visuals = [_row_visual(annotation.iloc[p]) for p in study["aux"]]
   sample["anchor_view_id"], sample["aux_view_ids"]
   feature_cache None → sample["aux_image"] = [...]
   ngược lại        → sample["aux_<enc>_feat"] = [...]
```

## Local variables
`study` — dict từ `_init_study_index`: `anchor`, `aux`, `anchor_view_id`, `aux_view_ids`
`ann` — hàng CSV của **anchor** (report và nhãn lấy từ đây, không từ aux)

## ★ Ba loại mask, ba vai trò

`classification_mask ← classification_valid` · `generation_mask ← target_valid`
đến thẳng từ CSV. `explanation_mask_valid` đến từ việc anchor có entry trong
cache. Cả ba đi vào `Blip2Qformer.forward` và gate loss tương ứng.

Study không có mask cache entry nhận zero `[112,112]`, `valid=False`, source mặc
định 0; vì valid false nên source đó không được diễn giải là lung supervision.

## Side effects
Đọc ảnh từ đĩa (hoặc feature cache) mỗi lần gọi; lần đầu mỗi worker có thể mở
explanation memmap read-only.

## Error handling
Ủy quyền cho [`_row_visual`](_row_visual.md) — path tuyệt đối/traversal → `ValueError`;
DICOM thiếu trong cache → `KeyError`.

Mask cache/index hỏng → `ValueError` không echo identifier.

## Important conditions
```python
if self.explanation_mask_cache_dir is not None
if self.multi_view
if self.feature_cache is None
```

## Tests
`tests/test_mimic_data_pipeline.py` · `tests/test_explanation_mask_pipeline.py`

## Modification risk
Thêm khóa mới phải cân nhắc `collater` xử lý ra sao — khóa ragged cần pad thủ công.

⚠ Code Vicuna cũ (`Conversation`, prompt) còn ở dạng comment `:672-711`. Không đang chạy.
