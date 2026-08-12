> Source: `model/lavis/data/ReportDataset.py:665-746`
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
 # chỉ khi multi_view:
 "anchor_view_id": int, "aux_view_ids": list[int], "aux_image": list[[3,448,448]]}
```
⚠ `aux_view_ids` và `aux_image` là **ragged** — `collater` pad chúng.

## Execution flow
```text
study = self.studies[index]
ann   = self.annotation.iloc[study["anchor"]]
   ↓
anchor_visual = _row_visual(ann)      → "image" hoặc "<enc>_feat"
caption = ann["findings"].strip()
chexpert_labels = ann[chexpert_cols].astype(float)
   ↓
sample = {... classification_mask=ann["classification_valid"],
              generation_mask=ann["target_valid"] ...}
sample.update(anchor_visual)
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

## ★ Hai mask đến thẳng từ CSV
`classification_mask ← classification_valid` · `generation_mask ← target_valid`
Chúng đi thẳng vào `Blip2Qformer.forward` và quyết định loss nào áp cho hàng nào.

## Side effects
Đọc ảnh từ đĩa (hoặc feature cache) mỗi lần gọi.

## Error handling
Ủy quyền cho [`_row_visual`](_row_visual.md) — path tuyệt đối/traversal → `ValueError`;
DICOM thiếu trong cache → `KeyError`.

## Important conditions
```python
if self.multi_view:                # :735
if self.feature_cache is None:     # :741
```

## Tests
`tests/test_mimic_data_pipeline.py`

## Modification risk
Thêm khóa mới phải cân nhắc `collater` xử lý ra sao — khóa ragged cần pad thủ công.

⚠ Code Vicuna cũ (`Conversation`, prompt) còn ở dạng comment `:672-711`. Không đang chạy.
