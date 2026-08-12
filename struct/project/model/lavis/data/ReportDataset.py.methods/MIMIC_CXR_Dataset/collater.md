> Source: `model/lavis/data/ReportDataset.py:751-797`
> Status: ✅ ACTIVE

# `MIMIC_CXR_Dataset.collater(samples)`

## Located in

[`ReportDataset.py`](../../ReportDataset.py.doc.md)

## Purpose
Pad số auxiliary view **ragged** về `N_max` của batch.

## ★ Nguyên tắc — docstring `:752`
> *"Pre-existing keys are delegated to the default collate untouched, so a
> `multi_view=False` batch is byte-identical to the original."*

Tức là bật multi-view **không** đổi hành vi của batch single-view.

## Signature
```python
def collater(self, samples) -> dict
```

## Execution flow
```text
not multi_view → super().collater(samples)      ← đường cũ, nguyên vẹn
   ↓
aux_keys = ["aux_image"] + [k cho k.startswith("aux_") and k.endswith("_feat")]
skip = aux_keys ∪ {"aux_view_ids"}
batch = super().collater([{k:v for k,v in s.items() if k not in skip} for s in samples])
   ↓
n_max = max(len(s["aux_view_ids"]) for s in samples)
aux_mask     = zeros(B, n_max) bool
aux_view_ids = full(B, n_max, UNKNOWN_VIEW_ID)
FOR i, s: nếu n>0 → aux_mask[i,:n]=True; aux_view_ids[i,:n]=...
   ↓
FOR key in aux_keys:
   anchor_key = "image" nếu key=="aux_image" ngược lại key[4:]
   n_max == 0 → batch[key] = zeros((B,0)+template.shape)      ← KHÔNG phải None
   ngược lại   → pad bằng zeros_like(s[anchor_key]) rồi stack
```

## Local variables
| Biến | Ý nghĩa |
|---|---|
| `n_max` | Số aux lớn nhất trong batch |
| `template` | `samples[0][anchor_key]` — lấy shape/dtype để tạo zeros |
| `UNKNOWN_VIEW_ID` | Giá trị điền cho vị trí pad |

## ★ `n_max == 0` trả tensor rỗng, không phải `None`
Nhờ vậy code xuôi dòng (`_encode_image_streams:488`) chỉ cần kiểm `.shape[1] > 0`,
không cần nhánh `None`.

## Side effects
Không.

## Error handling
Không có validate tường minh — dựa vào `super().collater` cho khóa thường.

## Tests
`tests/test_mimic_data_pipeline.py` · `tests/test_view_fusion.py` (dùng shape này)

## Modification risk
`aux_mask` là thứ `ViewFusionModule` dùng để gate. Sai `aux_mask` → study không có
aux vẫn được fuse với tensor 0, làm nhiễu biểu diễn **một cách im lặng**.
