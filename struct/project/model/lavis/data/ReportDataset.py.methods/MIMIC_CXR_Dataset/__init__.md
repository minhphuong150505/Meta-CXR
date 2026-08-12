> Source: `model/lavis/data/ReportDataset.py:222-455`
> Status: ✅ ACTIVE

# `MIMIC_CXR_Dataset.__init__(...)`

## Located in

[`ReportDataset.py`](../../ReportDataset.py.doc.md)

## Purpose
Nạp CSV, dựng transform, index study, mở feature cache.

## Signature
```python
def __init__(self, vis_processor, text_processor, vis_root, split, cfg,
             ann_paths=[], truncate=None)
```

## Parameters
| Tham số | Ghi chú |
|---|---|
| `vis_processor` / `text_processor` | ⚠ `pretraining/train.py` truyền **`None`** cho cả hai — transform dựng nội bộ |
| `vis_root` | `VIS_ROOT` từ `local_config`; phải chứa trực tiếp `files/` |
| `split` | `"train"` / `"val"` / `"test"` → chọn CSV tương ứng |
| `cfg` | LAVIS Config — đọc `model.data.*` và `datasets.*` |
| `truncate` | Cắt cho smoke test |

## Execution flow
```text
đọc CSV theo split (PROCESSED_{TRAIN,VAL,TEST}_CSV)
   ↓
_coerce_bool cho classification_valid / target_valid
   ↓
dựng transform: resize 512 → CenterCrop 448 → ExpandChannels
   train: thêm augmentation (affine ±5°, translate .02, scale ±.05, jitter)
   ↓
_init_study_index(cfg, truncate)      → self.studies
   ↓
_init_feature_cache(cfg)              → self.feature_cache (None nếu không bật)
   ↓
img_ids: dicom_id → int
```

## ★ Augmentation áp một lần, trong dataset
Comment `blip2_qformer.py:254`: *"Spatial/intensity augmentation is applied once in
ReportDataset so every encoder sees the same mildly transformed radiograph."*

Nếu mỗi encoder tự augment, chúng sẽ thấy **ba ảnh khác nhau** — và view fusion +
shared projector sẽ hợp nhất những thứ không khớp.

## Config dependencies
`datasets.mimic_cxr.vis_processor.{train,eval}.{image_size,resize_size,augmentation.*}` ·
`model.data.*` · `run.feature_cache_dir` · `run.truncate_*`

## Side effects
Đọc CSV vào RAM (`self.annotation`). Mở feature cache store.

## Modification risk
Đổi transform đổi phân phối đầu vào của **encoder đã pretrain** → kết quả không so
sánh được với checkpoint cũ.
