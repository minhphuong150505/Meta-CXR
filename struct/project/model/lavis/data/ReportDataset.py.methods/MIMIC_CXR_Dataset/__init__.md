> Source: `model/lavis/data/ReportDataset.py:222-472`
> Status: ✅ ACTIVE

# `MIMIC_CXR_Dataset.__init__(...)`

## Located in

[`ReportDataset.py`](../../ReportDataset.py.doc.md)

## Purpose
Nạp CSV, tách transform hình học/quang học, đọc mask-index, index study và mở
feature cache. Explanation `.npy` chưa được mở ở đây.

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
   geometric_trans: Resize → CenterCrop → RandomAffine (train)
   optical_trans: ColorJitter (train) → ToTensor → ExpandChannels
   ↓
_init_explanation_mask_cache(cfg)     → JSON index; memmap=None
   ↓
_init_feature_cache(cfg)              → self.feature_cache (None nếu không bật)
   ↓
_init_study_index(cfg, truncate)      → self.studies
   ↓
img_ids: dicom_id → int
```

## ★ Augmentation áp một lần, trong dataset
Comment `blip2_qformer.py:254`: *"Spatial/intensity augmentation is applied once in
ReportDataset so every encoder sees the same mildly transformed radiograph."*

Nếu mỗi encoder tự augment, chúng sẽ thấy **ba ảnh khác nhau** — và view fusion +
shared projector sẽ hợp nhất những thứ không khớp.

## ★ Mask cache không được mmap trước khi fork

`_init_explanation_mask_cache` chỉ validate tên file + JSON schema. `.npy` được
`_get_explanation_mask_memmap` mở ở lần `__getitem__` đầu trong worker và ghi
nhận PID; PID đổi thì reopen. Điều này tránh mang một mapping mở từ parent qua
DataLoader fork.

Cache cố định theo geometry 512→448; nếu `resize_size`/`image_size` bị override
khác cặp này trong lúc cache bật, dataset fail sớm thay vì dùng mask lệch ảnh.

Nếu `model.explanation.mask_cache_dir` vắng/rỗng, dataset không đọc cache và
đường transform/sample giữ hành vi trước Giai đoạn 2.

## Config dependencies
`datasets.mimic_cxr.vis_processor.{train,eval}.{image_size,resize_size,augmentation.*}` ·
`model.data.*` · `model.explanation.mask_cache_dir` · `run.feature_cache_dir` ·
`run.truncate_*`

## Side effects
Đọc CSV + explanation JSON index vào RAM. Mở feature cache store. Không mở
explanation memmap.

## Modification risk
Đổi transform đổi phân phối đầu vào của **encoder đã pretrain** → kết quả không so
sánh được với checkpoint cũ.

Khi cache bật, ảnh affine dùng bilinear theo yêu cầu explanation-aware; mask dùng
nearest. Khi cache tắt, `geometric_trans` + `optical_trans` ghép lại đúng chuỗi cũ.
