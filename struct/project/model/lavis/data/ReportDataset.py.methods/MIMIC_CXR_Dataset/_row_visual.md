> Source: `model/lavis/data/ReportDataset.py:627-663`
> Status: ✅ ACTIVE

# `MIMIC_CXR_Dataset._row_visual(ann)`

## Located in

[`ReportDataset.py`](../../ReportDataset.py.doc.md)

## Purpose
Từ một hàng CSV → ảnh đã transform **hoặc** feature đã cache. Đồng thời là **chốt
bảo mật đường dẫn**.

## Signature
```python
def _row_visual(self, ann) -> dict
```

## Returns
`{"image_path": str, "image": [3,448,448]}` hoặc
`{"image_path": str, "<enc>_feat": Tensor, ...}`

## ★ Chốt bảo mật đường dẫn
```python
raw = ann["image_path"].replace("\\", "/")
marker = "/mimic-cxr-jpg-lite/"
if marker in raw: rel = raw.split(marker, 1)[1]        # tương thích Kaggle cũ
else:
    if raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
        raise ValueError(f"image_path must be relative to vis_root, got: {raw}")
    rel = raw
rel = os.path.normpath(rel)
if rel == ".." or rel.startswith(f"..{os.sep}"):
    raise ValueError(f"image_path escapes vis_root: {raw}")
image_path = os.path.join(self.vis_root, rel)
```

Ba lớp: chặn tuyệt đối POSIX (`/`), chặn tuyệt đối Windows (`C:/`), chặn path
traversal (`..`) **sau khi normpath**.

Comment `:629`: giữ hỗ trợ marker Kaggle cũ nhưng **từ chối** input tuyệt đối/traversal
thay vì âm thầm thoát khỏi `vis_root`.

## Execution flow
```text
chuẩn hóa + validate path
   ↓
feature_cache is None
   → out["image"] = general_trans(load_image(Path(image_path)))
feature_cache có
   → FOR enc, store in feature_cache:
        dicom_id không có trong store["row"] → KeyError NÊU TÊN + gợi ý sửa
        out[f"{enc}_feat"] = from_numpy(store["feats"][row]).float()
```

## Error handling
| Điều kiện | Lỗi |
|---|---|
| Path tuyệt đối | `ValueError("image_path must be relative to vis_root, got: …")` |
| Path traversal | `ValueError("image_path escapes vis_root: …")` |
| DICOM vắng trong cache | `KeyError` **nêu DICOM, split, và gợi ý `study_sampling=false`** |

Thông điệp `KeyError` (`:654`) là ví dụ tốt: nó nói **cái gì thiếu, ở đâu, và cách sửa**.

## Config dependencies
`paths.mimic_cxr_jpg_root` → `vis_root` · `run.feature_cache_dir`

## Modification risk
⚠ **Đừng nới lỏng validate path.** Nó là thứ ngăn một CSV bị sửa đọc file tùy ý
trên máy.
