> Source: `utils/split_emb.py` (22 dòng)
> Status: ⚠⚠ NGUY HIỂM KHI IMPORT
> Last verified against source: 2026-08-12

# `utils/split_emb.py`

## ⚠⚠ File này CHẠY ngay khi import

Không có `if __name__ == "__main__"` guard. Toàn bộ nội dung ở module scope:

```python
input_path = "pretraining/embs/2025_05_20_blip2_pretrain_stage1_emb_embeddings_iu_xray_test.pkl"
...
with open(input_path, "rb") as f:      # ← chạy ngay khi import
```

`pretraining/embs/` bị `.gitignore` chặn và **không tồn tại** trong working tree.

**Hệ quả:** bất kỳ `import utils.split_emb` hay `from utils import *` nào cũng
thực thi và ném `FileNotFoundError`.
Ghi nhận là [I3](../_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa).

## Purpose (theo thiết kế)
Tách một file pickle embedding lớn thành nhiều file nhỏ theo `dicom_id`.

## Status
```text
⚠ POTENTIALLY_UNUSED — và không an toàn để import
```

## Calls / Called by
Gọi: `pickle`, `os`, `pathlib`.
Được gọi: **không ai**.

## Side effects
⚠ Đọc pickle và **ghi hàng loạt file** ngay lúc import.

## Developer notes
⚠ **Đừng thêm re-export vào `utils/__init__.py`** — sẽ kéo file này vào và làm mọi
`import utils` nổ.
Nếu cần dùng: thêm `if __name__ == "__main__":` và chuyển path thành argparse.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
