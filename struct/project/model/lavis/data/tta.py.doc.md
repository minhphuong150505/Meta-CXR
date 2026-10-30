> Source: `model/lavis/data/tta.py`
> Status: ⚠ CONDITIONAL — chỉ khi config có `datasets.mimic_cxr.vis_processor.eval.tta`
> Last verified against source: 2026-10-30

# `model/lavis/data/tta.py`

← [ReportDataset.py](ReportDataset.py.doc.md)

## Purpose

Test-time augmentation cho Stage 1: một phép xoay + phóng cố định quanh tâm, áp
lên ảnh X-quang gốc trước tiền xử lý riêng của mọi encoder (BioViL, PubMedCLIP,
MedCLIP Swin), nên mọi encoder thấy cùng một ảnh đã biến đổi. Một lượt chấm áp
một phép; TTA là trung bình xác suất giữa các lượt, tính ngoài mô hình
(`docs/handoff/PLAN-2026-10-30-stage1-tta.md`). Chỉ import PIL — suite CPU ghim được.

⚠ Thư mục `model/lavis/data/` bị `.gitignore` — thêm bằng `git add -f`.

## Config

```yaml
datasets:
  mimic_cxr:
    vis_processor:
      eval:
        tta: {degrees: 5.0, scale: 1.0}   # vắng = không biến đổi, giống từng byte
```

## API

| Tên | Vai trò |
|---|---|
| `TTAAffine` | dataclass `degrees` (ngược chiều kim đồng hồ, như `Image.rotate`), `scale` (> 1 là phóng to) |
| `resolve_tta_affine(raw, split)` | `None` khi khoá vắng; split `train`, khoá lạ, giá trị không hữu hạn hoặc `scale <= 0` → `ValueError` |
| `apply_tta_affine(image, tta)` | một lần lấy mẫu bilinear, ma trận nghịch như `Image.rotate` chia cho `scale`; góc ngoài ảnh tô 0 |

## Invariants (ghim bởi `tests/test_tta_affine.py`, 12 test)

- Khoá vắng → `None` ở mọi split; `train` có khoá → lỗi.
- `degrees=0, scale=1` trả lại đúng ảnh; xoay khớp `Image.rotate` (sai ≤ 1 mức xám).
- Phóng 2x đẩy một điểm cách tâm 10 px ra 20 px; thu nhỏ để lộ viền 0.

## Callers

`MIMIC_CXR_Dataset.__init__` (resolve; từ chối cùng `run.feature_cache_dir`) ·
`MIMIC_CXR_Dataset._row_visual` (áp; từ chối cùng `explanation_mask`) ·
`tests/test_tta_affine.py`.
