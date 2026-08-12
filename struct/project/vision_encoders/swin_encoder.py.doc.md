> Source: `vision_encoders/swin/swin_encoder.py` (140 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `vision_encoders/swin/swin_encoder.py`

## Purpose

Bọc SwinV2 thành encoder đóng băng, hỗ trợ **hai backend**: timm và HuggingFace.

## Status

```text
✅ ACTIVE — encoders.swin: true ở config production
```

## Main class

`SwinEncoder(nn.Module)` (`:10`)

| Method | Dòng | Vai trò |
|---|---|---|
| `__init__` | 11 | Chọn backend, đóng băng |
| `_init_timm` | 50 | Backend timm |
| `_init_huggingface` | 71 | Backend HF (**mặc định**) |
| `train(mode=True)` | 110 | Override giữ eval |
| `forward(x)` | 116 | `[B,3,H,W]` → `[B,P,embed_dim]` |

`self.embed_dim` được đọc ra sau khi model load — `blip2_qformer.py:272` dùng nó
làm `swin_dim`. ⚠ **Giá trị cụ thể cần runtime verification**; nó phụ thuộc
checkpoint `ChayanM/SwinV2-GPT2_Mimic`.

## Important configuration

```yaml
model:
  encoders: {swin: true}
  swin:
    backend: hf                              # hoặc timm
    model_name: "ChayanM/SwinV2-GPT2_Mimic"
    pretrained: true
    frozen: true
    normalize: true
```

Đọc ở `blip2_qformer.from_config:1332-1352`, có fallback từ key phẳng
(`swin_model_name`, `swin_backend`, …).

## ⚠ Swin từng gây shape mismatch

Số patch của Swin khác các encoder khác. `mhcac_12._resize_patch_sequence` (`:317`)
đồng bộ chúng bằng adaptive avg-pool 2-D (khi số patch là chính phương) hoặc nội
suy tuyến tính 1-D.

Notebook legacy `03` từng **vá `mhcac_12.py` bằng string replacement lúc runtime**
(`ensure_swin_mhcac_shape_patch`, dòng 843–874) cho đúng việc này. Giờ source đã
xử lý — đừng dùng lại cách vá đó.

## Calls / Called by

Gọi: `timm` hoặc `transformers`.
Được gọi: `blip2_qformer.py:27` (import), `:261` (dựng), `:448` (aux), `:524` (anchor).

## Side effects

Tải weight từ HF hub lần đầu. Cấp phát GPU.

## Error / edge cases

⚠ Backend không hợp lệ, model_name không tồn tại — hành vi cần runtime verification.

## Related tests

Không có test trực tiếp.

## Developer notes

1. `embed_dim` đọc từ model đã load, không hardcode — nhưng nó chảy vào
   `SharedVisualTokenProjector` và `ViewFusionModule`, nên **đổi checkpoint Swin
   sẽ đổi shape state dict** của cả hai.
2. Backend `hf` là mặc định và là thứ config production dùng.

## Source relationships

- **Parent:** [`vision_encoders/_index.md`](_index.md)
- **Related:** [`mhcac_12.py`](../mhcac/mhcac_12.py.doc.md) (`_resize_patch_sequence`)

← [HOME](../../HOME.md)
