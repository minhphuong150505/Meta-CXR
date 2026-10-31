# `scripts/swin_finetune_cache.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-31 · 🔬 chẩn đoán (chạy trên máy train)

## Purpose
Cache uint8 đầu vào 224×224 của MedCLIP-Swin cho phép đo tinh chỉnh Swin
(`docs/handoff/PLAN-2026-10-31-swin-finetune-probe.md`). Cùng study, thứ tự, nhãn và key với
[`encoder_probe_features.py`](encoder_probe_features.py.doc.md): chỉ ảnh anchor, không augmentation,
tập con `--limit` theo seed. Lưu ảnh của `medclip_preprocess` TRƯỚC khi chia 255 và chuẩn hóa (pad vuông
đen, resize 224 bicubic, một kênh). Từ chối split `test`.

## Đầu ra (chỉ trên máy train — dữ liệu dẫn xuất MIMIC)
`images.npy` `[N, 224, 224]` uint8, `labels.npy` `[N, 14]` (0/1/2, −100 = không nhãn), `keys.npy`
(`sample_keys` có salt của eval hook).

## Hàm thuần
`swin_image_to_uint8`: đảo chuẩn hóa `swin_image` của dataset về đúng giá trị uint8 (ảnh chỉ giải mã
một lần, khớp từng bit đầu vào của pipeline). Ghim bởi `tests/test_swin_finetune.py`.
