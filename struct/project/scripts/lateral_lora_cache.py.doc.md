# `scripts/lateral_lora_cache.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-11-04 · 🔬 chẩn đoán (chạy trên máy train)

## Purpose
Bước 1 của `docs/handoff/PLAN-2026-11-04-residual-lateral.md`: cache uint8 đầu vào 224 của
MedCLIP-Swin cho **cả hai view** của mọi study PL (frontal-anchor + một lateral). Một lượt dataset với
`anchor_priority=[PA, AP, lateral]`, `max_aux_views=1`: `swin_image` là ảnh frontal, `aux_swin_image`
là ảnh lateral (đúng ảnh `encoder_probe_features.py --views lateral` mã hoá). Chuyển uint8 bằng
`swin_finetune_cache.swin_image_to_uint8`.

## Đầu ra
`frontal.npy`, `lateral.npy` `[N,224,224]` uint8, `labels.npy` (-100 = không nhãn), `keys.npy` (key
frontal-anchor), `image_keys.npy` (key ảnh lateral), `view_map.npz`. `--limit` lấy tập con có seed như
encoder probe rồi mới lọc PL. `--split test` bị từ chối.

## Kiểm tra (đều dừng chương trình)
Thứ tự study của dataset = index dựng lại; mỗi study giữ lại có aux đúng là lateral PL và anchor đúng
là frontal mặc định (`pl_positions`); key mỗi batch = key frontal kỳ vọng, aux là lateral thật
(`check_batch`). Dữ liệu dẫn xuất MIMIC-CXR: chỉ để trên máy train.
