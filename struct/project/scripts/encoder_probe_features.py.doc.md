# `scripts/encoder_probe_features.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-31 · 🔬 chẩn đoán (chạy trên máy train)

## Purpose
Trích đặc trưng **từng encoder** cho phép đo encoder
(`docs/handoff/PLAN-2026-10-31-encoder-probe.md`): mỗi study (chỉ ảnh anchor, ảnh sạch, encoder
pretrained) ghi một **vector toàn cục** `<tên>_global.npy` `[N, Dg]` và một **lưới token thô**
`<tên>_grid.npy` `[N, T, D]` (memmap float16), cùng `labels.npy` (`[N, 14]`, 0/1/2, -100 = không
nhãn) và `keys.npy` — digest có salt giống `sample_keys` mà eval hook Stage 1 ghi, để ghép cặp với
dự đoán val của MHCAC.

| Tên | Tiền xử lý | Toàn cục | Lưới |
|---|---|---|---|
| `swin` | `swin_image` của dataset | token pooled (768) | 7×7 × 768 |
| `pubmedclip` | `pubmedclip_image` của dataset | CLS (768) | 7×7 × 768 |
| `biovil` | `image` 448 của dataset | `img_embedding` **512** (đầu ra thật của BioViL-T, KHÔNG qua projector 1408 ngẫu nhiên của pipeline) | 14×14 → 7×7 × 512 |
| `biovil128` | như trên | trung bình projector 128 chiều đã pretrain của BioViL-T | — |
| `raddino` (`--raddino DIR`) | gốc: cạnh ngắn 518, cắt giữa 518, mean/std của nó | [CLS ; trung bình patch] | 37×37 → `--grid` |
| `medsiglip` (`--medsiglip ID`) | gốc: 448×448, mean/std 0,5 | [pooler ; trung bình patch] | 32×32 → `--grid` |

`max_aux_views=0`, augmentation tắt (dừng nếu vẫn bật), cache mask giải thích tắt, autocast bf16.
`--split test` bị từ chối. Ảnh RAD-DINO/MedSigLIP giải mã lại từ cùng ảnh anchor trong worker
(`_WithNative`).

## Hàm thuần (ghim bởi `tests/test_encoder_probe.py`)
`raddino_preprocess`, `medsiglip_preprocess`, `grid_tokens` (patch theo hàng → gộp trung bình).

## Bẫy
- **Phải dùng gói `biovil_t/` ở gốc repo** (như pipeline), không phải `vision_encoders/biovil_t/`:
  bản sau thiếu `device.py`, import lỗi. Script chỉ thêm gốc repo và `training/` vào `sys.path`.
- Đặc trưng là dữ liệu dẫn xuất MIMIC-CXR: chỉ để trên máy train.

Đọc kết quả: [`encoder_probe_fit.py`](encoder_probe_fit.py.doc.md).
