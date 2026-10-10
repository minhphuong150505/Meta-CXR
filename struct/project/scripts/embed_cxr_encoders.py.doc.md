# `scripts/embed_cxr_encoders.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-11 · 🔬 chẩn đoán (chạy trên máy train)

## Purpose
Đặc trưng gộp **từng encoder** cho probe tuyến tính so sánh encoder Stage 1 (với
[`probe_image_features.py`](probe_image_features.py.doc.md)): BioViL-T (trung bình patch),
PubMedCLIP (CLS), MedCLIP-Swin (token pooled) và tùy chọn **RAD-DINO** (`--raddino <thư mục
local>`, [CLS ; trung bình patch]) — encoder repo hỗ trợ nhưng đang tắt. Chạy qua dataset
Stage 1 nên mỗi encoder nhận đúng tiền xử lý của nó; dùng cùng tập study con (seed/limit)
với các probe khác. `--checkpoint none` = encoder pretrained; một checkpoint Stage 1 = encoder
đã tinh chỉnh của checkpoint đó.

`--softtok` ghi thêm trung bình 32 soft token Q-Former từ `Blip2Qformer.forward_image` —
đúng thứ Stage 2 cache — để so thông tin soft token giữa checkpoint 1a / 1b / 1c.
`--no-encoder-features` bỏ ba đặc trưng encoder khi chỉ cần `softtok`.

## Hai điểm phải giữ (sửa 2026-10-11)
- **Key = tên file ảnh CÓ đuôi** (`Path(image_path).name`), vì `probe_image_features.py`
  khớp theo `image_path.rsplit("/", 1)[-1]` của record cache. Bản đầu ghi `.stem` (không
  đuôi) nên probe sẽ không khớp được study nào.
- **Tắt augmentation train trước khi dựng dataset.** Split `train` mặc định áp affine +
  jitter ngẫu nhiên lên tensor BioViL (PubMedCLIP và Swin có tiền xử lý riêng, không
  augmentation), nên đặc trưng train và test sẽ đến từ hai phân phối ảnh khác nhau. Script
  đặt `vis_processor.train.augmentation.enabled = False` và dừng nếu dataset vẫn bật.

Lưu ý liên quan (chưa sửa, ghi nhận): `build_stage1_records` của Stage 2 dựng dataset
train theo cùng cách, nên soft token / cue của record **train** Stage 2 được tính từ ảnh
BioViL đã augmentation ngẫu nhiên (một lần cho mỗi study), còn val/test từ ảnh sạch.

## Đầu ra
Một `.npz` mỗi nguồn (`<out-prefix>_<tên>.npz`, `keys` + `emb` fp16). Dữ liệu dẫn xuất —
chỉ để trên máy train. In số lượng, không in nội dung.
