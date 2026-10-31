# `scripts/biovil_prior_features.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-31 · 🔬 chẩn đoán (chạy trên máy train)

## Purpose
Đặc trưng Swin + BioViL-T có và không có ảnh lần chụp trước, cho phép đo
`docs/handoff/PLAN-2026-10-31-biovil-prior-probe.md`. BioViL-T là encoder hai ảnh: 512 chiều mỗi ô = 256 chiều tĩnh
của ảnh hiện tại + 256 chiều khác biệt từ transformer thời gian. Không có ảnh trước thì nửa khác biệt là hằng số
`missing_previous_emb` — đúng trường hợp pipeline luôn chạy. Cùng study, thứ tự, nhãn, key với
[`encoder_probe_features.py`](encoder_probe_features.py.doc.md). Từ chối split `test`.

## Đầu ra (chỉ trên máy train)
`swin_*`, `biovil_*` (không ảnh trước), `biovilprior_*` (có ảnh trước khi có, không thì bằng `biovil_*`) —
`*_global` `[N, D]`, `*_grid` `[N, 49, D]` fp16; `has_prior.npy`, `prior_days.npy`, `labels.npy`, `keys.npy`.

## Hàm (ghim bởi `tests/test_biovil_prior_probe.py`)
- `prior_frontal_table(split, times)`: lần chụp trước = study sớm hơn hẳn (`merge_asof`, `allow_exact_matches=False`)
  gần nhất của cùng bệnh nhân có ảnh thẳng; PA trước AP rồi `dicom_id` nhỏ nhất; hòa thời điểm lấy `study_id` lớn hơn.
- `prior_lookup`: chỉ cho ảnh trước khi anchor là ảnh thẳng.
- Kiểm tra lúc chạy: (a) phép biến đổi ảnh trước = tensor `image` của dataset (4 study); (b) ảnh trước giữ nguyên nửa
  tĩnh (cosine ≥ 0,999) và đổi nửa khác biệt.
