# `scripts/lateral_expert_fit.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-11-04 · 🔬 chẩn đoán (chạy trên máy train)

## Purpose
Bước 0 của `docs/handoff/PLAN-2026-11-04-residual-lateral.md`: fit **chuyên gia lateral `L_enc`**
(`--view lateral`) và **đối chứng frontal `C_enc`** (`--view frontal`) trên CÙNG các study PL của
tập train, một encoder đóng băng, ba lớp, head attention của encoder probe
(`encoder_probe_fit.build_attention_probe`), cùng recipe với `encoder_probe_fit.fit_attention`
(AdamW 3e-4, wd 0,05, OneCycle, batch 256, 12 epoch, chọn epoch theo macro AP "có bệnh" trên
holdout 10%). Khác `fit_attention`: lưu xác suất val **từng seed** (cần để ước NB).

Tập study luôn là `keys.npy` của thư mục lateral (key frontal-anchor của study PL); đặc trưng
frontal (`~/encprobe_20261031`) được căn theo key. **Cổng G0:** nhãn hai thư mục phải trùng từng ô,
nếu không thì dừng.

## Đầu ra
`<out>/<encoder>_<view>_s<seed>.npz` (`keys`, `probabilities` [N,14,3] trên val PL, `labels` -100 =
không nhãn) và `report_<encoder>_<view>.json` (lịch sử holdout, epoch chọn, macro AP val).
Từ chối ghi đè.

## Hàm
`holdout_split` (giống `swin_finetune_fit.holdout_split`), `normalise_missing` (-1 → -100),
`load_view`, `check_labels`, `fit_one_seed`.

## Liên kết
Đầu vào: [`encoder_probe_features.py`](encoder_probe_features.py.doc.md) (`--views lateral` và mặc định).
Đọc kết quả: [`residual_stack.py`](residual_stack.py.doc.md). Test: `tests/test_residual_lateral.py`.
