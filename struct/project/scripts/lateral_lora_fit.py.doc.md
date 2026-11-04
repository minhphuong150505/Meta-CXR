# `scripts/lateral_lora_fit.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-11-04 · 🔬 chẩn đoán (chạy trên máy train)

## Purpose
Bước 1 của `docs/handoff/PLAN-2026-11-04-residual-lateral.md`: chuyên gia trên MedCLIP-Swin cho study PL,
ba arm cùng study, cùng holdout 10%, cùng head (`build_attention_probe`), cùng augmentation (chỉ train,
không lật), cùng lịch và budget:

| arm | ảnh | LoRA |
|---|---|---|
| `L_frozen` | lateral | không (đối chứng cho phần tăng dữ liệu + augmentation) |
| `L_lora` | lateral | r=8, alpha=16, dropout 0,05, query+value stage 3–4 |
| `C_lora` | frontal | như trên (đối chứng frontal cho D2) |

Cố định trước khi có số: LR head 3e-4, LR LoRA 2e-4, wd 0,05, batch 64, bf16, tối đa 8 epoch
(warmup-cosine trên 8), dừng sớm sau 2 epoch không tăng macro AP "có bệnh" trên holdout.

## `fit`
Cổng **G0**: nhãn của cache = nhãn các thư mục Bước 0 (trên study chung; mọi study PL của Bước 0 phải có
trong cache), và Swin đóng băng trên cache tái tạo đặc trưng Swin Bước 0 (cosine ≥ 0,999, 64 study val,
cả hai view). Cổng **G2**: tham số LoRA có đổi và backbone bit-identical (arm frozen: tất cả
bit-identical). Ghi `<arm>_s<seed>.npz` (`keys`, `probabilities` [N,14,3] val PL, `labels`, đúng định
dạng `residual_stack.py --expert`) và `report_<arm>_s<seed>.json`.

## `compare`
Cổng **G1(a)**: hiệu trung bình theo seed của AP Positive trung bình trên T (F1-5) giữa hai bộ seed,
bootstrap ghép cặp 2.000 lần seed 16, NB từ seed; qua khi cận dưới > 0 và delta > NB/√k. `--names-from`
lấy thứ tự finding từ npz của OFF. G1(b) chạy bằng [`residual_stack.py`](residual_stack.py.doc.md) với
`L=L_lora`, `C=C_lora`.

Hàm thuần `should_stop`, `shared_rows`, `compare`, `load_set` ghim bởi `tests/test_lateral_lora.py`.
