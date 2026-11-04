# `vision_encoders/lora.py`

> Thư mục: [`vision_encoders/`](_index.md) · Thêm 2026-11-04 · 🧪 opt-in (chỉ script chẩn đoán dùng)

## Purpose
LoRA tối giản cho `nn.Linear` (Hu et al. 2021), chỉ dùng torch, vì venv Stage 1 không có `peft` và
không được thêm dependency. Dùng ở Bước 1 của `docs/handoff/PLAN-2026-11-04-residual-lateral.md`
([`lateral_lora_fit.py`](../scripts/lateral_lora_fit.py.doc.md)). Không module nào của pipeline train
import file này.

## API
- `LoRALinear(base, r, alpha, dropout)`: `base(x) + (alpha/r) · B(A(dropout(x)))`; `base` đóng băng,
  `A` Kaiming-uniform, `B = 0` → ở bước 0 đầu ra trùng từng bit. `merge()` cộng `BA` vào trọng số gốc.
- `apply_lora(module, pattern, r, alpha, dropout)`: đóng băng mọi tham số cũ, bọc mọi `nn.Linear` có tên
  khớp regex; trả về danh sách tên; báo lỗi nếu không khớp gì.
- `SWIN_STAGE34_QV`: query + value ở stage 3–4 của HF `SwinModel` (16 lớp với Swin-T: 2 + 6 block).
- `lora_parameters`, `lora_modules`, `set_lora_train` (chỉ bật dropout của LoRA; `SwinEncoder(frozen=True)`
  luôn giữ backbone ở eval), `merge_lora`, `lora_state_dict`, `base_state_dict` (mọi tensor trừ LoRA, cho
  cổng "backbone bit-identical").

Ghim bởi `tests/test_lateral_lora.py`.
