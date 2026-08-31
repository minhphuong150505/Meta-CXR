> Source: `scripts/train_healthcheck.sh` (shell)
> Status: 🧰 ACTIVE UTILITY
> Last verified against source: 2026-08-31

# `train_healthcheck.sh`

## Purpose

Monitor chỉ đọc cho Stage 1 và Stage 2. Script in báo cáo ngắn để người dùng hoặc
cron quyết định có cần triage hay không; nó không kill, restart hay đổi recipe.

## Inputs và exit status

| Biến | Ý nghĩa |
|---|---|
| `RUN_DIR` | thư mục artifact đang theo dõi |
| `LOG` | raw training log |
| `EXPECT_RUNNING=1` | process biến mất là ALERT thay vì IDLE |
| `STALL_MIN` | ngưỡng phút không có checkpoint write, mặc định 45 |
| `GPU_IDLE_PCT` | ngưỡng GPU idle khi process còn sống, mặc định 5 |

Exit: `0=OK`, `2=WARN`, `3=ALERT`, `4=IDLE`.

## Progress invariant

Progress là **mtime mới nhất**, không phải số file. Stage 1 rewrite `*.pth`; Stage
2 rewrite `adapter_model.safetensors` và `trainer_state.pt` trong
`checkpoints/last/`. Cả ba định dạng phải được giữ trong phép tìm kiếm.

`find -printf %T@` trả timestamp có phần thập phân; script bỏ phần này trước khi
tính tuổi bằng shell arithmetic. Không làm vậy gây `arithmetic syntax error` và
vô hiệu hóa cảnh báo stall.

## Health signals

- process Python thật, loại shell tự match;
- GPU utilisation/VRAM/nhiệt/công suất/quạt và cờ thermal slowdown;
- CPU package và NVMe temperature;
- checkpoint write age và log write age;
- kernel corruption signatures, dung lượng `/home`, mount `/mnt/drive1tb`.

`EXPECT_RUNNING=1` là ràng buộc dành cho thí nghiệm đã lên lịch. Nó bắt trường hợp
GPU/driver làm process biến mất hoàn toàn — trường hợp mà kiểm tra GPU-idle chỉ
khi process sống không thể thấy.

## Called by

Người dùng qua SSH và cron wrapper trên training host. `scripts/supervise_stage1.sh`
là supervisor có quyền can thiệp; file này chỉ quan sát.

## Tests

`tests/test_train_healthcheck.py` kiểm trạng thái expected-run bị mất và artifact
cứu hộ Stage 2 với timestamp thập phân.

← [scripts/](./_index.md) · [HOME](../../HOME.md)
