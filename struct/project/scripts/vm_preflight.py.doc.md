> Source: `scripts/vm_preflight.py` (202 dòng)
> Status: 🧰 UTILITY — ACTIVE
> Last verified against source: 2026-08-12

# `scripts/vm_preflight.py`

## Purpose
Kiểm tra một checkout mới có train được trên máy đích **TRƯỚC** khi bắt đầu run dài.

## Why it exists
Docstring `:4`: nó **cố ý không tải weight và không download gì**, nên chạy lại
nhiều lần là an toàn và nhanh. Phát hiện sớm một thứ thiếu rẻ hơn nhiều so với
phát hiện nó ở giờ thứ ba của một run GPU.

## Entry point
```bash
python scripts/vm_preflight.py            # chung
python scripts/vm_preflight.py --stage 1  # riêng Stage 1
```

## Main functions
| Hàm | Dòng | Kiểm gì |
|---|---|---|
| `main(argv)` | 166 | ★ Trả exit code |
| `check_python()` | 42 | Phiên bản Python |
| `check_torch_cuda(stage)` | 48 | CUDA/GPU |
| `check_ram()` / `check_disk()` / `check_shm()` | 71,83,89 | ★ `/dev/shm` hay bị quên → DataLoader worker chết |
| `check_write_perms()` | 99 | Quyền ghi output |
| `check_imports(stage)` | 112 | Import theo stage |
| `check_paths()` | 130 | Path trong env_config |
| `check_train_configs()` | 151 | `mimic_cxr_full.yaml` tồn tại |
| `check_env_vars()` | 157 | HF auth … |
| `record(status, name, detail)` | 31 | Ghi kết quả |
| `_spec(mod)` | 35 | Dò module không import |

## Side effects
Chỉ đọc. Không tải, không cấp GPU.

## Related tests
Không có.

## Developer notes
Chạy nó **trước mọi run dài**. `--stage 1` và `--stage 2` kiểm bộ import khác nhau
vì hai stage dùng hai venv.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
