> Source: `pyproject.toml` (config)
> Status: 🧰 UTILITY
> Last verified against source: 2026-08-12

# `pyproject.toml`

## Purpose
Cấu hình tooling. **Cố ý khai báo ZERO runtime dependency.**

## ★ Vì sao zero dependency
Runtime pin nằm trong các requirements file theo workflow thay vì metadata package.
`requirements-stage2.txt` hiện **include** `requirements-stage1.txt` rồi thêm QLoRA
extras, nên pin hiện tại không xung đột. Hai venv vẫn được cloud setup khuyến nghị
để cách ly môi trường và giữ Stage 1 nhẹ hơn.

Khai báo runtime dependency ở đây sẽ làm `pip install .` mang nghĩa package mà
repo path-based hiện chưa hỗ trợ và che khuất lock file nào đang được dùng.

Repo **không phải package src-layout**; module import theo path từ repo root.

## Nội dung chính
| Mục | Vai trò |
|---|---|
| `[tool.ruff]` | Lint (không có formatter pass) |
| ruff exclude | ★ `model/lavis/` và `mhcac/mhcac_8..11.py` |

## ⚠ Vì sao loại `model/lavis/` khỏi ruff
Đó là fork Salesforce LAVIS. Reformat sẽ làm **mọi diff upstream sau này không đọc
được**. `mhcac_8..11` là variant legacy — dấu hiệu chúng đã được coi là legacy từ
trước.

## Lệnh
```bash
ruff check .
```

## Developer notes
**Đừng thêm runtime dependency vào đây** nếu chưa biến repo thành package cài đặt
thật. Sửa lock file đúng workflow và giữ quan hệ include Stage2→Stage1 rõ ràng.

← [HOME](../HOME.md)
