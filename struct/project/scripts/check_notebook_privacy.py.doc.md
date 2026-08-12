> Source: `scripts/check_notebook_privacy.py` (363 dòng)
> Status: ✅ ACTIVE — ★ pre-commit hook
> Last verified against source: 2026-08-12

# `scripts/check_notebook_privacy.py`

## Purpose
Chặn notebook mang dữ liệu MIMIC-CXR vào Git.

## ★ Vì sao đây là script quan trọng nhất trong `scripts/`
Docstring `:3`: MIMIC-CXR là dữ liệu credentialed, DUA cấm redistribute, và remote
này **public**. Notebook là đường rò rỉ dễ nhất — source trông vô hại, nhưng
**outputs** nhúng patient identifier, report text và `findings_clean`.

`.gitignore` bảo vệ hai notebook đã biết, nhưng **một `git add -f`, một lần đổi
tên, hay một notebook mới là đủ để vượt qua**.

> **Đừng bypass hook này.**

## Entry point
Chạy tự động qua `.pre-commit-config.yaml`. Thủ công:
```bash
python scripts/check_notebook_privacy.py
```

## Main functions
| Hàm | Dòng | Vai trò |
|---|---|---|
| `main(argv)` | 316 | ★ |
| `check_notebook(path)` | 180 | ★ Quét một notebook |
| `_scan_text(text, path, cell_index, location)` | 99 | ★ Khớp pattern |
| `_output_text(output)` | 154 | Trích text từ output cell |
| `sanitize_notebook(source, destination)` | 225 | Dọn outputs |
| `redact(value)` | 75 | ★ Che giá trị trong báo cáo lỗi |
| `staged_notebooks()` / `tracked_notebooks()` / `is_tracked(path)` | 254,259,264 | Qua `git` |
| `report(violations)` | 272 | ★ In vi phạm **đã redact** |
| `Violation` | 85 | |
| `build_parser()` | 300 | |

## ★ `redact()` — chi tiết dễ bỏ sót
Báo cáo vi phạm **không được in nguyên văn** dữ liệu vi phạm — làm vậy là rò rỉ
lần hai, vào log CI.

## Calls / Called by
Gọi: `json`, `re`, `subprocess` (git).
Được gọi: `.pre-commit-config.yaml`; `tests/test_notebook_privacy.py`.

## Side effects
Đọc notebook; `sanitize_notebook` ghi bản sạch; chạy `git`.

## Related tests
`tests/test_notebook_privacy.py` (279 dòng) + 5 fixture ở `tests/fixtures/notebooks/`
(clean, credential_like, executed_output, kaggle_ids, synthetic_identifier)

⚠ Fixture dùng đuôi `.ipynb.fixture` để **không** bị chính script này quét — chúng
cố ý chứa pattern giống dữ liệu thật.

## Developer notes
Thêm notebook mới vào repo? Chạy script này trước khi commit, và đảm bảo outputs
rỗng.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
