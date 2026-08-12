> Source: `training/torch_io.py` (26 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `training/torch_io.py`

## Purpose

`load_torch_checkpoint(path)` — một hàm, một việc.

## Why it exists

Docstring `:4` nói rõ: nó tồn tại **riêng** để load checkpoint dùng được ở cả hai
phía mà **không kéo theo `training/stage1/lavis_loader.py`** (thứ pull toàn bộ
stack Stage 1).

Đây là ví dụ nhỏ nhưng chuẩn về cách giữ ranh giới kiến trúc: tách phần chung
xuống một module không phụ thuộc.

## Status

```text
✅ ACTIVE
```

## Main function

`load_torch_checkpoint(path: str | Path)` (`:16`)

⚠ Xử lý `weights_only` / `map_location` ⚠ chi tiết cần đọc code — 26 dòng, đọc trực tiếp nhanh hơn.

## Calls / Called by

Gọi: `torch.load`.
Được gọi: `training/stage1/lavis_loader.py:32` · `training/trainer/checkpointing.py:23` ·
`fig9:102,113` (dual-import).

## Side effects

Đọc file từ đĩa; cấp phát tensor.

## Related tests

Gián tiếp qua `tests/test_trainer_resume.py`.

## Developer notes

**Đừng thêm import LAVIS vào file này.** Toàn bộ lý do nó tồn tại là để không có
import đó.

← [training/](_index.md) · [HOME](../../HOME.md)
