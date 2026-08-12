> Source: `model/lavis/runners/runner_base.py:499-526`
> Status: ✅ ACTIVE

# `RunnerBase.setup_output_dir()`

## Purpose

Chọn thư mục output/result cho Stage 1. Khi resume từ một checkpoint local, hàm
dùng **thư mục cha của chính checkpoint đó** thay vì tạo thư mục timestamp mới.

## Why this matters

Resume in-place giữ `checkpoint_best.pth`, metric history và
`checkpoint_last.pth` của các epoch trước/sau resume trong cùng một run. Nếu tách
sang thư mục mới, bước reload best cuối train có thể không thấy best checkpoint
được tạo trước lúc resume.

URL checkpoint không được coi là local path. Run mới vẫn dùng `output_dir/run_name`
và thêm timestamp nếu thư mục đã tồn tại; evaluate-only bỏ hậu tố `_eval`.

← [`runner_base.py`](../runner_base.py.doc.md) · [HOME](../../../../../HOME.md)
