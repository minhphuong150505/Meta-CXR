> Source: `cloud/run_encoder_comparison.sh` (6 dòng)
> Status: 🕰 COMPATIBILITY ALIAS
> Last verified against source: 2026-08-12

# `cloud/run_encoder_comparison.sh`

## Purpose

Backward-compatible tên lệnh cũ. Implementation chỉ `exec run_stage1.sh`; nó
**không tự quét nhiều encoder config**.

Mọi argument/process state được thay bằng `run_stage1.sh`. Encoder comparison ở
commit hiện tại là evaluation-time ablation; tên file không phản ánh behavior
riêng nữa.

← [`cloud/`](_index.md) · [`run_stage1.sh`](run_stage1.sh.doc.md)
