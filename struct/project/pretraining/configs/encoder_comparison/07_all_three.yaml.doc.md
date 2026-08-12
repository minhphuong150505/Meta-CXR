> Source: `pretraining/configs/encoder_comparison/07_all_three.yaml` (77 dòng)
> Status: 🧪 ABLATION / historical recipe
> Last verified against source: 2026-08-12

# `pretraining/configs/encoder_comparison/07_all_three.yaml`

## Purpose

Recipe so sánh dùng cả BioViL, PubMedCLIP và Swin, resolve theo `run_name` bởi
`training/stage1/lavis_loader.py::default_stage1_config_path`.

## Key behavior

- `run_name: 07_all_three`.
- Microbatch `4` × accumulation `32` = effective batch `128`.
- `warmup_steps: 32000` được comment là quy đổi theo sample count từ recipe cũ.
- `output_dir` là absolute path `/home/phuong/output/07_all_three` → không portable.
- `wandb_entity` gắn tên account cụ thể; cần override trên máy khác.
- Không khai báo `multi_view`, các lambda loss mới, selection metric hay bf16 như
  production `mimic_cxr_full_l4.yaml`.

## Status note

Tên config còn được Stage 2 tìm theo convention khi `run_name=07_all_three`, nhưng
không phải recipe production hiện tại. Dùng nó cần đối chiếu checkpoint cùng tên.

← [`pretraining/configs/`](../_index.md) · [HOME](../../../../HOME.md)
