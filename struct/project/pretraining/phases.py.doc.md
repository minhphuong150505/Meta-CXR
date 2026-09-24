> Source: `pretraining/phases.py`
> Status: ✅ ACTIVE — gọi bởi `pretraining/train.py` và `RunnerBase`
> Last verified against source: 2026-09-24

# `pretraining/phases.py`

← [pretraining](_index.md) · [D-020](../_meta/DECISIONS.md#d-020--meta-former-3-pha-theo-bài-báo-medclip-swin-mhcac-một-nhánh)

Lịch học biểu diễn 3 pha theo bài báo META-CXR. Mỗi pha = MỘT lần chạy
`pretraining.train` với `run.phase=<tên>`; `scripts/run_stage1_phases.sh` nối
các pha qua `<run.phase_root>/checkpoint_<pha>.pth`. Không có `run.phases` →
một pha như cũ; có `run.phases` mà thiếu `run.phase` → lỗi.

| Hàm | Vai trò |
|---|---|
| `apply_phase_to_config(config, name)` | gộp `run.phases.<name>.{run,model,datasets}` vào config, đặt `run.max_epoch = epochs`, tắt `encoder_finetune` trừ khi `unfreeze_encoder_blocks` |
| `build_spec` / `parse_phases` | kiểm tra khối pha (khóa lạ, prefix có dấu chấm cuối, transition trỏ ra ngoài trainable → lỗi) |
| `apply_trainable(model, spec, never_train)` | `requires_grad` = khớp prefix (`name == p` hoặc `name.startswith(p + ".")`); mọi thứ khác đóng băng, kể cả encoder |
| `param_role`, `transition_multipliers` | pha 1b: `fade_in` 0→1, `fade_out` 1→0, tuyến tính, trong `fraction` đầu số update của pha |
| `checkpoint_keep` | checkpoint theo pha giữ cả tham số mà BẤT KỲ pha nào train (Q-Former bị đóng băng cuối 1b vẫn được lưu) |
| `resolve_init_checkpoint` | `init_from: {phase: X}` → `<phase_root>/checkpoint_X.pth` |

Test: `tests/test_phases.py`.
