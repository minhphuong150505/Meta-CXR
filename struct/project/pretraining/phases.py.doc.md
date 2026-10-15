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
| `resolve_init_checkpoint` | `init_from: {phase: X}` → `<phase_root>/checkpoint_X.pth`; `{path: <file>}` → file đó |
| `filter_init_state(state, skip, ignore, model_keys)` | 2026-10-15: bỏ tensor dưới `init_from.skip` (model có module, giữ khởi tạo của nó) và `init_from.ignore` (model của pha KHÔNG dựng module đó) khỏi state nạp vào; tiền tố nào không khớp tensor của checkpoint → `PhaseConfigError`; tiền tố `ignore` mà model lại có tensor → lỗi (phải dùng `skip`) |

`init_from` chỉ nhận `phase` / `path` / `skip` / `ignore` và phải có `phase` hoặc `path` (khóa lạ
→ lỗi; một tiền tố ở cả `skip` lẫn `ignore` → lỗi); `PhaseSpec.init_skip` / `init_ignore`.
`ignore` có vì smoke tích hợp 2026-10-15: checkpoint 1c có 18 tensor `mpc_heads.*`, mà model
`phase1q` (`lambda_mpc 0`) không dựng → trình nạp báo "tensor lạ".

**`phase1q` (2026-10-15, opt-in, không nằm trong chuỗi của `run_stage1_phases.sh`):** train lại
CHỈ META-Former trên một checkpoint đã xong, đóng băng — xem
[`mimic_cxr_full.yaml`](configs/mimic_cxr_full.yaml.doc.md) và
`docs/handoff/PLAN-2026-10-15-qformer-refit.md`. Vì `keep` là hợp các `trainable`, checkpoint của
1q KHÔNG chứa block encoder mà 1c mở bằng `encoder_finetune` → chỉ dùng qua
[`scripts/merge_qformer_refit.py`](../scripts/merge_qformer_refit.py.doc.md).

Test: `tests/test_phases.py`.
