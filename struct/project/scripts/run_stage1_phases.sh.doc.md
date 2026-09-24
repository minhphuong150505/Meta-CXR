> Source: `scripts/run_stage1_phases.sh`
> Status: ✅ ACTIVE — cách chạy Stage 1 production (D-020)
> Last verified against source: 2026-09-24 (smoke)

# `scripts/run_stage1_phases.sh`

← [scripts](_index.md) · [D-020](../_meta/DECISIONS.md#d-020--meta-former-3-pha-theo-bài-báo-medclip-swin-mhcac-một-nhánh)

`ROOT=<thư mục mới> bash scripts/run_stage1_phases.sh`. Kiểm tra: ổ dữ liệu là
`ntfs3`, không tiến trình train/generate nào đang chạy, `<ROOT>/<pha>` chưa tồn
tại. Nếu chạy pha 1a mà chưa có cache: `precompute_features --anchor-only`
(tự DỪNG nếu không đủ đĩa). Mỗi pha một lần `pretraining.train` với
`run.phase`, `run.phase_root=ROOT`, `run.output_dir=ROOT/<pha>`
(`run.feature_cache_dir` chỉ cho 1a). Exit 3 nếu gate ITC pha 1a trượt
(`PHASE_GATE_FAILED`), exit 4 nếu một pha xong mà không có
`ROOT/checkpoint_<pha>.pth`. Biến: `CACHE`, `PHASES`, `EXTRA_OPTS`, `CACHE_OPTS`,
`RESUME=1` (2026-09-25): pha đã có `ROOT/checkpoint_<pha>.pth` thì bỏ qua, pha có
thư mục thì resume từ `checkpoint_last.pth`; nếu `ROOT` đã có `PHASE_GATE_FAILED`
thì exit 3 luôn. Kiểm chứng trên GPU 2026-09-25: cổng bị ép trượt sau epoch 2 →
exit 3, không chạy 1b, không có `checkpoint_phase1a`; kill giữa epoch rồi
`RESUME=1` → nạp `checkpoint_last` (chạy lại epoch từ batch đầu), hoàn tất.
