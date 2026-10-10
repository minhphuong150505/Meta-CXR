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
`RESUME=1` (2026-09-24): pha đã có `ROOT/checkpoint_<pha>.pth` thì bỏ qua, pha có
thư mục thì resume từ `checkpoint_last.pth`; nếu `ROOT` đã có `PHASE_GATE_FAILED`
thì exit 3 luôn. Kiểm chứng trên GPU 2026-09-24: cổng bị ép trượt sau epoch 2 →
exit 3, không chạy 1b, không có `checkpoint_phase1a`; kill giữa epoch rồi
`RESUME=1` → nạp `checkpoint_last` (chạy lại epoch từ batch đầu), hoàn tất.

## 2026-10-11 — mặc định chỉ 1a → 1b
`PHASES` mặc định `"phase1a phase1b"`; pha 1c chỉ chạy khi đặt rõ
`PHASES="phase1a phase1b phase1c"` (tái lập run cũ). Lý do: trên test, 1b tốt nhất so với 1c
AUROC +0,0037 [−0,0012; +0,0095] (không có ý nghĩa) và 1c làm ITC R@5 0,65 → 0,17
(`docs/TONG_HOP_2026-10-11.md`). Vì 1b là pha cuối, `phase1b.run.test_splits: [test]`.
