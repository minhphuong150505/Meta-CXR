> Source: `scripts/` (21 file)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-31

# `scripts/`

## Purpose

CLI cho những việc **không phải training**: preflight, calibrate, evaluate, phân
tích prompt, chẩn đoán kiến trúc, và guard quyền riêng tư.

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

### Vận hành

| File | Doc | Status | Vai trò |
|---|---|---|---|
| `vm_preflight.py` (202) | [📄](vm_preflight.py.doc.md) | 🧰 | Kiểm tra CUDA, RAM, disk, shm, path, HF auth. **Không tải weight, không download** |
| `train_healthcheck.sh` | [📄](train_healthcheck.sh.doc.md) | 🧰 | Monitor chỉ đọc Stage 1/2; exit 0/2/3/4 cho OK/WARN/ALERT/IDLE |
| `check_notebook_privacy.py` (363) | [📄](check_notebook_privacy.py.doc.md) | ✅ ★ | Pre-commit hook chặn notebook mang dữ liệu MIMIC vào Git |
| `check_itc_gate.py` (305) | [📄](check_itc_gate.py.doc.md) | 🔬 | Cổng ITC: ITC đã thoát chance chưa, trước khi đốt ~33 h GPU. Exit 1 khi trượt |

### Evaluation

| File | Doc | Status |
|---|---|---|
| `calibrate_thresholds.py` | [📄](calibrate_thresholds.py.doc.md) | ✅ Calibrate — **chỉ validation** |
| `evaluate_stage1.py` (328) | [📄](evaluate_stage1.py.doc.md) | ✅ Chấm classification từ `.npz` |
| `evaluate_stage2.py` (294) | [📄](evaluate_stage2.py.doc.md) | ✅ Chấm generation từ `.jsonl` |
| `evaluate_explanation.py` | [📄](evaluate_explanation.py.doc.md) | ✅ XAI — load checkpoint, cần autograd sống, không train |
| `explain_stage2.py` | [📄](explain_stage2.py.doc.md) | ✅ XAI Stage 2 — JSONL + npz lưới gốc; cổng triệt tiêu huỷ được cả lần chạy |

### Phân tích prompt

| File | Doc | Status | Cảnh báo |
|---|---|---|---|
| `run_prompt_ablation.py` | [📄](run_prompt_ablation.py.doc.md) | 🧪 | **Dry run** — không load model, không sinh metric model |
| `export_stage2_prompt_samples.py` | [📄](export_stage2_prompt_samples.py.doc.md) | 🧪 | ⚠ **Output CHỨA findings text** — `--output` phải ở nơi riêng tư |
| `prompt_length_statistics.py` | [📄](prompt_length_statistics.py.doc.md) | 🧪 | Không có tokenizer MedGemma → fallback whitespace, đánh dấu `"approximate": true` |
| `audit_temporal_targets.py` | [📄](audit_temporal_targets.py.doc.md) | 🧪 | Đo ngôn ngữ so sánh thời gian khi input không có prior |

### Chẩn đoán kiến trúc

| File | Doc | Status | Vai trò |
|---|---|---|---|
| `diagnose_stream_scale.py` (231) | [📄](diagnose_stream_scale.py.doc.md) | 🔬 | Đo RMS token mỗi encoder tại điểm nối + attention mass của MHCAC. **Read-only** — đo được chênh lệch **32×**, xem trang doc |
| `probe_soft_tokens.py` (215) | [📄](probe_soft_tokens.py.doc.md) | 🔬 | Linear probe trên 32 soft token của Q-Former, trước khi đốt ~70 h cho arm C. Đo được **macro AUROC 0,6847** (xáo trộn 0,4838; MHCAC 0,7643) — soft token có tín hiệu dù cross-attention chưa train |
| `_stage2_fixtures.py` | [📄](_stage2_fixtures.py.doc.md) | 🧪 | ⚠ Dữ liệu **tổng hợp, KHÔNG phải MIMIC** |
| `__init__.py` | — | ✅ | |

## `check_notebook_privacy.py` — quan trọng hơn nó có vẻ

MIMIC-CXR là dữ liệu credentialed và remote này **public**. Notebook là đường rò
rỉ dễ nhất: source trông vô hại, nhưng **outputs** nhúng `subject_id`, `study_id`,
report text và `findings_clean`.

`.gitignore` bảo vệ hai notebook đã biết — nhưng một `git add -f`, một lần đổi
tên, hay một notebook mới là đủ để vượt qua. Script này chạy như **pre-commit
hook**. Fixture kiểm thử ở `tests/fixtures/notebooks/`.

> **Đừng bypass hook này.**

## `_stage2_fixtures.py` — vì sao tồn tại

Cho phép các script prompt chạy end-to-end **không cần dữ liệu, không cần GPU**.
Record có đúng hình dạng pipeline thật phát ra (`pred_groups`, views, prior flags,
`ref`).

> Mọi con số sinh từ fixture là **minh họa**, không bao giờ là kết quả model.

## Main responsibilities

1. Kiểm tra máy trước run dài và theo dõi run đang chạy mà không can thiệp.
2. Calibrate threshold rồi chấm điểm classification/generation — không GPU.
3. Phân tích prompt mà không train.
4. Chặn rò rỉ dữ liệu vào Git.
5. Chạy pass Stage-1 có grad để tạo và chấm CAM, không optimizer/update.

## Entry points

Xem [ENTRYPOINTS.md](../_meta/ENTRYPOINTS.md#python--utility).

## Dependencies

`training/evaluation/*` (lõi chỉ cần numpy) · `stage2/prompts/*` (stdlib-only) ·
`training/dataio/manifest` · `numpy`. Riêng `evaluate_explanation.py` import trễ
Stage-1/LAVIS, torch, pandas/Pillow và cần checkpoint + dataset. Plot cần extra
`eval-plots`; METEOR/CIDEr/BERTScore cần `eval-generation`.

## Used by

Người dùng. `check_notebook_privacy.py` được `.pre-commit-config.yaml` gọi.

## Status

```text
✅ ACTIVE — preflight, evaluation, privacy guard
🧪 EXPERIMENTAL — 5 script phân tích prompt
```

## Notes

- **`calibrate_thresholds.py` phải chạy TRƯỚC `evaluate_stage1.py`**, và **chỉ
  trên validation**. Calibrate trên test là rò rỉ test set.
- `--min-positive 20`: bệnh lý dưới 20 mẫu positive giữ threshold 0.5.
- `evaluate_stage1.py:294` import `visualization` **trễ, trong hàm** — script vẫn
  chạy được khi không có matplotlib.
- `evaluate_stage2.py:159` import `clinical` trễ tương tự, và báo **unavailable**
  chứ không phải 0.
- `evaluate_explanation.py` là ngoại lệ model-driven: `RunnerBase.eval_epoch` có
  `@torch.no_grad()` nên CAM không thể đi qua evaluator hiện hữu. PNG/NPZ là dữ
  liệu bệnh nhân; output trong repo chỉ được phép ở vùng Git ignore và không có
  identifier trong tên file/stdout.

## Related documentation

[PIPELINES.md → P4, P5, P6](../_meta/PIPELINES.md#p4--evaluation-stage-1) ·
[`training/evaluation/_index.md`](../training/evaluation/_index.md) ·
[`stage2/prompts/_index.md`](../stage2/prompts/_index.md)

← [Về HOME](../../HOME.md)
