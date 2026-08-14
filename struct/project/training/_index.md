> Source: `training/` (32 file Python)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `training/`

## Purpose

**Stage 2** (sinh báo cáo bằng MedGemma), cộng hai thứ dùng chung: lớp đọc dữ
liệu độc lập với LAVIS (`dataio/`) và toàn bộ evaluator (`evaluation/`).

## ⚠ Ràng buộc quan trọng nhất của thư mục này

Mọi import LAVIS/Stage-1 **chỉ được nằm trong `training/stage1/lavis_loader.py`**.

```text
run_medgemma_qlora.py, pipeline_modes.py, dataio/, medgemma/, stage2_utils.py
        │  KHÔNG import LAVIS ở module scope
        ▼
training/stage1/lavis_loader.py     ★ CỬA DUY NHẤT
        ▼
model/lavis/, mhcac/, vision_encoders/
```

Enforce bằng `tests/test_native_independence.py`. Vi phạm → `medgemma_direct`
không khởi động nổi trên máy thiếu LAVIS, và ablation bị nhiễm.

## Role in project

```text
Stage 1 checkpoint ──(tùy mode)──► training/ ──► LoRA adapter + reports JSONL
split CSV ─────────────────────────►         ──► training/evaluation/ ──► metrics
```

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

### Files

| File | LOC | Doc | Status | Vai trò |
|---|---|---|---|---|
| `run_medgemma_qlora.py` | 527 | [📄](run_medgemma_qlora.py.doc.md) | ✅ ★ | **Entrypoint Stage 2** |
| `train_eval_figure9_llm_variants_200.py` | 1746 | [📄](train_eval_figure9_llm_variants_200.py.doc.md) | ✅ ★ | **Động cơ Stage 2** — ⚠ tên file gây hiểu nhầm |
| `pipeline_modes.py` | 196 | [📄](pipeline_modes.py.doc.md) | ✅ | Resolve kiến trúc; stdlib-only |
| `stage2_utils.py` | 298 | [📄](stage2_utils.py.doc.md) | ✅ | Helper dùng chung |
| `run_context.py` | — | [📄](run_context.py.doc.md) | ✅ | `Stage1Context` |
| `torch_io.py` | — | [📄](torch_io.py.doc.md) | ✅ | `load_torch_checkpoint` — tách ra để không kéo LAVIS |

### Subdirectories

| Thư mục | Doc | Status | Nội dung |
|---|---|---|---|
| `dataio/` | [📁](dataio/_index.md) | ✅ | `manifest.py` (CHỈ pandas), `validate_manifest.py` |
| `stage1/` | [📁](stage1/_index.md) | 🟡 | `lavis_loader.py` — cửa duy nhất sang Stage 1 |
| `medgemma/` | [📁](medgemma/_index.md) | ✅ | `soft_tokens.py`, `capabilities.py` |
| `trainer/` | [📁](trainer/_index.md) | ❓ | `checkpointing.py`, `state.py` — [D-001](../_meta/DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline) |
| `evaluation/` | [📁](evaluation/_index.md) | ✅ | 17 module evaluator |

## Main responsibilities

1. Resolve `--pipeline-mode` thành kiến trúc cụ thể.
2. Dựng record: native (từ CSV) hoặc Q-Former (qua Stage 1).
3. Fine-tune MedGemma bằng QLoRA, chọn checkpoint theo validation CE.
4. Sinh báo cáo cho test cohort, đúng một lần.
5. Chấm điểm cả Stage 1 và Stage 2 (`evaluation/`).

## Entry points

```bash
python training/run_medgemma_qlora.py [flags]                       # ★ chính
python -m training.dataio.validate_manifest --section-mode …        # kiểm tra manifest
```

⚠ `train_eval_figure9_llm_variants_200.py` có `__main__` nhưng **không nên gọi
trực tiếp** — bỏ qua resolve mode, kiểm tra leakage và ràng buộc section.

## Dependencies

| Nhóm | Phụ thuộc |
|---|---|
| Luôn | `torch`, `transformers`, `peft`, `bitsandbytes`, `pandas` |
| stdlib-only | `pipeline_modes.py`, `dataio/manifest.py` (chỉ pandas) |
| Chỉ mode Q-Former | `model/lavis/`, `mhcac/`, `vision_encoders/` — qua `stage1/lavis_loader.py` |
| Prompt | [`stage2/prompts/`](../stage2/prompts/_index.md) |
| Evaluation | chỉ `numpy` cho phần lõi metric; Grad-CAM CLI nạp Stage 1/PyTorch theo kiểu lazy import |

## Used by

`scripts/evaluate_stage1.py`, `evaluate_stage2.py`, `evaluate_explanation.py`,
`calibrate_thresholds.py` (→ `evaluation/`) · `medgemma_inference/run_pretrained_findings.py`
(→ `dataio/manifest.py`, `stage2_utils.stable_fingerprint`) ·
`model/pretrained_medgemma/output_schema.py` (→ `manifest.split_generated_report`) ·
`model/lavis/tasks/image_text_pretrain.py:223,259` (→ `evaluation/`, import trễ)

## Execution flow

Xem [CALL_GRAPH.md §3](../_meta/CALL_GRAPH.md#3-stage-2--top-down).

## Important configurations

| Nguồn | Nội dung |
|---|---|
| CLI flags | `--pipeline-mode`, `--section-mode`, `--prompt-config`, `--checkpoint-root`, `--stage1-run`, `--stage1-config`, `--train/val/test-limit`, `--no-upload`, `--gcs-output` |
| `configs/stage2_prompt_v2.yaml` | Prompt v2 (opt-in) |
| `pretraining/configs/mimic_cxr_full.yaml` | Mặc định `--stage1-config` |
| `configs/env_config.yaml` | Đường dẫn split CSV, ảnh |

## Status

```text
✅ ACTIVE
❓ UNKNOWN — trainer/, evaluation/config.py, evaluation/counterfactual.py + perturbations.py
```

## Notes

- ⚠ **Dual-import shim bắt buộc.** Mọi module ở đây phải mang:
  ```python
  try:    from stage2_utils import X          # khi chạy như script
  except ImportError: from training.stage2_utils import X   # khi chạy python -m
  ```
  Vì `run_medgemma_qlora.py:26` làm `sys.path.insert(0, dirname(__file__))`.
  **Giữ pattern này trong file mới** — bỏ đi sẽ làm một trong hai cách gọi hỏng.

- ⚠ **Tên `train_eval_figure9_llm_variants_200.py` gây hiểu nhầm nghiêm trọng.**
  Nó không vẽ figure; nó là động cơ Stage 2. Docs nội bộ ghi nó đang chờ tách nhỏ.

- **Stage 2 không có DDP.** Single process, single GPU. Hai job với
  `CUDA_VISIBLE_DEVICES=0` và `=1` là hai experiment độc lập, không chia sẻ gradient.

- **Mode Q-Former chỉ hỗ trợ `findings_only`** — yêu cầu section khác sẽ **báo lỗi
  và dừng**, không âm thầm đổi target.

## Related documentation

[PIPELINES.md → P2](../_meta/PIPELINES.md#p2--stage-2-medgemma-qlora) ·
[ARCHITECTURE.md §3](../_meta/ARCHITECTURE.md#3-stage-2--chi-tiết) ·
[DATA_FLOW.md §4](../_meta/DATA_FLOW.md#4-stage-2--csv--tensor) ·
[`stage2/prompts/_index.md`](../stage2/prompts/_index.md)

← [Về HOME](../../HOME.md)
