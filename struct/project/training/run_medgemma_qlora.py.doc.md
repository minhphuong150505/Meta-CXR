> Source: `training/run_medgemma_qlora.py` (527 dòng)
> Status: ✅ ACTIVE — ★ ENTRYPOINT Stage 2
> Last verified against source: 2026-08-12

# `training/run_medgemma_qlora.py`

## Purpose

Entrypoint Stage 2. Resolve kiến trúc từ `--pipeline-mode`, dựng record, gọi động
cơ huấn luyện, upload artifact.

## Why it exists

Tách **quyết định kiến trúc** (mode nào, cần Stage 1 không, section nào hợp lệ)
khỏi **cơ chế huấn luyện** (nằm ở `train_eval_figure9_llm_variants_200.py`). Nhờ
vậy mode resolution testable trên CPU không cần torch.

## Role in architecture

```text
CLI ──► run_medgemma_qlora.main()
          ├─ resolve_pipeline_modes()          stdlib
          ├─ build_native_records()   HOẶC  build_stage1_records()
          └─ train_mode() ──► fig9.VariantLLM
```

## Status

```text
✅ ACTIVE — PRIMARY
```

## Entry point

```bash
CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py \
    --pipeline-mode medgemma_direct --section-mode findings_and_impression \
    --output-dir training/outputs/run1 --no-upload
```

Chạy **trực tiếp bằng python**, không torchrun — Stage 2 single-process.

## Inputs

| Flag | Mặc định | Ghi chú |
|---|---|---|
| `--pipeline-mode` | `medgemma_direct` | 5 mode + `both_for_ablation` |
| `--image-mode` | — | ⚠ alias deprecated |
| `--section-mode` | `findings_and_impression` | Q-Former chỉ `findings_only` |
| `--prompt-config` | — | Bỏ → prompt legacy |
| `--checkpoint-root` | `checkpoints` | |
| `--stage1-run` | `mimic_cxr_full_blip2` | |
| `--stage1-config` | `pretraining/configs/mimic_cxr_full.yaml` | |
| `--train/val/test-limit` | — | Smoke |
| `--output-dir`, `--gcs-output`, `--no-upload` | | |

## Outputs

`<output-dir>/<mode>/` — adapter LoRA, `adapter_config.json`, `trainer_state.pt`,
`img_proj*` (mode Q-Former); `meta.json`; `generated_reports.jsonl`.

## Important imports

```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # :26
from dataio.manifest import (...)          # :27  ← dual-import shim
from pipeline_modes import (...)           # :33
from run_context import Stage1Context      # :42
import train_eval_figure9_llm_variants_200 as fig9   # :49  ← động cơ
```

⚠ Comment `:44-48` ghi rõ: import `fig9` kéo theo torch/transformers/peft (mà
`medgemma_direct` cần dù sao) **nhưng không còn kéo LAVIS**. LAVIS bị giam trong
`training/stage1/lavis_loader.py`, chỉ gọi từ trong `build_stage1_records()`.
Enforce bởi `tests/test_native_independence.py`.

## Main functions

| Hàm | Dòng | Doc | Vai trò |
|---|---|---|---|
| `main` | 404 | [📄](run_medgemma_qlora.py.methods/main.md) | ★ |
| `train_mode` | 211 | [📄](run_medgemma_qlora.py.methods/train_mode.md) | Chạy một mode |
| `build_native_records` | 364 | [📄](run_medgemma_qlora.py.methods/build_native_records.md) | CSV → record, không LAVIS |
| `parse_args` | 52 | — | CLI |
| `resumable_adapter` | 180 | — | Đủ weight + projector + `adapter_config.json` + `trainer_state.pt` |
| `upload_safe_run` | 186 | — | Upload chỉ artifact an toàn |
| `deterministic_subset` | 172 | — | Lấy mẫu con tái lập |
| `load_split_frame` | 354 | — | Đọc CSV split |

## Execution flow

```text
main()
 ├─ parse_args()
 ├─ resolve_pipeline_modes(sel)  → [PipelineMode]     ⚠ raise nếu là mode inference ngoài
 ├─ IF Q-Former mode + section != findings_only  → LỖI VÀ DỪNG (:434-440)
 ├─ IF --prompt-config: load_prompt_config()          (:410)
 ├─ modes_require_stage1?
 │     no  → build_native_records()   → dataio.manifest (CHỈ pandas)
 │     yes → build_stage1_records()   → training/stage1/lavis_loader  ★ cửa duy nhất
 ├─ FOR mode in modes:  train_mode(mode, ...)  → fig9.VariantLLM
 └─ upload_safe_run() nếu không --no-upload
```

## Calls / Called by

Gọi: `pipeline_modes`, `dataio.manifest`, `run_context.Stage1Context`,
`stage2.prompts.load_prompt_config`, `fig9.*`, `training/stage1/lavis_loader` (có điều kiện).
Được gọi: người dùng (trực tiếp trên máy train).

## Side effects

Ghi adapter + JSONL + `meta.json` · Upload GCS (trừ `--no-upload`) · Cấp phát GPU ·
`sys.path.insert` (`:26`) — **thay đổi import path toàn process**

## Error / edge cases

| Tình huống | Hành vi |
|---|---|
| Mode Q-Former + `findings_and_impression` | **Raise**, nêu lý do (`ReportDataset.text_output` không có IMPRESSION) |
| `--pipeline-mode pretrained_medgemma_*` | **Raise**, chỉ đúng lệnh `python -m medgemma_inference.run_pretrained_findings` |
| Mode lạ | `ValueError` liệt kê `CHOICES` |
| Manifest thiếu cột impression | `assert_columns` raise nêu tên cột |
| Split leakage | `assert_no_leakage` raise |

## Related tests

`tests/test_pipeline_modes.py` · `tests/test_native_independence.py` (★) ·
`tests/test_manifest.py` · `tests/test_section_metrics.py`

## Developer notes

1. ⚠ **Đừng thêm import LAVIS ở module scope.** Test sẽ fail và nêu đúng cách sửa.
2. **`sys.path.insert` ở `:26`** khiến `training/` thành import root — đó là lý do
   mọi module trong `training/` cần dual-import shim.
3. `both_for_ablation` chạy `medgemma_direct` **trước**, để crash ở ablation vẫn
   để lại kết quả chính trên đĩa.
4. **Không có DDP.** Đừng thêm `device_map` rộng để giả làm multi-GPU.

## Source relationships

- **Parent:** [`training/_index.md`](_index.md)
- **Methods:** [`run_medgemma_qlora.py.methods/`](run_medgemma_qlora.py.methods/)
- **Related:** [`train_eval_figure9…`](train_eval_figure9_llm_variants_200.py.doc.md) · [`pipeline_modes.py`](pipeline_modes.py.doc.md)

← [HOME](../../HOME.md)
