> Source: `training/run_medgemma_qlora.py:404-527`
> Status: ✅ ACTIVE

# `main()`

## Located in

[`run_medgemma_qlora.py`](../run_medgemma_qlora.py.doc.md)

## Purpose
Điều phối Stage 2: resolve kiến trúc → dựng record → train từng mode → upload.

## Execution flow
```text
parse_args()
   ↓
resolve_pipeline_modes(selection)     ⚠ raise nếu là mode inference ngoài
   ↓
★ IF mode Q-Former và section != findings_only:
      raise — "ReportDataset emits FINDINGS only"        (:434-440)
   ↓
IF --prompt-config: stage2.prompts.load_prompt_config()  (:410)
   ↓
modes_require_stage1(modes)?
   ✗ → build_native_records()   → dataio.manifest (CHỈ pandas)
   ✓ → build_stage1_records()   → training/stage1/lavis_loader  ★ CỬA DUY NHẤT
   ↓
tạo immutable Stage1Context(run_name, config, checkpoint, thresholds)  (:418-423)
   ↓
FOR mode in modes: train_mode(mode, records, context=stage1_context, ...)
   ↓
--no-upload? ✗ → upload_safe_run(root, adapter_dirs, gcs_output)
```

## ★ Ràng buộc section fail-closed
`:434-440` **báo lỗi và dừng** thay vì âm thầm đổi target. Comment giải thích:
`ReportDataset.text_output` không phát ra IMPRESSION, nên mode Q-Former không thể
phục vụ `findings_and_impression`.

## Stage-1 identity không còn dùng global

`Stage1Context` là frozen dataclass và sao chép `thresholds` vào
`MappingProxyType`. Context này được truyền tường minh xuyên suốt build/evaluate,
nên hai run trong cùng process không còn ghi đè identity của nhau.

## Returns
`None`. Kết quả nằm ở đĩa + GCS.

## Side effects
Ghi adapter/JSONL/meta.json · Upload GCS · Cấp phát GPU

## Error handling
| Điều kiện | Lỗi |
|---|---|
| Mode Q-Former + section sai | raise nêu lý do |
| Mode inference ngoài | `ValueError` chỉ đúng lệnh thay thế |
| Manifest thiếu cột | `assert_columns` nêu tên cột |
| Split leakage | `assert_no_leakage` |

## Tests
`tests/test_pipeline_modes.py` · `tests/test_native_independence.py`

## Modification risk
⚠ Đừng nới lỏng kiểm tra section — âm thầm đổi target làm kết quả không so sánh được.
