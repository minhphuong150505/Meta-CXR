> Source: `training/run_medgemma_qlora.py:211-353`
> Status: ✅ ACTIVE

# `train_mode(mode, records, ...)`

## Located in

[`run_medgemma_qlora.py`](../run_medgemma_qlora.py.doc.md)

## Purpose
Chạy trọn một `PipelineMode`: dựng `VariantLLM`, train, generate test, lưu adapter.

## Execution flow
```text
xác định output dir theo mode.image_mode ("native" / "qformer")
   ↓
resumable_adapter(path, image_mode)?  → nạp lại thay vì train từ đầu
   ↓
fig9.VariantLLM(mode, prompt_config, ...)
   ├─ load MedGemma 4-bit NF4 + LoRA
   ├─ (mode Q-Former) SoftTokenEmbeddingWrapper + img_proj
   └─ assert_vision_tower_frozen()
   ↓
train_fine(train_records, val_records, ...)   ← chọn checkpoint theo validation CE
   ↓
generate cho test cohort — ĐÚNG MỘT LẦN
   ↓
compute_nlg / compute_sectioned_nlg
   ↓
save_adapter() + meta.json (prompt version, config hash, template hash)
```

## Parameters
`mode: PipelineMode` · `records` (train/val/test) · đường dẫn output, checkpoint,
prompt config, các limit.

## ★ `mode.image_mode` là khóa lưu trữ
`"native"` / `"qformer"` — chuỗi cũ, giữ nguyên để thư mục adapter và `meta.json`
đã tồn tại vẫn load được (xem `pipeline_modes.py:22-24`).

## Side effects
Cấp phát MedGemma trên GPU · Ghi adapter + JSONL + meta.json

## Error handling
`requires_multimodal` không thỏa → `MultimodalModelLoadError` (fail-closed) ·
Vision tower không đóng băng → raise

## Tests
`tests/test_multimodal_capability.py` · `tests/test_soft_token_injection.py`

## Modification risk
Đổi `image_mode` string → adapter cũ không nhận diện được.
