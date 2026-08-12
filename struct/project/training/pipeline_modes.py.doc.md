> Source: `training/pipeline_modes.py` (196 dòng)
> Status: ✅ ACTIVE — stdlib-only
> Last verified against source: 2026-08-12

# `training/pipeline_modes.py`

## Purpose

Đặt tên tường minh cho từng **kiến trúc** Stage 2 và resolve lựa chọn CLI thành
danh sách mode.

## Why it exists

Cờ cũ `--image-mode {qformer,native}` mô tả **chi tiết cài đặt**, không mô tả kiến
trúc — khiến hybrid Q-Former dễ bị gọi nhầm là "native MedGemma" trong báo cáo.
Docstring `:3-8` nói thẳng điều đó.

Module **cố ý stdlib-only** để resolve mode testable trên máy CPU không có torch,
transformers hay LAVIS.

## Status

```text
✅ ACTIVE
```

## Main types

`PipelineMode` — dataclass **frozen**: `name`, `image_mode`, `requires_stage1`,
`uses_mhcac_prompt`, `description`, `requires_multimodal=True`.

⚠ `image_mode` **cố ý giữ chuỗi cũ** `"native"`/`"qformer"` để thư mục adapter,
`meta.json` và `manifest.json` đã tồn tại vẫn load được.

## Sáu mode

| Hằng | Stage 1 | `image_mode` | Ghi chú |
|---|---|---|---|
| `MEDGEMMA_DIRECT` | ❌ | native | **Mặc định** |
| `META_CXR_QFORMER` | ✅ | qformer | Hybrid — *"Not native MedGemma"* |
| `META_CXR_QFORMER_WITH_MHCAC_PROMPT` | ✅ | qformer | + text P/N/U |
| `TEXT_ONLY_LANGUAGE_PRIOR_ABLATION` | ❌ | text_only | ⚠ **`requires_multimodal=False`** — mode duy nhất |
| `PRETRAINED_MEDGEMMA_FINDINGS_FIRST` | ❌ | native | Inference ngoài; **không** chạy qua CLI này |
| `PRETRAINED_MEDGEMMA_IMPRESSION_PHASE2` | ❌ | native | ⛔ khai báo nhưng bị guard chặn |

## Hai cơ chế phòng vệ

### `EXTERNAL_INFERENCE_MODES` — không cho inference ngoài lọt vào fine-tuning

```python
if selection in EXTERNAL_INFERENCE_MODES:
    raise ValueError(f"{selection!r} is an external-checkpoint inference mode ...")
```

Thông điệp lỗi **nêu đúng lệnh cần dùng**:
`python -m medgemma_inference.run_pretrained_findings --config …`

### `requires_multimodal` — không có đường âm thầm thành text-only

Docstring `:190-196` ghi rõ: `text_only_language_prior_ablation` phải được **gọi
tên tường minh**; `both_for_ablation` **không** bao gồm nó; nó **không** phải mặc định.

## Main functions

| Hàm | Dòng | Vai trò |
|---|---|---|
| `resolve_pipeline_modes(selection)` | 170 | ★ Chuỗi → `list[PipelineMode]` |
| `requires_stage1(modes)` | — | Bất kỳ mode nào cần Stage 1 |
| `requires_multimodal(mode)` | — | Model có phải nhận pixel thật |

`both_for_ablation` → `[MEDGEMMA_DIRECT, META_CXR_QFORMER]` — **thứ tự có chủ đích**:
pipeline chính chạy trước, nên crash ở ablation vẫn để lại kết quả chính.

`LEGACY_IMAGE_MODE_ALIASES = {"native": …, "qformer": …, "both": …}` — cờ cũ vẫn chạy.

## Calls / Called by

Gọi: chỉ `dataclasses`.
Được gọi: `run_medgemma_qlora.py:33`; `tests/test_pipeline_modes.py`,
`tests/test_multimodal_capability.py:31`.

## Side effects

Không. Thuần dữ liệu.

## Error / edge cases

Mode ngoài → `ValueError` chỉ lệnh đúng · Mode lạ → `ValueError` liệt kê `CHOICES`

## Related tests

`tests/test_pipeline_modes.py` — phân biệt `meta_cxr_qformer` vs `..._with_mhcac_prompt`

## Developer notes

1. **Đừng đổi `image_mode`** — nó là khóa lưu trữ, đổi làm adapter cũ không load được.
2. Thêm mode mới: thêm hằng, thêm vào `PIPELINE_MODES`, và **cân nhắc** có nên vào
   `EXTERNAL_INFERENCE_MODES` không.
3. Giữ module stdlib-only. Thêm `import torch` ở đây phá test CPU.

← [training/](_index.md) · [HOME](../../HOME.md)
