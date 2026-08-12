> Source: `model/pretrained_medgemma/findings_loader.py` (197 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `model/pretrained_medgemma/findings_loader.py`

## Purpose
Nạp checkpoint `erjui/medgemma-4b-srrg-findings` và **fail-closed** nếu nó không
thật sự đa phương thức.

## ⚠ Provenance
Docstring `:3`: checkpoint do **bên thứ ba (erjui)** fine-tune từ
`google/medgemma-4b-it` trên csrrg_ift (MIMIC-CXR + CheXpert+). *"This project does
not train it and does not claim it."*

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `PretrainedFindingsLoader` | 81 | [📄 class/load flow](findings_loader.py.methods/PretrainedFindingsLoader/_index.md) ★ |
| `FindingsModelBundle` | 33 | model + processor + metadata |
| `_assert_has_image_processor(processor, model_id)` | 56 | ★ |
| `_assert_has_vision_tower(model, model_id)` | 66 | ★ |

Hai `_assert_*` raise `NotMultimodalError` — **không hạ cấp sang text-only**. Lý do
ở `errors.py`: báo cáo từ model text-only trông hợp lý nhưng chưa từng thấy pixel.

## Calls / Called by
Gọi: `transformers`, `runtime.device.plan_device` (`:24`), `.errors`.
Được gọi: `medgemma_inference/runner.py` (qua `_build_reporter`).

## Side effects
Tải checkpoint/cache (mạng ở lần đầu) · Cấp phát device theo `DevicePlan`

## Error / edge cases
`FindingsModelLoadError`, `NotMultimodalError` · `QuantizationUnavailable` từ
`check_4bit_available` — raise **trước** khi tải

## Related tests
`tests/test_pretrained_findings.py`

## Developer notes
`HF_HOME` nên trỏ ra ngoài working tree; `.gitignore` có `hf_cache/`, `**/models--*/`
phòng trường hợp trỏ vào trong.

← [`_index.md`](_index.md) · [HOME](../../../HOME.md)
