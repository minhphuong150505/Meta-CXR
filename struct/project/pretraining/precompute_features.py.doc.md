> Source: `pretraining/precompute_features.py` (181 dòng)
> Status: 🟡 CONDITIONAL
> Last verified against source: 2026-08-12

# `pretraining/precompute_features.py`

## Purpose
Tính trước đầu ra encoder **đóng băng** để Stage 1 khỏi chạy lại chúng mỗi epoch.

## Why it exists
Ba encoder đóng băng chiếm phần lớn thời gian forward mà **luôn cho cùng kết quả**
với cùng ảnh. Cache chúng một lần đổi lấy tốc độ mỗi epoch sau đó.

## Entry point
```bash
python -m pretraining.precompute_features \
    --cfg-path pretraining/configs/mimic_cxr_full_l4.yaml \
    --options model.encoders.biovil=true model.encoders.pubmedclip=true
```

## Outputs
```text
<output_dir>/pubmedclip/...     # raw ViT patches, P × 768
<output_dir>/biovil/...
<output_dir>/swin/...
```
Docstring `:15` ghi rõ đây là **raw patch, TRƯỚC `ln_vision` và trước projection**.

## ★ Cache chỉ thay encoder, không thay training
`Blip2Qformer._encode_image_streams` dùng `cached[name]` thay cho forward encoder,
nhưng **các projection có thể train vẫn chạy** (comment `blip2_qformer.py:473-475`).
Training giống hệt bản không cache.

## Important configuration
| Key | Vai trò |
|---|---|
| `run.feature_cache_dir` | Bật cache; không đặt → không dùng |
| `model.encoders.*` | Encoder nào được cache (`:142`) |
| `model.data.study_sampling` | ⚠ đặt `false` khi dựng cache |

## ⚠ Hai bẫy
1. **Dựng cache với `study_sampling=false`**, nếu không auxiliary view vắng mặt.
   `ReportDataset._row_visual:654` raise `KeyError` nêu đúng DICOM thiếu **và** gợi
   ý đúng cách sửa.
2. Docstring `:31` ghi: mount volume rồi truyền **đường dẫn local đã mount** làm
   `run.feature_cache_dir`.

## Calls / Called by
Gọi: `model.lavis` (Config, model), `Blip2Qformer` encoder, `numpy`.
Được gọi: người dùng. `tests/test_inference_only_invariants.py:173` liệt kê file này.

## Side effects
⚠ Cấp phát GPU · Ghi **nhiều GB** feature · `:` có `.cuda()` trực tiếp
([S12 trong docs cũ](../_meta/LEGACY_AND_OPTIONAL.md))

## Error / edge cases
DICOM thiếu trong cache lúc train → `KeyError` nêu tên · Cache dựng với encoder
khác config train → ⚠ hành vi cần runtime verification

## Related tests
`tests/test_inference_only_invariants.py`

## Developer notes
⚠ **Feature cache là dẫn xuất trực tiếp từ ảnh MIMIC** → chịu cùng lệnh cấm
redistribute. `.gitignore` chặn `features/`, `feature_cache/`, `*.npy`, `*.npz`.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
