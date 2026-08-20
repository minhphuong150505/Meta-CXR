> Source: `training/evaluation/schemas.py` (359 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-20

# `schemas.py`

## Purpose

Schema có kiểu cho dữ liệu evaluation: `ClassificationPredictions` (đọc `.npz`) và
`GenerationRecord` (đọc `.jsonl`). Cộng `_digest` để tạo `sample_key` **có salt**.

## Why it exists

Evaluator đọc file do inference sinh ra. Nếu không có schema, một `.npz` sai cấu
trúc sẽ tạo ra metric sai mà không ai biết. `SchemaError` fail sớm và rõ.

## Status

```text
✅ ACTIVE
```

## Main items

| Tên | Dòng | Vai trò |
|---|---|---|
| `ClassificationPredictions` | 64 | ★ logits, labels, keys, split |
| `GenerationRecord` | 246 | ★ prediction + reference |
| `load_generation_records(path)` | 270 | Đọc `.jsonl` |
| `save_generation_records(...)` | 326 | Ghi `.jsonl` |
| `build_sample_keys(...)` | 228 | ★ Khóa mẫu **có salt** |
| `_digest(value, salt)` | 58 | Băm |
| `SchemaError` | 54 | |
| `CLASS_NAMES` | — | `negative`/`positive`/`uncertain` |

## Mã hóa nhãn

`0 = negative, 1 = positive, 2 = uncertain` — docstring `:19` ghi nó **khớp
`ReportDataset`**. Hai nơi phải đồng bộ.

`MISSING`, `POSITIVE`, `UNCERTAIN` được `uncertain_policy.py:33` import lại.

## ⚠ `sample_key` có salt

Không phải `study_id`. Nhờ vậy file kết quả **không mang định danh bệnh nhân** mà
vẫn ghép được các bản ghi với nhau.

## `mention_probabilities` (thêm 2026-08-20)

Field optional `[N, P]`: xác suất của **mention gate** — *"báo cáo có nhắc tới
finding này không?"*. Chỉ có ở run mà eval hook thu thập gate
(`model/lavis/tasks/image_text_pretrain.py`).

Đây là thừa số mà [`label_framing.presence_scores`](label_framing.py.doc.md) nhân
vào `q_positive` để ra `P(present)`. Không có nó thì score `marginal_presence`
**raise**, không rơi ngầm về `conditional_positive`.

Được validate shape trong `__post_init__`, ghi/đọc qua `save()`/`load()`. File
`.npz` cũ không có key này → `load()` trả `None`, không lỗi.

## Calls / Called by

Gọi: `numpy`, `hashlib`, `json`.
Được gọi: `scripts/evaluate_stage1.py:46`, `evaluate_stage2.py:53`,
`calibrate_thresholds.py:30`; `tasks/image_text_pretrain.py:259`;
`classification_metrics.py:33`, `threshold_calibration.py:40`, `baselines.py:29`,
`uncertain_policy.py:33`; nhiều test.

## Side effects

Đọc/ghi file.

## Error / edge cases

`SchemaError` khi shape/khóa sai.

## Related tests

`tests/test_classification_metrics.py:29` · `tests/test_evaluation_integration.py:28` ·
`tests/test_generation_metrics.py:39`

## Developer notes

⚠ `.npz` và `.jsonl` là **dẫn xuất từ dữ liệu bệnh nhân**. `.gitignore` chặn cả hai.
Đừng commit, đừng đưa vào `struct/`.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md) · [`scripts/_index.md`](../../scripts/_index.md)

← [HOME](../../../HOME.md)
