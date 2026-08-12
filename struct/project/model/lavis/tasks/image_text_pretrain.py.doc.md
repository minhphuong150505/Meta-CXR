> Source: `model/lavis/tasks/image_text_pretrain.py` (320 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `image_text_pretrain.py`

## Purpose

Task hook của Stage 1: định nghĩa `evaluation()` — chạy model trên một split, thu
logits, gọi evaluator, và lưu prediction `.npz`.

## Why it exists

`BaseTask.evaluation` gốc chỉ gom kết quả caption. Stage 1 cần thu **logits phân
loại** rồi tính chỉ số 14×3 và ghi ra `.npz` để calibrate threshold offline.

## Role in architecture

```text
RunnerBase.eval_epoch ──► ImageTextPretrainTask.evaluation
                              ├─ model(samples) → logits
                              ├─ _build_predictions
                              ├─ training.evaluation.classification_metrics  ← import TRỄ
                              └─ _save_predictions → .npz
```

## Status

```text
✅ ACTIVE
```

## Used in

Validation ✅ · Test ✅ · Training ❌ (train_step ở `BaseTask`)

## Entry point

Không. `@registry.register_task(...)`, chọn qua `run.task: image_text_pretrain_eval`.

## Inputs

`model`, `data_loader`, `cuda_enabled`; và `split_name`/`epoch` qua
`set_evaluation_context` (`:48`).

## Outputs

Dict metric cho `RunnerBase.validate` · file `.npz` khi `run.save_predictions: true`

## Important configuration

| Key | Ảnh hưởng |
|---|---|
| `run.task` | Phải là `image_text_pretrain_eval` để dùng class này |
| `run.save_predictions` | Có ghi `.npz` không |
| `run.save_text_predictions` | Có lưu text sinh ra không (prod: `false`) |
| `run.uncertain_policy` | prod: `ignore_uncertain` |
| `run.include_meta_labels` | prod: `false` |

## Main classes

`ImageTextPretrainTask(BaseTask)` (`:20`)

## Main methods

| Method | Vai trò |
|---|---|
| `evaluation` (`:53`) | [📄](image_text_pretrain.py.methods/ImageTextPretrainTask/evaluation.md) ★ Vòng eval, thu logits, gọi evaluator |
| `set_evaluation_context` (`:48`) | Đặt `split_name`/`epoch` để đặt tên file |
| `_build_predictions` (`:257`) | Ghép chunk logits/label/key thành `ClassificationPredictions` |
| `_save_predictions` (`:286`) | Ghi `.npz` |
| `setup_task` (`:44`), `__init__` (`:37`) | Dựng từ cfg |

## ⚠ Import ngược chiều, cố ý đặt trong hàm

```python
from training.evaluation.classification_metrics import (...)   # :223
from training.evaluation.schemas import (...)                  # :259
```

Đây là phụ thuộc **`model/lavis/` → `training/`** — ngược với hướng thông thường.
Nó nằm **trong hàm**, không ở module scope, để `model/lavis/` vẫn import được khi
`training/evaluation/` chưa sẵn sàng.

Đừng nâng hai import này lên module scope.

## Calls

`model(samples)` → `Blip2Qformer.forward` · `training.evaluation.classification_metrics.*` ·
`training.evaluation.schemas.ClassificationPredictions`

## Called by

`RunnerBase.eval_epoch` (`:806`) → `validate` và `evaluate`

## Data flow

```text
data_loader ─► model(samples) ─► BlipOutput.classification_logits [B,14,3]
                                          │  gom theo chunk
                                          ▼
                          _build_predictions(logits, labels, keys)
                                          │
                     ┌────────────────────┴─────────────────┐
                     ▼                                      ▼
        classification_metrics.evaluate_classification   _save_predictions
                     │                                      │
              dict metric → RunnerBase                   .npz trên đĩa
```

## Side effects

Ghi `.npz` vào `<output_dir>/<run_name>/result/` · Đặt model sang `.eval()` ·
Cấp phát bộ nhớ giữ chunk logits

⚠ `.npz` là **dẫn xuất từ dữ liệu bệnh nhân** — `.gitignore` chặn `*.npz`.

## Error / edge cases

⚠ Hành vi khi `selection_metric` không có trong dict trả về — cần runtime verification.

## Related tests

`tests/test_stage1_eval_hook.py` (183 dòng) ⚠ cần `model.lavis`, fail trên CPU thuần

## Related documentation

[`runner_base.py`](../runners/runner_base.py.doc.md) ·
[`training/evaluation/_index.md`](../../../training/evaluation/_index.md) ·
[PIPELINES.md → P4](../../../_meta/PIPELINES.md#p4--evaluation-stage-1)

## Developer notes

1. `.npz` là thứ cho phép **calibrate threshold không cần GPU**. Đừng tắt
   `save_predictions` trên run production.
2. Hai import trễ ở `:223`/`:259` là **cố ý**. Giữ nguyên vị trí.
3. Class này chỉ thu logits phân loại. Metric NLG của Stage 1 (nếu cần) không đi
   qua đây.

## Source relationships

- **Parent:** [`model/lavis/_index.md`](../_index.md)
- **Related:** [`runner_base.py`](../runners/runner_base.py.doc.md) · [`blip2_qformer.py`](../models/blip2_models/blip2_qformer.py.doc.md)

← [HOME](../../../../HOME.md)
