> Source: `medgemma_inference/config.py` (241 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `medgemma_inference/config.py`

## Purpose
Schema và validate cho `configs/experiments/*.yaml`.

## ⚠ Phạm vi hẹp — docstring `:3` ghi rõ
Validator này chạy **chỉ** trên `configs/experiments/*.yaml`. Config Stage 1 dưới
`pretraining/configs/` là **namespace riêng, không bao giờ bị parse ở đây** — các
key `learning_rate`, `optimizer`, `warmup_steps` ở đó vẫn hoàn toàn hợp lệ.

Ghi chú này tồn tại vì `_find_obsolete_keys` sẽ báo lỗi nếu thấy key fine-tuning,
và người đọc dễ tưởng nó áp cho toàn repo.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng |
|---|---|
| `load_config(path)` | 234 |
| `parse_config(raw, source)` | 153 |
| `validate_config(config, source)` | 199 |
| `ExperimentConfig` | 113 |
| `FindingsModelConfig` / `ImpressionModelConfig` | 66, 79 |
| `RuntimeConfig` | 88 |
| `EvaluationConfig` / `PrivacyConfig` | 98, 106 |
| `_find_obsolete_keys(node, path)` | 122 | ★ Bắt key fine-tuning sót lại |
| `_subset(cls, data, source)` | 137 |
| `ConfigError` / `ObsoleteFineTuningConfigError` | 57, 61 |

`:230` kiểm `runtime.hourly_cost_usd >= 0` và `runtime.budget_limit_usd >= 0`.

## Calls / Called by
Gọi: `yaml`, `dataclasses`.
Được gọi: `runner.py`, `run_pretrained_findings.py`; `tests/test_pretrained_findings.py`.

## Side effects
Đọc file.

## Error / edge cases
Key fine-tuning trong config inference → `ObsoleteFineTuningConfigError` **nêu
đường dẫn key**.

## Related tests
`tests/test_pretrained_findings.py`

## Developer notes
`_find_obsolete_keys` đi đệ quy và trả **đường dẫn đầy đủ** của key sai — quan
trọng khi YAML lồng nhiều tầng.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
