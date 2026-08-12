> Source: `configs/experiments/pretrained_medgemma_findings_first.yaml` (config)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `configs/experiments/pretrained_medgemma_findings_first.yaml`

## Purpose
Cấu hình pipeline [P8](../../_meta/PIPELINES.md#p8--external-medgemma-inference-baseline)
— inference trên checkpoint MedGemma bên thứ ba.

## Consumer
`medgemma_inference/config.py:load_config` → `parse_config` → `validate_config`

⚠ Validator này chạy **chỉ** trên `configs/experiments/*.yaml`. Config Stage 1 là
namespace riêng.

## Các khối
| Khối | Dataclass | Nội dung |
|---|---|---|
| findings model | `FindingsModelConfig` (`:66`) | `erjui/medgemma-4b-srrg-findings`, revision |
| impression model | `ImpressionModelConfig` (`:79`) | ⛔ Phase 2, bị guard chặn |
| `runtime` | `RuntimeConfig` (`:88`) | ★ `hourly_cost_usd`, `budget_limit_usd` — cả hai phải ≥ 0 (`:230`) |
| `evaluation` | `EvaluationConfig` (`:98`) | |
| `privacy` | `PrivacyConfig` (`:106`) | |

## Error
Key fine-tuning sót lại → `ObsoleteFineTuningConfigError` nêu **đường dẫn key**
(`_find_obsolete_keys:122`).

## Related documentation
[`medgemma_inference/_index.md`](../../medgemma_inference/_index.md) ·
[D-005](../../_meta/DECISIONS.md#d-005--track-inference-checkpoint-ngoài-là-baseline-chính-thức)

## Developer notes
`budget_limit_usd` được in ra banner **trước khi chạy** — cố ý, để thấy trần chi
phí trước khi bấm enter.

← [`_index.md`](../_index.md) · [HOME](../../../HOME.md)
