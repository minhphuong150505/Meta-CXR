> Source: `pretraining/retired_keys.py` (95 dòng)
> Status: ✅ ACTIVE — gọi bởi `pretraining/train.py`, `Blip2Qformer.from_config`/`load_state_dict`, `ImageTextPretrainTask.__init__`
> Last verified against source: 2026-09-29

# `pretraining/retired_keys.py`

← [pretraining](_index.md) · [D-023](../_meta/DECISIONS.md#d-023--ba-lớp-không-bao-giờ-nhị-phân-gỡ-toàn-bộ-khung-nhị-phân)

Chốt chặn cho khung nhị phân đã gỡ. Một config cũ còn khóa đã gỡ sẽ **báo lỗi to**
thay vì bị OmegaConf lặng lẽ bỏ qua.

| Tên | Vai trò |
|---|---|
| `RETIRED_MODEL_KEYS` | `(loss, lambda_gate)`, `(loss, lambda_mention_conditioned_cls)`, `(mhcac, gate_class_weights)`, `(mhcac, mention_conditioned_pos_weights)`, `(mhcac, uncertain_policy)` |
| `RETIRED_RUN_KEYS` | `uncertain_policy`, `report_study_presence`, `include_meta_labels`, `label_framing` |
| `RETIRED_STATE_PREFIXES` | `mhcac.mention_heads.`, `gate_loss_fn.`, `mention_conditioned_loss_fn.` |
| `RetiredConfigKey(ValueError)` | lỗi có thông điệp nhắc D-023 và "three-class" |
| `reject_retired_model_keys(model_cfg)` / `reject_retired_run_keys(run_cfg)` | raise nếu có khóa đã gỡ; `train.py` gọi cho config gộp **và** mọi `run.phases.*` |
| `drop_retired_state(state_dict)` | bỏ tham số head đã gỡ để checkpoint cũ vẫn load được; giữ `_metadata` |

Test: `tests/test_three_class_only.py` (cũng quét AST toàn bộ source để cấm tên cũ).
