> Source: `configs/stage2_prompt_v2.yaml` (config)
> Status: ✅ ACTIVE — OPT-IN
> Last verified against source: 2026-08-12

# `configs/stage2_prompt_v2.yaml`

## Purpose
Cấu hình Prompt v2 cho Stage 2.

## ⚠ OPT-IN
Chỉ có tác dụng khi truyền `--prompt-config configs/stage2_prompt_v2.yaml`.
**Bỏ flag → code dùng prompt legacy.** Đây là chỗ dễ bỏ sót khi so sánh kết quả:
hai run có thể khác nhau chỉ vì một run quên flag này.

## Consumer
`training/run_medgemma_qlora.py:410` → `stage2.prompts.load_prompt_config` →
`stage2/prompts/validation.py:197`

## Các nhóm key
| Nhóm | Được validate bởi | Ảnh hưởng |
|---|---|---|
| `visual_mode` | `schemas.VisualMode` | 5 mode; quyết định có thấy Stage-1 label không |
| `normal_policy` | `policies.NormalPolicy` | Khi không có positive/uncertain |
| `negative_policy` + số lượng tối đa | `policies.NegativePolicy` | Bao nhiêu negative vào prompt |
| `uncertainty_policy` | `policies.UncertaintyPolicy` | Ngôn ngữ thận trọng |
| `temporal_target_policy` | `policies.TemporalTargetPolicy` | ⚠ mặc định `keep` |
| ràng buộc số câu | `templates.sentence_constraint` | |

## Vân tay
Mỗi prompt sinh ra kèm `prompt_version`, `config_hash`, `template_hash` ghi vào
metadata artifact — truy được kết quả về đúng cấu hình.

## Related documentation
[`stage2/prompts/_index.md`](../stage2/prompts/_index.md) ·
[`prompt_ablation/_index.md`](prompt_ablation/_index.md) ·
[ARCHITECTURE.md §4](../_meta/ARCHITECTURE.md#4-prompt-v2)

## Developer notes
Thêm key mới **phải** thêm validate ở `stage2/prompts/validation.py`, nếu không
typo bị bỏ qua âm thầm.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
