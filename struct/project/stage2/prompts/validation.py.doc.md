> Source: `stage2/prompts/validation.py` (208 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `stage2/prompts/validation.py`

## Purpose
`PromptConfig` + nạp/validate YAML prompt.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `PromptConfig` | 41 | ★ Cấu hình prompt đã validate |
| `config_from_mapping(block)` | 125 | dict → `PromptConfig` |
| `load_prompt_config(path)` | 197 | ★ YAML → `PromptConfig` |
| `_coerce(enum_cls, value, field_name)` | 30 | ★ Chuỗi → Enum, **nêu tên trường khi sai** |
| `PromptConfigError` | 26 | |

`_coerce` nêu **tên trường** trong thông điệp lỗi — một YAML sai một chữ sẽ chỉ ra
đúng dòng nào, thay vì `ValueError: invalid value`.

## Calls / Called by
Gọi: `yaml`, `schemas`, `policies`, `ontology`.
Được gọi: `run_medgemma_qlora.py:410`; `fig9:146`; các script prompt;
`tests/test_stage2_prompts.py:403`.

## Side effects
Đọc file YAML.

## Error / edge cases
`PromptConfigError` với tên trường sai · Enum value lạ → nêu giá trị hợp lệ

## Related tests
`tests/test_stage2_prompts.py:401-403` — `configs/stage2_prompt_v2.yaml` parse được

## Developer notes
`configs/stage2_prompt_v2.yaml` và 9 file `configs/prompt_ablation/P*.yaml` đều
phải qua đây. Thêm trường mới nhớ thêm validate, nếu không typo sẽ bị bỏ qua âm thầm.

## Source relationships

- **Parent:** [`_index.md`](_index.md)

← [HOME](../../../HOME.md)
