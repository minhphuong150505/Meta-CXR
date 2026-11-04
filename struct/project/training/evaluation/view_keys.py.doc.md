# `training/evaluation/view_keys.py`

> Thư mục: [`training/evaluation/`](_index.md) · Thêm 2026-11-04 · 🔬 dùng cho nhánh residual-lateral
> (`docs/handoff/PLAN-2026-11-04-residual-lateral.md`) · numpy + stdlib

## Purpose
Ghép một lượt chạy trên **ảnh lateral** với dự đoán **frontal** theo key. Eval hook Stage 1 ghi
`sample_keys = build_sample_keys([str(image_id)])`, trong đó `image_id` là **vị trí dòng annotation
của ảnh anchor**. Lượt có anchor lateral vì vậy đánh key theo dòng lateral, không ghép được với npz
frontal. Module dựng lại, từ chính `dataset.annotation` theo thứ tự của nó, cho mỗi study:
`frontal_key` (anchor mặc định `[PA, AP, lateral]`), `lateral_key` (ảnh lateral mà
`max_aux_views=1` ghép, `''` nếu không phải PL), `group` và `num_views`.

Nhóm (giống evidence package 2026-10-27; val PL 797 / F1 930 / other 81):
- `PL`: anchor PA/AP và aux có view id 2 (LATERAL/LL);
- `F1`: anchor PA/AP, không aux;
- `other`: còn lại (PA+AP, chỉ lateral, view không rõ).

## API
| Tên | Vai trò |
|---|---|
| `build_view_map(rows)` | `ViewMap` cho các dòng annotation (`subject_id`, `study_id`, `ViewPosition`) |
| `ViewMap.save / load` | npz chỉ chứa key, nhóm, số view — **không ghi vị trí dòng** |
| `lateral_pass_positions(map, lateral_studies, order)` | Vị trí các study PL trong dataset anchor-lateral; raise nếu thứ tự study lệch hoặc anchor lateral không trùng lateral của PL |
| `check_num_views(map, num_views, keys)` | Kiểm `1 + len(aux) == num_views` từng dòng (npz có `num_views`) |
| `align(keys, target)` | Chỉ số `idx` để `keys[idx] == target`; raise nếu thiếu hoặc trùng key |
| `study_group(study)` | Nhóm của một mục `build_study_index` |

## Caller
- [`scripts/encoder_probe_features.py`](../../scripts/encoder_probe_features.py.doc.md) `--views lateral`
- [`scripts/lateral_expert_fit.py`](../../scripts/lateral_expert_fit.py.doc.md) (`align`)
- [`scripts/residual_stack.py`](../../scripts/residual_stack.py.doc.md) (`ViewMap`, `align`, `PL`)

## Callee
`model/lavis/data/mimic_cxr_utils.build_study_index`, `schemas.build_sample_keys`.

## Bẫy
- `rows` phải là `dataset.annotation` theo đúng thứ tự của nó: key băm VỊ TRÍ dòng.
- Map vẫn là dẫn xuất MIMIC-CXR (digest có salt): chỉ để trên máy train.

Test: `tests/test_residual_lateral.py`.
