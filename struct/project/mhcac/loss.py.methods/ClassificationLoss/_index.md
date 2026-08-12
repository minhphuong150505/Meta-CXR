> Source: `mhcac/loss.py:35-122`
> Status: ✅ ACTIVE

# `class ClassificationLoss(nn.Module)`

## Located in

[`loss.py`](../../loss.py.doc.md)

## Responsibility
Cross-entropy 3 lớp cho 14 bệnh lý, có class weight, label smoothing, uncertain
policy, và **`sample_mask`**.

## Constructor (`:43`)
| Tham số | Prod | Vai trò |
|---|---|---|
| `class_weights` | 14×3 sqrt inverse-freq, cap 10 | `[]` → tắt (ablation) |
| `num_abnormalities` | 14 | |
| `label_smoothing` | 0.05 | |
| `uncertain_policy` | `ignore_uncertain` | |

`class_weights` mặc định (`blip2_qformer.py:357`) = `sqrt(prevalence_negative /
prevalence_class)`, cap ở 10, tính từ **cohort train theo study** (không phải theo ảnh).

## `forward(logits, true_labels, sample_mask=None)` (`:83`)
| Tham số | Shape |
|---|---|
| `logits` | `[B,14,3]` |
| `true_labels` | `[B,14]` long — `0` neg, `1` pos, `-1` uncertain |
| `sample_mask` | `[B]` bool |

Trả scalar.

## ★ `sample_mask` là hợp đồng của cả hệ loss
Dòng không có nhãn CheXpert đóng góp **đúng 0** — không phải "một giá trị nhỏ",
không phải "được điền giá trị mặc định". Nhờ vậy 4.851 dòng train thiếu nhãn
(~1,33%) vẫn train được phần sinh báo cáo mà không làm nhiễu loss phân loại.

## Callers
`Blip2Qformer.forward:913` (student) · `:933` (teacher, `sample_mask=teacher_mask`)

## Tests
`tests/test_stage1_objectives.py`

## Modification risk
Bỏ `sample_mask` → dòng thiếu nhãn đóng góp nhiễu vào gradient, và không có tín
hiệu nào báo.
