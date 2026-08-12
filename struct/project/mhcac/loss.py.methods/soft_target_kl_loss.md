> Source: `mhcac/loss.py:123-214`
> Status: ✅ ACTIVE

# `soft_target_kl_loss(student_logits, teacher_logits, sample_mask, temperature)`

## Located in

[`loss.py`](../loss.py.doc.md)

## Purpose
Chưng cất kiến thức teacher → student bằng KL divergence trên phân phối đã làm mềm.

## Signature
```python
def soft_target_kl_loss(student_logits, teacher_logits, sample_mask=None,
                        temperature=2.0) -> Tensor
```

## Parameters
| Tham số | Shape / Giá trị |
|---|---|
| `student_logits` | `[B,14,3]` — nhánh chỉ-ảnh |
| `teacher_logits` | `[B,14,3]` — nhánh ảnh+text, **sẽ bị detach** |
| `sample_mask` | `[B]` bool = `teacher_mask` |
| `temperature` | `mhcac.distill_temperature`, prod **2.0** |

## ★ Teacher bị detach
Gradient **không** chảy ngược vào teacher. Teacher là nguồn tri thức tĩnh trong
mỗi bước; student học theo nó. Bỏ `detach` = teacher học ngược từ student, phá vỡ
toàn bộ ý nghĩa distillation.

## ★ Vì sao `temperature=2.0`
Làm mềm phân phối để student học được **thứ tự tương đối giữa các lớp**, không chỉ
lớp thắng. Với 3 lớp P/N/U, thông tin "negative nhưng gần uncertain" là tín hiệu
thật và sẽ mất nếu temperature = 1.

## Execution flow
```text
teacher_logits.detach()
   ↓
log_softmax(student / T)  ·  softmax(teacher / T)
   ↓
KL divergence
   ↓
áp sample_mask → chỉ mẫu có teacher hợp lệ
   ↓
× T² (bù scale gradient do chia T)
```

## Called by
`Blip2Qformer.forward:936` — chỉ khi `teacher_mask.any()` và `λ_distill > 0`.

## Side effects
Không.

## Error handling
`sample_mask` rỗng → trả 0 nối đồ thị.

## Tests
`tests/test_stage1_objectives.py`

## Modification risk
Bỏ `detach()` là lỗi tinh vi: loss vẫn giảm, nhưng teacher bị kéo về phía student
và mất giá trị làm giám sát.
