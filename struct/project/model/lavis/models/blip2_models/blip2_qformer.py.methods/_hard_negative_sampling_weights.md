> Source: `model/lavis/models/blip2_models/blip2_qformer.py:68-115`
> Status: ✅ ACTIVE

# `_hard_negative_sampling_weights(similarities, candidate_valid, positive_indices)`

## Located in

[`blip2_qformer.py`](../blip2_qformer.py.doc.md)

## Purpose
Trả về **xác suất hữu hạn** trên các candidate hợp lệ không phải positive, để lấy
hard negative.

## ★ Vì sao hàm này tồn tại — docstring `:57`
> *"Masking the positive **after** softmax is numerically unsafe: once the model
> becomes confident, BF16 can assign the positive probability 1 and every negative
> probability 0. Removing the positive then leaves an all-zero row, which makes
> `torch.multinomial` trigger a CUDA device-side assertion."*

Đây là một bug thật, chỉ xuất hiện **sau khi model đã học tốt** và **chỉ ở BF16**
— tức là muộn trong run production, đúng lúc đắt nhất.

## Signature
```python
def _hard_negative_sampling_weights(similarities, candidate_valid, positive_indices) -> Tensor
```

## Parameters
| Tham số | Shape | Ghi chú |
|---|---|---|
| `similarities` | `[batch, candidates]` | ⚠ phải 2-D |
| `candidate_valid` | `[candidates]` bool | |
| `positive_indices` | `[batch]` long | Vị trí positive mỗi hàng |

## Returns
`[batch, candidates]` — xác suất, **luôn hữu hạn**, tổng 1 mỗi hàng.

## Execution flow
```text
validate: ndim==2, candidate_valid khớp, positive_indices khớp  → ValueError
   ↓
allowed = candidate_valid.expand(batch,-1).clone()
allowed[rows, positive_indices] = False          ← LOẠI POSITIVE TRƯỚC
   ↓
fallback_total = allowed.sum(1); nếu có hàng ==0 → ValueError
   ↓
logits = nan_to_num(similarities.detach().FLOAT(), nan=-1e4, ±inf=±1e4)   ← FP32
logits = masked_fill(~allowed, -inf)
weights = softmax(logits).masked_fill(~allowed, 0)
weights = nan_to_num(weights, 0)
normalized = weights / totals.clamp_min(tiny)
   ↓
fallback = allowed.float() / fallback_total       ← ĐỀU
bad_rows = ~isfinite(totals) | (totals <= 0)
return where(bad_rows, fallback, normalized)      ← fallback giữ training sống
```

## Detailed logic
Ba lớp phòng vệ chồng nhau:
1. **FP32** — tính trong độ chính xác cao hơn BF16 của model.
2. **Loại positive trước softmax** — hàng không bao giờ thành toàn 0.
3. **Fallback đều** — nếu similarity vẫn không hữu hạn, dùng phân phối đều thay vì crash.

`detach()` vì đây là lấy mẫu, không cần gradient.

## Side effects
Không. Hàm thuần cấp module.

## Error handling
| Điều kiện | Lỗi |
|---|---|
| `similarities.ndim != 2` | `ValueError("similarities must have shape [batch, candidates]")` |
| `candidate_valid.numel() != candidate_count` | `ValueError` |
| `positive_indices.numel() != batch_size` | `ValueError` |
| Hàng không còn candidate nào | `ValueError("hard-negative sampling needs at least one candidate per row")` |

## Called by
`Blip2Qformer._image_text_matching:742,745` (dưới `no_grad`)

## Tests
`tests/test_blip2_negative_sampling.py` ⚠ cần torchvision

## Modification risk
⚠ **Đừng "đơn giản hóa" bằng cách mask sau softmax.** Đó chính là bug hàm này sửa,
và nó chỉ hiện ra muộn trong run GPU dài.
