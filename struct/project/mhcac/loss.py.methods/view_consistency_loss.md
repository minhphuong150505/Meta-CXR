> Source: `mhcac/loss.py:590-628`
> Status: ✅ ACTIVE

# `view_consistency_loss(fused_logits, anchor_logits, has_aux)`

## Located in

[`loss.py`](../loss.py.doc.md)

## Purpose
Ép dự đoán từ bản **đã fuse** gần với dự đoán **chỉ anchor**.

## ★ Vì sao cần
Fusion thêm thông tin từ view phụ. Nhưng nó không được **đảo ngược** kết luận một
cách tùy tiện — nếu anchor rõ ràng cho thấy tràn khí màng phổi, thêm một view
nghiêng không nên làm dự đoán biến mất.

Loss này là ràng buộc mềm: cho phép fusion tinh chỉnh, phạt nó khi đi quá xa.

## Signature
```python
def view_consistency_loss(fused_logits, anchor_logits, has_aux) -> Tensor
```
| Tham số | Shape |
|---|---|
| `fused_logits` | `[B,14,3]` — từ `shared_visual` đã fuse |
| `anchor_logits` | `[B,14,3]` — MHCAC chạy trên **chỉ anchor** |
| `has_aux` | `[B]` bool = `aux_mask.any(dim=1)` |

## ⚠ Chi phí: MHCAC chạy lần thứ ba
`anchor_logits` đòi hỏi `Blip2Qformer.forward:964-969` tái dựng
`SharedVisualTokens` từ `_last_prefusion_streams` (áp `ln_vision` lại cho biovil)
rồi **gọi MHCAC lần nữa**.

Đây là lý do MHCAC chạy tối đa 3 lần mỗi forward.

## Config dependencies
`loss.lambda_view_consistency` — prod **0.05**. Chỉ chạy khi `> 0` (`:955`).

## Called by
`Blip2Qformer.forward:970`

## Side effects
Không (nhưng nhánh gọi nó tốn một lần forward MHCAC).

## Tests
`tests/test_multiview_losses.py`

## Modification risk
`lambda` quá cao → fusion bị ép thành no-op, mất hết lợi ích multi-view. Quá thấp →
fusion tự do đảo kết luận.
