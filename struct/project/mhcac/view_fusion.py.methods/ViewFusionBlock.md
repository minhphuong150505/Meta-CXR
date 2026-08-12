> Source: `mhcac/view_fusion.py:11-77`
> Status: ✅ ACTIVE

# `class ViewFusionBlock(nn.Module)`

## Located in

[`view_fusion.py`](../view_fusion.py.doc.md)

## Purpose
Một khối pre-norm cross-attention có **residual gating zero-init**.

## Công thức — docstring `:14`
```text
h   = anchor + Dropout(W_O · MHA(LN_q(anchor + view_emb),
                                 LN_kv(aux + view_emb), mask))
out = h      + Dropout(FFN(LN_f(h)))
```

## Constructor (`:24`)
`(dim, num_heads, ffn_ratio, dropout)` — validate `dim % num_heads == 0`, nếu không
`ValueError` nêu cả hai số.

Sub-module: `norm_q`, `norm_kv`, `norm_ffn`, `w_q/w_k/w_v/w_o`, `ffn` (Linear→GELU→
Dropout→Linear), `dropout`.

## ★ Zero-init — identity chính xác tại step 0
```python
nn.init.zeros_(self.w_o.weight);     nn.init.zeros_(self.w_o.bias)      # :48
nn.init.zeros_(self.ffn[-1].weight); nn.init.zeros_(self.ffn[-1].bias)  # :50
```
Cả hai nhánh residual ra 0 → `out == anchor` **chính xác**. Checkpoint single-view
load vào không có regression nào. `tests/test_view_fusion.py` kiểm đúng điều này.

## ★ Pre-norm giữ scale — docstring `:19`
> *"Pre-norm keeps the residual stream unscaled: the fused tensors are raw
> pre-projection encoder outputs, and the downstream pretrained projections were
> fitted against that scale."*

Post-norm sẽ chuẩn hóa lại residual → đổi scale → projection pretrained ở hạ nguồn
không còn đúng.

## `forward(anchor, anchor_view_emb, aux_tokens, attn_bias, gate)` (`:57`)
| Tham số | Shape |
|---|---|
| `anchor` | `[B, P, D]` |
| `anchor_view_emb` | `[B, 1, D]` hoặc `None` |
| `aux_tokens` | `[B, N*P, D]` |
| `attn_bias` | `[B, 1, 1, N*P]` — dùng `MASK_NEG = -1e4` |
| `gate` | `[B, 1, 1]` float — **`0` cho study không có aux** |

## `_split_heads(x)` (`:53`)
`[B,L,D]` → `[B, heads, L, head_dim]`

## Modification risk
⚠ Bỏ zero-init → step 0 đã làm hỏng biểu diễn, checkpoint cũ không dùng lại được.
⚠ Đổi `MASK_NEG` thành `-inf` → hàng toàn padding cho NaN.
