> Source: `mhcac/view_fusion.py` (153 dòng)
> Status: 🟡 CONDITIONAL — `multi_view: true`
> Last verified against source: 2026-08-12

# `mhcac/view_fusion.py`

## Purpose

Hợp nhất anchor view với auxiliary view bằng cross-attention, **trên đầu ra thô của
encoder, trước mọi phép chiếu**.

## Why it exists

Một study X-quang thường có nhiều góc chụp; một bất thường thấy rõ ở góc nghiêng
có thể mờ ở góc thẳng. Module này để anchor "hỏi" các view khác.

Ba ràng buộc khiến nó phải viết cẩn thận: (1) checkpoint single-view cũ phải load
được không hỏng, (2) study không có aux không được làm vỡ batch, (3) scale của
tensor phải giữ nguyên vì projection pretrained ở hạ nguồn đã fit theo scale đó.

## Role in architecture

```text
encoder raw [B,P,D] ──► ViewFusionModule[tên_encoder] ──► [B,P,D] ──► SharedProjector
        ▲                                                    (shape KHÔNG đổi)
aux raw [B,N,P,D] ──────┘
```

Một module **cho mỗi encoder bật**, vì mỗi encoder có `D` khác nhau.

## Status

```text
🟡 CONDITIONAL — chỉ dựng khi model.multi_view: true (prod: true, legacy 2gpu: false)
```

## Used in

Training ✅ · Validation ✅ · Inference ✅ (nếu checkpoint có multi_view)

## Entry point

Không.

## Inputs

`forward(anchor, aux, aux_mask=None, anchor_view_id=None, aux_view_ids=None)`

| Tham số | Shape |
|---|---|
| `anchor` | `[B, P, D]` |
| `aux` | `[B, N, P, D]` |
| `aux_mask` | `[B, N]` bool |
| `anchor_view_id` | `[B]` long |
| `aux_view_ids` | `[B, N]` long |

## Outputs

`[B, P, D]` — **shape hệt input anchor**. Đây là hợp đồng khiến MHCAC và Q-Former
không cần biết multi-view có bật hay không.

## Important configuration

```yaml
model:
  view_fusion:
    num_heads: 8        ffn_ratio: 4      num_blocks: 1
    num_view_types: 4   dropout: 0.1      p_view_drop: 0.15
```

⚠ `from_config` đọc **tường minh** sáu key này (`blip2_qformer.py:1370-1377`).
Thêm key mới mà quên thêm dòng đọc = key không có hiệu lực.

## Main classes

| Class | Dòng | Doc |
|---|---|---|
| `ViewFusionBlock` | 11 | [📄](view_fusion.py.methods/ViewFusionBlock.md) |
| `ViewFusionModule` | 78 | [📄](view_fusion.py.methods/ViewFusionModule.md) |

## Ba quyết định thiết kế — mỗi cái giải một vấn đề cụ thể

### 1. Zero-init → identity chính xác tại step 0

```python
nn.init.zeros_(self.w_o.weight);   nn.init.zeros_(self.w_o.bias)      # :48
nn.init.zeros_(self.ffn[-1].weight); nn.init.zeros_(self.ffn[-1].bias) # :50
```

Cả nhánh attention và nhánh FFN đều ra 0 → `out == anchor` chính xác. **Checkpoint
single-view load vào không có regression nào**; model bắt đầu đúng bằng model cũ
rồi mới học phần fusion.

Đây là thứ `tests/test_view_fusion.py` kiểm.

### 2. Gate = 0 thay vì loại khỏi batch

Study không có aux được nhân với `gate [B,1,1] = 0`. Batch giữ nguyên shape; không
có nhánh điều khiển phụ thuộc dữ liệu, nên DDP không lệch giữa các rank.

### 3. `MASK_NEG = -1e4`, không phải `-inf`

```python
MASK_NEG = -1e4   # :8
```

Hàng toàn padding vẫn cho softmax hợp lệ thay vì NaN. `-inf` sẽ tạo `NaN` lan ra
toàn bộ gradient.

### Pre-norm giữ scale

Docstring `:19`: pre-norm giữ residual stream **không bị scale lại** — tensor được
fuse là output thô của encoder, và projection pretrained ở hạ nguồn đã fit theo
đúng scale đó.

## Execution flow

```text
ViewFusionModule.forward(anchor, aux, aux_mask, anchor_view_id, aux_view_ids)
 ├─ view embedding cho anchor và aux (num_view_types=4)
 ├─ p_view_drop: ngẫu nhiên bỏ aux view lúc train  (regularization)
 ├─ aux [B,N,P,D] → flatten → [B, N*P, D]
 ├─ attn_bias [B,1,1,N*P] từ aux_mask, dùng MASK_NEG
 ├─ gate [B,1,1] = 0 cho study không có aux
 └─ num_blocks × ViewFusionBlock(anchor, view_emb, aux_tokens, attn_bias, gate)
        h   = anchor + Dropout(W_O · MHA(LN_q(anchor+emb), LN_kv(aux+emb), mask))
        out = h + Dropout(FFN(LN_f(h)))
```

## Calls

`torch.nn` (`LayerNorm`, `Linear`, `Sequential`, `GELU`, `Dropout`, `Embedding`),
`torch.nn.functional`.

## Called by

`Blip2Qformer.__init__:315` (dựng `nn.ModuleDict`, một cho mỗi encoder) ·
`Blip2Qformer._fuse:465` · `tests/test_view_fusion.py`

## Data flow

Xem [DATA_FLOW.md §2.4](../_meta/DATA_FLOW.md#24-trong-model).

## Side effects

Dropout khi training. Không mutate state ngoài.

## Error / edge cases

| Tình huống | Hành vi |
|---|---|
| `dim % num_heads != 0` | `ValueError` nêu cả hai số (`:26`) |
| `aux` là `None` | `_fuse` trả `anchor` nguyên vẹn, không gọi module |
| `N_max == 0` | `has_aux_input` False → không encode aux |
| Hàng toàn padding | `MASK_NEG` giữ softmax hợp lệ |

## Related tests

`tests/test_view_fusion.py` — **identity tại step 0**, đây là test quan trọng nhất
`tests/test_multiview_losses.py` — loss đi kèm

## Related documentation

[ARCHITECTURE.md §2.2](../_meta/ARCHITECTURE.md#22-view-fusion--hợp-nhất-đa-view) ·
[GLOSSARY: View fusion](../_meta/GLOSSARY.md#view-fusion)

## Developer notes

1. ⚠ **Đừng bỏ zero-init.** Nó là thứ khiến checkpoint cũ load được. Bỏ đi thì
   step 0 đã làm hỏng biểu diễn.
2. ⚠ **Đừng đổi `MASK_NEG` thành `-inf`.**
3. **Shape contract `[B,P,D]` → `[B,P,D]` là bất khả xâm phạm** — MHCAC và Q-Former
   dựa vào nó.
4. Fusion chạy trên output **thô, trước projection**. Đổi vị trí sang sau projection
   sẽ đổi ngữ nghĩa và làm checkpoint không tương thích.
5. `p_view_drop: 0.15` chỉ tác dụng lúc train.

## Source relationships

- **Parent:** [`mhcac/_index.md`](_index.md)
- **Methods:** [`view_fusion.py.methods/`](view_fusion.py.methods/)
- **Related:** [`blip2_qformer.py`](../model/lavis/models/blip2_models/blip2_qformer.py.doc.md) · [`loss.py`](loss.py.doc.md)

← [HOME](../../HOME.md)
