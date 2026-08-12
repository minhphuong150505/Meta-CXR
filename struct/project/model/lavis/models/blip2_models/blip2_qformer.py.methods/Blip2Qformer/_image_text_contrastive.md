> Source: `model/lavis/models/blip2_models/blip2_qformer.py:670-747`
> Status: ✅ ACTIVE

# `Blip2Qformer._image_text_contrastive(...)`

## Located in

[`blip2_qformer.py`](../../blip2_qformer.py.doc.md)

## Purpose
ITC loss, có **negative queue 1024 mẫu** cho microbatch nhỏ.

## Signature
```python
def _image_text_contrastive(self, image_features, text_features, valid_mask)
    -> (loss_itc, sim_i2t, sim_t2i, valid_all)
```

## Parameters
`image_features [B,32,256]` · `text_features [B,256]` · `valid_mask [B]` bool
(= `generation_mask`)

## ★ Hai chi tiết quyết định tính đúng đắn

### 1. Queue tắt khi validate
```python
queue_filled = int(self.itc_queue_filled.item()) if self.training else 0   # :649
```
Comment `:646`: *"The queue is a training-only source of negatives. Validation must
not depend on whichever train samples happened to fill the ring buffer at the end
of an epoch."*

Không có dòng này, điểm validation sẽ phụ thuộc vào **thứ tự batch train** — không
tái lập và không so sánh được.

### 2. `.clone()` tách autograd khỏi ring buffer
Comment `:651`: *"clone() decouples autograds saved operands from the ring-buffer
update performed later in this same forward pass."*

`_update_itc_queue` chạy **sau** trong cùng forward. Không clone, autograd giữ
tham chiếu tới bộ nhớ sẽ bị ghi đè.

## Execution flow
```text
_gather_with_local_grad(image_features), (text_features)   ← all-gather giữ grad local
concat_all_gather(valid_mask)
   ↓ nếu training và queue có dữ liệu
cat với itc_image_queue[:filled].clone(), itc_text_queue[:filled].clone()
candidate_valid = cat(valid_all, ones(queue_filled))
   ↓
temperature = temp.clamp(1e-3, 0.5)
sim_i2t = einsum("bqd,nd->bnq").amax(-1) / T      ← amax: query token khớp nhất
sim_t2i = einsum("bd,nqd->bnq").amax(-1) / T
masked_fill(~candidate_valid, -inf)
   ↓
targets = rank * B + arange(B)                    ← vị trí positive trong all-gather
loss = 0.5 * (CE(sim_i2t[valid], targets[valid]) + CE(sim_t2i[valid], targets[valid]))
   ↓
return loss, sim_i2t[:, :current_count], sim_t2i[:, :current_count], valid_all
```

⚠ `sim_*` trả về **bị cắt về batch hiện tại** (`:714`) — vì ITM cần ảnh/token id
thật mà queue không giữ (comment `:710`).

## Returns
`(loss_itc, sim_i2t, sim_t2i, valid_all)` — ba cái sau đi thẳng vào `_image_text_matching`.

## Error handling
`valid_mask` không có mẫu nào → trả `zero` nối đồ thị + sim đã cắt.

## Config dependencies
`loss.itc_queue_size` (prod 1024), `loss.lambda_itc`

## Tests
`tests/test_blip2_negative_sampling.py` ⚠ cần torchvision

## Modification risk
Bỏ điều kiện `if self.training` → validation nhiễm negative từ train, điểm không
còn tin được.
