> Source: `model/lavis/models/blip2_models/blip2_qformer.py:634-668`
> Status: ✅ ACTIVE

# `Blip2Qformer._update_itc_queue(...)`

## Located in

[`blip2_qformer.py`](../../blip2_qformer.py.doc.md)

## Purpose
Đẩy feature của batch hiện tại vào ring buffer ITC.

## Signature
```python
@torch.no_grad()
def _update_itc_queue(self, image_features, text_features, valid_mask) -> None
```

## Execution flow
```text
itc_queue_size == 0 HOẶC not self.training → return ngay
   ↓
concat_all_gather(image_features.detach()), (text_features.detach()), (valid_mask)
lọc theo valid → bỏ mẫu không hợp lệ
images rỗng → return
   ↓
nếu count >= queue_size: chỉ giữ queue_size phần tử CUỐI
   ↓ ghi vòng (wrap-around)
pointer = itc_queue_ptr
first = min(count, queue_size - pointer)
ghi [pointer : pointer+first]
remaining = count - first
nếu remaining: ghi [0 : remaining]        ← quấn về đầu
   ↓
itc_queue_ptr = (pointer + count) % queue_size
itc_queue_filled = min(queue_size, filled + count)
```

## Detailed logic
**Ring buffer thật, không phải append.** `itc_queue_ptr` quấn vòng; `itc_queue_filled`
theo dõi đã đầy bao nhiêu (để `_image_text_contrastive` chỉ dùng phần hợp lệ).

**Chỉ chạy khi `self.training`** — validation không được làm bẩn queue.

**`.detach()` + `@torch.no_grad()`** — queue là nguồn negative tĩnh, không nhận gradient.

**Gọi SAU cả ba loss** trong `forward:904` — nếu gọi trước, batch hiện tại sẽ nằm
trong queue và contrastive tự so với chính nó.

## Side effects
⚠ **Mutate 4 buffer**: `itc_image_queue`, `itc_text_queue`, `itc_queue_ptr`,
`itc_queue_filled`.

## Config dependencies
`loss.itc_queue_size` (prod 1024). `0` → tắt hoàn toàn.

## Modification risk
⚠ **Đừng đổi vị trí gọi lên trước `_image_text_contrastive`** — sẽ làm mẫu tự so
với chính nó, contrastive loss mất ý nghĩa mà vẫn giảm đẹp.
