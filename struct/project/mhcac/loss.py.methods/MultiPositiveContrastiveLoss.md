> Source: `mhcac/loss.py:523-589`
> Status: ✅ ACTIVE

# `class MultiPositiveContrastiveLoss(nn.Module)`

## Located in

[`loss.py`](../loss.py.doc.md)

## Purpose
Kéo các view **của cùng một study** lại gần nhau trong không gian đặc trưng.

## Signature
```python
MultiPositiveContrastiveLoss(temperature=0.07)
forward(anchor, aux, aux_mask) -> Tensor
```

| Tham số | Shape |
|---|---|
| `anchor` | `[B, P, D]` — raw, **trước** projection |
| `aux` | `[B, N, P, D]` |
| `aux_mask` | `[B, N]` bool |

## ★ "Multi-positive" nghĩa là gì
Contrastive thường có **một** positive cho mỗi anchor. Ở đây một study có thể có
nhiều view, và **tất cả** đều là positive của nhau. Loss phải xử lý nhiều positive
cùng lúc thay vì chọn một.

## Chạy trên tensor pre-fusion
`Blip2Qformer.forward:948-954` lấy từ `self._last_prefusion_streams` — tức là
**trước khi fusion diễn ra**. Đúng như vậy: mục tiêu là ép các view *đã* gần nhau,
để fusion có gì đó hợp lý để hợp nhất.

Nó chạy cho **từng encoder** rồi lấy trung bình (`torch.stack(terms).mean()`).

## Config dependencies
`loss.lambda_mpc` — prod **0.1**; `0.0` ở `mimic_cxr_2gpu.yaml` legacy.
Chỉ dựng khi `lambda_mpc > 0` (`blip2_qformer.py:319`).

## Called by
`Blip2Qformer.forward:949` — chỉ khi `multi_view` và `aux_mask.any()`.

## Side effects
Không.

## Tests
`tests/test_multiview_losses.py`

## Modification risk
Chạy trên tensor **sau** fusion sẽ vô nghĩa — fusion đã trộn chúng rồi.
