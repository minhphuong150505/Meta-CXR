> Source: `mhcac/explanation.py:6-29`
> Status: 🟡 CONDITIONAL

# `logit_difference_squared(logits, labels, sample_mask=None)`

## Purpose

Tính một score cho mỗi study:

```text
s[b] = Σ_k 1[labels[b,k] == Positive] · (logits[b,k,1] - logits[b,k,0])²
```

## Inputs / outputs

- `logits [B,A,C]`, với index `0=Negative`, `1=Positive`;
- `labels [B,A]`;
- `sample_mask [B]` tùy chọn.

Trả tuple `(score [B] float32, valid [B] bool)`. `valid=True` khi study vừa qua
`sample_mask` vừa có ít nhất một bệnh Positive. Study không hợp lệ có score đúng 0.

## Error handling

Sai rank, shape logit/label không khớp, thiếu hai class đầu, hoặc `sample_mask`
không có đúng B phần tử → `ValueError`.

## Called by

`ExplanationLoss.forward` và test đơn vị trực tiếp.
