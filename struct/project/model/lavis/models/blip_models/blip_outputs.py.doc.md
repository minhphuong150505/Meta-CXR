> Source: `model/lavis/models/blip_models/blip_outputs.py` (146 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-13

# `blip_outputs.py`

## Purpose

Định nghĩa các `ModelOutput` dataclass của fork BLIP. `BlipOutput` là hợp đồng giữa
`Blip2Qformer.forward`, train/eval task và logger.

## Main dataclasses

| Class | Vai trò |
|---|---|
| `BlipSimilarity` | Similarity i2t/t2i và target |
| `BlipIntermediateOutput` | Embedding/output trung gian BLIP |
| `BlipOutput` | Total loss, thành phần loss Stage 1, logits và mask phân loại |
| `BlipOutputWithLogits` | Mở rộng `BlipOutput` bằng logits tổng quát |
| `BlipOutputFeatures` | Feature image/text/multimodal |

## `BlipOutput` extension của Meta-CXR

Ngoài field upstream, fork thêm các loss phân loại/MHCAC/multi-view. Giai đoạn
explanation-aware thêm:

```python
loss_explanation: Optional[torch.FloatTensor] = None
```

Khi `lambda_explanation == 0`, `Blip2Qformer.forward` để field này là `None`, nên
đường mặc định không thêm metric loss zero vào dict logger. Khi feature được bật,
field là scalar graph-connected, kể cả batch thiếu mask hợp lệ.

## Called by

`Blip2Qformer.forward` dựng `BlipOutput`. `BaseTask.train_step` thu mọi key có chữ
`loss`; `ImageTextPretrainTask.evaluation` cũng cộng các loss không phải `None`.

## Dependencies

`dataclasses`, `typing`, `torch`, và `transformers.modeling_outputs`.

## Source relationships

- **Parent:** [`model/lavis/_index.md`](../../_index.md)
- **Caller:** [`blip2_qformer.py`](../blip2_models/blip2_qformer.py.doc.md)

← [HOME](../../../../../HOME.md)
