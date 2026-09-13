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

## `loss_mention_conditioned` — thêm 2026-09-11

Field mới trên `BlipOutput`, mặc định `None`. **Bắt buộc phải có** khi bật
`model.loss.lambda_mention_conditioned_cls`: chế độ đó ép `lambda_cls` và
`lambda_gate` về 0,0, nên `loss_cls` và `loss_gate` in đúng `0.0000`. Không có
field này thì term thay thế chúng không hiện ở đâu, và một run có `--options`
không ăn sẽ trông hoàn toàn khỏe mạnh trong khi không train objective phân loại
nào. `base_task.train_step` tự thu mọi key chứa `loss`, nên thêm field là đủ để
nó vào MetricLogger — không phải sửa gì ở task.
