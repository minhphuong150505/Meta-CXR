> Source: `model/lavis/models/blip2_models/blip2_qformer.py:1119-1182`
> Status: ✅ ACTIVE — inference bridge

# `Blip2Qformer.forward_image(...)`

## Purpose

Inference-only bridge: ảnh/cache/multi-view → image-only MHCAC logits và 32
Q-Former embeddings cho Vicuna demo hoặc Stage-2 Q-Former records.

## Flow

`_encode_image_streams(apply_aug=False)` → MHCAC student không text → Q-Former
query cross-attention → trả `(logits [B,14,3], query [B,32,768])`.

## Important invariant

Không được đưa report text vào hàm này: output phải giống deployed image-only
student. Aux view chỉ ảnh hưởng qua configured view fusion/cache.

## Called by / tests

`inference.py:get_response`, `training/stage1/lavis_loader` record build;
invariant gián tiếp trong `tests/test_stage1_objectives.py`.

← [`Blip2Qformer`](_index.md) · [`blip2_qformer.py`](../../blip2_qformer.py.doc.md)
