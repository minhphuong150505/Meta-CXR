> Source: `model/lavis/models/blip2_models/blip2_qformer.py:421-472`
> Status: ✅ ACTIVE

# `Blip2Qformer._encode_aux_streams(aux_image, cached=None)`

## Located in

[`blip2_qformer.py`](../../blip2_qformer.py.doc.md)

## Purpose
Encode auxiliary view, **batched** và dưới `no_grad`.

## Signature
```python
def _encode_aux_streams(self, aux_image, cached=None) -> dict[str, Tensor]
```

## Returns
`dict[name, [B, N, P, D]]` — raw output của encoder đóng băng.

## ★ Chi tiết quan trọng — docstring `:401`
> *"Batched (one encoder call over B*N images, not a per-image loop) and under
> no_grad. That detaches the auxiliary **features** only — the fusion modules
> W_K/W_V are applied outside this block and still get gradient."*

Hai điều:
1. **Một lời gọi encoder cho B×N ảnh**, không phải vòng lặp — khác biệt lớn về tốc độ.
2. `no_grad` chỉ detach **feature**, không detach fusion. `W_K`/`W_V` nằm trong
   `ViewFusionBlock`, được áp **bên ngoài** block này nên vẫn học được.

Hiểu sai điểm 2 dẫn tới kết luận sai rằng nhánh aux không học gì.

## Execution flow
```text
cached có sẵn → dùng luôn (đã shape [B,N,P,D])
   ↓
need = encoder bật nhưng chưa có trong streams
need rỗng → return
aux_image is None mà need không rỗng → ValueError nêu tên stream
   ↓
flat = aux_image.flatten(0,1)          [B*N,3,H,W]
with torch.no_grad():
   biovil     → visual_encoder(flat).projected_patch_embeddings.reshape(...,1408)
   pubmedclip → pubmedclip(flat, apply_aug=False)[0]     ← 768, KHÔNG chiếu
   swin       → swin(flat)
   raddino    → raddino(flat)
   ↓ unflatten về [B,N,P,D]
```

Comment `:442`: luồng pubmedclip fuse ở **768**; phép chiếu 1408 được tính lại từ
token đã fuse, nên projection của aux bị bỏ đi ở đây.

## Error handling
`aux_image is None` mà cần encode → `ValueError` **nêu tên các stream thiếu** và
gợi ý feature cache (`:428`).

## Tests
`tests/test_view_fusion.py` (gián tiếp)

## Modification risk
Bỏ `no_grad` → gradient chảy vào encoder đóng băng qua nhánh aux, tăng bộ nhớ và
phá giả định "encoder không học".
