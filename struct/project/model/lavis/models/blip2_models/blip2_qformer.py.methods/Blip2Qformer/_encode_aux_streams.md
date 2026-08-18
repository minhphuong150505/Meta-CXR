> Source: `model/lavis/models/blip2_models/blip2_qformer.py:560-629`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-18

# `Blip2Qformer._encode_aux_streams(aux_image, cached=None, aux_mask=None)`

## Located in

[`blip2_qformer.py`](../../blip2_qformer.py.doc.md)

## Purpose
Encode auxiliary view, **batched** và dưới `no_grad`.

## Signature
```python
def _encode_aux_streams(self, aux_image, cached=None, aux_mask=None) -> dict[str, Tensor]
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

## ★ Lọc hàng aux rỗng trước khi vào encoder — 2026-08-18

`collater` pad các study ragged bằng `torch.zeros_like(anchor)`, và **44.7% study
train không có auxiliary view nào**. Trước thay đổi này, những slot padding đó vẫn
đi qua cả ba encoder đóng băng — encode một tấm ảnh toàn số 0 để rồi
`ViewFusionModule` gate kết quả về đúng 0.

`aux_mask [B, N]` giờ được truyền xuống đây và chỉ những hàng thật mới được encode:

```python
keep = real_aux_rows(aux_mask, flat.shape[0], flat.device)   # bool [B*N] hoặc None
real = flat if keep is None else flat[keep]
... scatter_aux_rows(encoded, keep, B, N)                    # slot padding = 0
```

Ba tính chất giữ cho thay đổi này **không đổi kết quả**:

1. `scatter_aux_rows` trả về đúng shape `[B, N, P, D]`, nên mọi caller không đổi.
2. Slot padding mang giá trị 0 thay vì "đáp ứng của encoder với ảnh toàn 0" —
   không ai đọc: `ViewFusionModule` mask chúng khỏi softmax và gate residual về 0,
   `MultiPositiveContrastiveLoss` loại chúng qua `cand_valid`.
3. `real_aux_rows` trả `None` khi **mọi** slot đều thật, nên batch đầy đủ giữ
   nguyên đường dense và không trả phí indexing.

Trường hợp **không study nào có aux**: hàm trả về sớm, không gọi encoder, `_fuse`
rơi về `anchor` — đúng bằng thứ gate-0 sẽ tạo ra, nhưng bớt một forward encoder và
một fusion block.

⚠ Đây là thay đổi **hiệu năng**, không phải thay đổi ngữ nghĩa. Nhưng nó **đổi số
lần rút RNG** của dropout trong các encoder chạy ở train mode, nên một run mới sẽ
không trùng bit-for-bit với run cũ dù cùng seed.

## Execution flow
```text
cached có sẵn → dùng luôn (đã shape [B,N,P,D])
   ↓
need = encoder bật nhưng chưa có trong streams
need rỗng → return
aux_image is None mà need không rỗng → ValueError nêu tên stream
   ↓
flat = aux_image.flatten(0,1)          [B*N,3,H,W]
keep = real_aux_rows(aux_mask, B*N, device)     ← None nghĩa là "không cần lọc"
keep có nhưng toàn False → return streams (không encode gì cả)
real = flat[keep]                      [n_keep,3,H,W]   n_keep ≈ 0.55 × B*N
with torch.no_grad():
   biovil     → visual_encoder(real).projected_patch_embeddings.reshape(...,1408)
   pubmedclip → pubmedclip(real, apply_aug=False)[0]     ← 768, KHÔNG chiếu
   swin       → swin(real)
   raddino    → raddino(real)
   ↓ scatter_aux_rows về [B,N,P,D], slot padding = 0
```

Comment `:442`: luồng pubmedclip fuse ở **768**; phép chiếu 1408 được tính lại từ
token đã fuse, nên projection của aux bị bỏ đi ở đây.

## Error handling
`aux_image is None` mà cần encode → `ValueError` **nêu tên các stream thiếu** và
gợi ý feature cache (`:428`).

## Tests
`tests/test_view_fusion.py` — `test_real_aux_rows_*`, `test_scatter_aux_rows_round_trip_leaves_padding_at_zero` và `test_filtered_aux_fuses_identically_to_dense_aux` khoá bất biến "lọc rồi fuse == fuse dense". Bản thân method này không import được trên máy CPU (kéo theo torchvision), nên logic lọc nằm ở `mhcac/view_fusion.py` để test được.

## Modification risk
Bỏ `no_grad` → gradient chảy vào encoder đóng băng qua nhánh aux, tăng bộ nhớ và
phá giả định "encoder không học".

Quên truyền `aux_mask` từ `_encode_image_streams` → im lặng quay lại encode toàn bộ
slot padding. Không có lỗi nào nổ ra; chỉ chậm đi. Nếu đo throughput mà không thấy
lợi, kiểm tra chỗ gọi trước.
