> Source: `vision_encoders/pubmedclip/pubmed_clip.py` (68 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `vision_encoders/pubmedclip/pubmed_clip.py`

## Purpose

Bọc PubMedCLIP thành encoder đóng băng, chiều ra **768**.

## Role in architecture

Một trong ba encoder bật ở config production. Đầu ra đi vào `ViewFusionModule`
rồi `SharedVisualTokenProjector`.

## Status

```text
✅ ACTIVE — encoders.pubmedclip: true ở mọi config production
```

## Main class

`Pubmedclip(nn.Module)` (`:5`)

| Method | Dòng | Ghi chú |
|---|---|---|
| `__init__(aug=None, device='cuda', project=True)` | 6 | ⚠ **`device='cuda'` là default cứng** |
| `train(mode=True)` | 37 | Override để giữ eval mode |
| `forward(image, apply_aug=True)` | 42 | Trả tuple; `[0]` là patch embedding |

## ⚠ Được dựng với `project=False`

```python
Pubmedclip(aug=None, project=False).eval()    # blip2_qformer.py:258
```

Head `mlp` riêng của encoder **bị bỏ qua có chủ đích** — comment `:257` ghi rõ
"SharedVisualTokenProjector owns the 1408 projection". Mọi stream được chiếu như
nhau, ở một chỗ.

## ⚠ `aug=None` và `apply_aug=False`

Augmentation áp **một lần trong `ReportDataset`**, để mọi encoder thấy cùng một
ảnh đã biến đổi (comment `blip2_qformer.py:254`). Vì thế encoder được gọi với
`apply_aug=False` ở cả `:445` và `:515`.

## Calls / Called by

Gọi: thư viện CLIP ⚠ package chính xác cần runtime verification.
Được gọi: `blip2_qformer.py:26` (import), `:258` (dựng), `:445` (aux), `:515` (anchor);
`inference.py:343-344` đặt `.device = "cuda"`.

## Side effects

Tải weight PubMedCLIP lần đầu (mạng). Cấp phát GPU.

## Error / edge cases

⚠ `device='cuda'` mặc định — trên máy không có CUDA cần override. Hành vi cụ thể
cần runtime verification.

## Related tests

Không có test trực tiếp; `tests/test_shared_visual_tokens.py` dùng chiều 768 giả lập.

## Developer notes

1. **Đừng bật lại `project=True`** — sẽ có hai phép chiếu chồng nhau.
2. Chiều 768 được **hardcode** ở `blip2_qformer.py:310` và `:333`. Đổi encoder khác
   chiều phải sửa cả hai chỗ đó.

## Source relationships

- **Parent:** [`vision_encoders/_index.md`](_index.md)
- **Related:** [`shared_visual_tokens.py`](shared_visual_tokens.py.doc.md)

← [HOME](../../HOME.md)
