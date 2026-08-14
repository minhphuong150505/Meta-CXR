> Source: `vision_encoders/pubmedclip/pubmed_clip.py` (92 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-14

# `vision_encoders/pubmedclip/pubmed_clip.py`

## Purpose

Bọc PubMedCLIP thành encoder đóng băng, chiều ra **768**.

## Role in architecture

Một trong **hai** encoder bật ở config production (Swin tắt 2026-08-14). Đầu ra đi vào `ViewFusionModule`
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


## Tiền xử lý ảnh — đúng nội dung, và từ 2026-08-14 chạy trên GPU

`forward` không dùng tensor 448×448 dùng chung trực tiếp: nó cho qua image
processor riêng của CLIP, ra 224×224 với `image_mean/std` của CLIP.
`do_rescale=False` là **bắt buộc** — tensor từ dataset đã ở `[0,1]` sau
`ToTensor()`, để processor chia 255 lần nữa sẽ cho đặc trưng gần như hằng số.

`__init__` dùng `CLIPImageProcessorFast`, **không** phải `CLIPProcessor`:

- Bản chậm kéo tensor CUDA về **CPU numpy**, resize ở đó, trả tensor CPU rồi
  `forward` copy ngược lên GPU. Đo tại batch 6, 448×448: **55.5 ms/batch so với
  0.3 ms**, trên một bước ~570 ms — khoảng một phần mười thời gian train, tức
  ~6 giờ của một run 10 epoch, dùng để resize ảnh trên CPU.
- Đặc trưng ra gần như không đổi: lệch tương đối **0.883%**, cosine similarity
  **0.99998** mỗi mẫu. Đo trên nhiễu ngẫu nhiên — trường hợp tệ nhất cho khác
  biệt resampling vì toàn tần số cao; ảnh X-quang thật mượt hơn nên sát hơn.
- Gọi thẳng image processor chứ không qua `CLIPProcessor(use_fast=True)`: wrapper
  đó vẫn giả định class chậm và ném `AttributeError: '_valid_processor_keys'`
  trên transformers 4.53. `forward` chỉ đọc `pixel_values`, không chạm tokenizer.

`CLIPModel.from_pretrained` truyền `use_safetensors=True`. Trên venv đúng
(torch 2.9.1) nó nạp được cả hai cách; cờ này giữ lại vì nạp safetensors vẫn hơn
nạp pickle.

⚠ **`forward` bọc vision model trong `torch.no_grad()` và `train()` ép
`self.model.eval()`.** Nên đặt `requires_grad=True` để fine-tune encoder này sẽ
**không có tác dụng gì** — gradient không chảy qua được. Muốn mở đóng băng phải
sửa cả hai chỗ đó trước.
