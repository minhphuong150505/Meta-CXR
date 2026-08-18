> Source: `vision_encoders/pubmedclip/pubmed_clip.py` (122 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-14

# `vision_encoders/pubmedclip/pubmed_clip.py`

## Purpose

Bọc PubMedCLIP thành encoder đóng băng, chiều ra **768**, trả **50 token**:
1 CLS (toàn cục) + 49 patch đã trừ DC (cục bộ).

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
| `__init__(aug=None, device=None, project=True)` | 6 | `device=None` → resolve `cuda` nếu có, ngược lại `cpu` (sửa 2026-08-18) |
| `train(mode=True)` | 61 | Override để giữ eval mode |
| `forward(image, apply_aug=True)` | 66 | Trả tuple; `[0]` là `[B, 50, 768]` |

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

## ★ Cách đọc đầu ra — hai chỉnh sửa bắt buộc (2026-08-14)

Đo trên ảnh thật của dataset (16–32 ảnh, CPU). Không có hai bước này thì
PubMedCLIP gần như không đóng góp gì cho MHCAC.

**1. `post_layernorm`.** HF trả `last_hidden_state` của tháp thị giác **chưa qua**
LayerNorm cuối (`modeling_clip.py:763-765` chỉ áp cho CLS đã pool). ViT của CLIP
là pre-LN nên tensor đó là residual stream thô. Token 0 cũng chỉ có nghĩa sau
LayerNorm này — nó chính là vector CLIP được huấn luyện contrastive.

**2. Trừ DC khỏi patch token.** 49 patch có cosine trung bình đôi một **0.674**:
hai phần ba mỗi token là một hướng cố định không mang thông tin không gian, nên
attention lên chúng gần như phẳng và cả stream hoạt động như một hằng số. 97%
hướng đó là hằng số trên toàn dataset (`cos = 0.970` giữa mean từng ảnh và mean
toàn cục) → là offset, không phải nội dung.

| Cách đọc | cosine giữa 49 patch |
|---|---|
| `last_hidden_state` thô (trước 2026-08-14) | 0.674 |
| `+ post_layernorm` | 0.587 |
| `+ post_layernorm + trừ mean từng ảnh` | **−0.014** |

Phép tách là có chủ đích và là thứ làm hai encoder bổ trợ nhau: **token 0 mang
cái nhìn toàn cục, 49 patch mang sai lệch cục bộ so với nó.** Xem
[D-016](../_meta/DECISIONS.md#d-016--mỗi-encoder-giữ-thang-đo-riêng).

⚠ **Đừng đưa 448 vào encoder này.** Đã thử `interpolate_pos_encoding=True`:
cosine *tệ hơn* (0.714 thô) và nó đẩy CLIP ra ngoài phân bố huấn luyện trong khi
encoder đóng băng nên không có cơ hội thích nghi. 224 là độ phân giải gốc, và
lưới 7×7 thô của nó là **vai trò** chứ không phải khiếm khuyết — BioViL đã lo
phần chi tiết ở 14×14.

⚠ `forward` bọc tháp thị giác trong `torch.no_grad()` và `train()` ép `eval()`.
Đặt `requires_grad=True` lên encoder này **không có tác dụng gì** — muốn
fine-tune phải gỡ cả hai chỗ.

## Calls / Called by

Gọi: thư viện CLIP ⚠ package chính xác cần runtime verification.
Được gọi: `blip2_qformer.py:26` (import), `:258` (dựng), `:445` (aux), `:515` (anchor);
`inference.py:343-344` đặt `.device = "cuda"`.

## Side effects

Tải weight PubMedCLIP lần đầu (mạng). Cấp phát GPU.

## Error / edge cases

**`device='cuda'` cứng — ĐÃ SỬA 2026-08-18.** Giữ lại ghi chép vì triệu chứng của
nó không hề trỏ về file này:

```
RuntimeError: No CUDA GPUs are available
  vision_encoders/pubmedclip/pubmed_clip.py:28 in __init__
```

Đây là module DUY NHẤT trong vision stack không dựng được trên máy không có GPU,
và nạn nhân là `scripts/check_itc_gate.py --device cpu` — đúng cái script có
nhiệm vụ quyết định (rẻ) xem các objective vision-language có đáng một run 33 h
hay không. Nó chết ngay trong constructor, trước khi đọc một tấm ảnh nào, nên
`--device cpu` **chưa bao giờ chạy được**. Nó cũng vi phạm quy tắc "không hardcode
cuda" của repo (xem `runtime/device.py`).

Giờ `device=None` → `"cuda" if torch.cuda.is_available() else "cpu"`. Cả hai chỗ
dùng `self.device` đều là `.to()` lúc init và `forward()` không đọc nó, nên trên
máy có GPU giá trị resolve ra vẫn là `cuda` và đường training không đổi một byte.

⚠ Điều này khiến chạy gate check trên CPU **song song** với một run GPU trở nên
khả thi — cần thiết, vì run chiếm ~15.1 GB / 16.3 GB và không còn chỗ cho model
thứ hai trên card.

## Related tests

Không có test trực tiếp; `tests/test_shared_visual_tokens.py` dùng chiều 768 giả lập.

## Developer notes

1. **Đừng bật lại `project=True`** — sẽ có hai phép chiếu chồng nhau.
2. Chiều 768 được **hardcode** ở `blip2_qformer.py:310` và `:333`. Đổi encoder khác
   chiều phải sửa cả hai chỗ đó.
3. Số token (50) được `Blip2Qformer._native_stream_layouts` suy ra từ
   `model.config.vision_config` chứ không hardcode; MHCAC kiểm lại lúc forward và
   raise nếu lệch.

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
