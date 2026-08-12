> Source: `mhcac/mhcac_12.py` (448 dòng)
> Status: ✅ ACTIVE — ★ bản DUY NHẤT được wire
> Last verified against source: 2026-08-12

# `mhcac/mhcac_12.py`

## Purpose

Bộ phân loại 14 bệnh lý × 3 lớp bằng **expert token cross-attention**. 14 token
học được, mỗi token chuyên trách một bệnh lý, cross-attend vào token thị giác qua
6 lớp.

## Why it exists

Một classifier head phẳng trên visual token gộp sẽ trộn lẫn bằng chứng của mọi
bệnh lý. Expert token cho mỗi bệnh lý một truy vấn riêng vào ảnh — và attention
map của nó trở thành thứ giải thích được.

Đây là bản thứ 12 và là **bản duy nhất được wire**. 11 bản trước là legacy
([D-003](../_meta/DECISIONS.md#d-003--mhcac-variants-và-encoder-trùng-lặp-là-legacy)).

## Role in architecture

```text
SharedVisualTokens ──► AbnormalityClassificationModel ──► logits [B,14,3]
        ▲                        │                        + orth/contrastive/sparsity loss
        │                        │
   (teacher-only)                └── attention_weights_list
   text_embeddings
```

## Status

```text
✅ ACTIVE
```

## Used in

Training ✅ (student + teacher + anchor) · Validation ✅ · Inference ✅ (chỉ student)

## Entry point

Không.

## Inputs

| Tham số | Kiểu | Ghi chú |
|---|---|---|
| `shared_visual_tokens` | `SharedVisualTokens` | `.tokens [B,N,1408]` + `.spans` |
| `text_embeddings` | `[B,L,768]` hoặc `None` | ⚠ **`None` = student, có giá trị = teacher** |
| `text_attention_mask` | `[B,L]` | Chỉ teacher |
| `labels` | `[B,14]` long | Cho contrastive loss; `None` → bỏ qua |
| `sample_mask` | `[B]` bool | Loại mẫu không nhãn |

## Outputs

Tuple 5 phần tử:
```python
(logits [B,14,3], attention_weights_list, contrastive_loss, orth_loss, sparsity_loss)
```

## Important configuration

Khởi tạo tại `blip2_qformer.py:342`:

```python
embed_dim=768, num_abnormalities=14, num_classes=3, num_layers=6,
num_commmon_tokens=14,      # ⚠ typo: ba chữ 'm'
visual_dim=1408, text_dropout_rate=<mhcac.text_dropout>,
use_cnn=<use_biovil>, uncertain_policy=<mhcac.uncertain_policy>
```

Từ YAML: `mhcac.text_dropout`, `mhcac.uncertain_policy`.

## Main classes

| Class | Dòng | Vai trò |
|---|---|---|
| `AbnormalityClassificationModel` | 207 | ★ Model chính |
| `DownsamplePatches` | 7 | Giảm 196 → `target_patch_count` cho luồng CNN |
| `CrossModalEmbeddingAlignment` | 66 | Chiếu visual/text/expert về `embed_dim` — **một lần, trên chuỗi đã merge** |
| `TrainablePositionalEncoding` | 103 | Pos-enc học được, **riêng cho từng encoder** |
| `ExpertTokenCrossAttention` | 112 | Một lớp attention; `expert_to_text_attention` là **teacher-gated** |

## Main methods

| Method | Doc | Vai trò |
|---|---|---|
| `AbnormalityClassificationModel.forward` (`:354`) | [📄](mhcac_12.py.methods/AbnormalityClassificationModel/forward.md) | ★ |
| `AbnormalityClassificationModel.__init__` (`:208`) | [📄](mhcac_12.py.methods/AbnormalityClassificationModel/__init__.md) | |
| `_resize_patch_sequence` (`:317`) | [📄](mhcac_12.py.methods/AbnormalityClassificationModel/_resize_patch_sequence.md) | ★ Đồng bộ số patch giữa encoder |

## Execution flow

```text
forward(shared_visual_tokens, text_embeddings, ...)
 │
 ├─ Bước 1 — validate: ndim==3, dim==visual_dim, spans không rỗng  → ValueError nếu sai
 ├─ Bước 2 — expert_tokens [14,D] → expand [B,14,D]
 ├─ Bước 3 — embedding_alignment(visual, text, expert)   ← MỘT phép chiếu cho tất cả
 ├─ Bước 4 — FOR name, span in sorted(spans, key=start):
 │             stream = visual_proj[:, span, :]
 │             biovil → cnn_downsampler   |  khác → _resize_patch_sequence
 │             → pos_enc(stream)          ← pos-enc RIÊNG mỗi encoder
 │          image_patches = cat(streams, dim=1)
 ├─ Bước 5 — FOR i, layer in attention_layers (6 lớp):
 │             có expert_to_text_attention → truyền text  (TEACHER)
 │             i == num_layers-2           → cộng lại expert token gốc đã chuẩn hóa
 │             còn lại                     → text_embeddings=None
 ├─ Bước 6 — expert_loss(...) → pooled, orth, contrastive, sparsity
 └─ Bước 7 — 14 × classifiers → stack → logits [B,14,3]
```

## Điểm thiết kế: một phép chiếu, nhiều view

Docstring `:364` nói rõ: `spans` **chỉ** dùng để cấp pos-enc riêng và resize riêng
cho từng encoder. **Phép chiếu về `embed_dim` xảy ra một lần, trên tensor đã merge.**

Nghĩa là MHCAC và Q-Former dùng chung một biểu diễn thị giác — không có đường
encode thứ hai. Trước đây hai nhánh tự chiếu và có thể trôi dạt khỏi nhau.

## Calls

`torch.nn`, `torch.nn.functional` (`adaptive_avg_pool2d`, `interpolate`).
**Không** import LAVIS, không import `vision_encoders`.

## Called by

`Blip2Qformer.forward` — **ba lần**: student (`:907`), teacher (`:925`),
anchor-only cho view consistency (`:965`).

## Data flow

```text
tokens [B,ΣP,1408] ─► embedding_alignment ─► visual_proj [B,ΣP,768]
                                                  │ slice theo spans
        ┌─────────────────┬───────────────────────┘
        ▼                 ▼
  biovil span        pubmedclip/swin span
        │                 │
  cnn_downsampler   _resize_patch_sequence
        │                 │
        └──── pos_enc ────┘
                │  cat
        image_patches [B, k·target_patch_count, 768]
                │
  expert_tokens [B,14,768] ── 6 lớp ──► [B,14,768] ──► 14×Linear ──► [B,14,3]
```

`target_patch_count` ⚠ **cần runtime verification** (đọc từ
`pos_enc.positional_encoding.size(1)`).

## Side effects

Không mutate state ngoài tham số. `attention_weights_list` được trả về, không lưu.

## Error / edge cases

| Tình huống | Hành vi |
|---|---|
| `tokens.ndim != 3` | `ValueError` nêu shape thật (`:373`) |
| `tokens.shape[-1] != visual_dim` | `ValueError` nêu cả hai giá trị (`:377`) |
| `spans` rỗng | `ValueError("shared_visual_tokens carries no encoder spans")` (`:381`) |
| `image_streams` rỗng | `ValueError` (`:407`) |
| Số patch không phải chính phương | `_resize_patch_sequence` rơi về nội suy tuyến tính 1-D |

## Related tests

`tests/test_stage1_objectives.py::test_mhcac_text_is_teacher_only_and_student_shape_matches_inference`
— **test quan trọng nhất cho file này**: xác nhận student không bao giờ thấy text.

## Related documentation

[ARCHITECTURE.md §2.4](../_meta/ARCHITECTURE.md#24-mhcac--phân-loại-bất-thường) ·
[`mhcac/_index.md`](_index.md) · [GLOSSARY: MHCAC](../_meta/GLOSSARY.md#mhcac)

## Developer notes

1. ⚠ **Typo `num_commmon_tokens`** (ba chữ `m`) là API công khai. Sửa = breaking change.
2. **`text_embeddings=None` là ranh giới teacher/student.** Đừng thêm default khác `None`.
3. **Lớp áp chót cộng lại expert token gốc** (`:~424`) — residual chống trôi bản sắc
   sau 6 lớp. Đổi `num_layers` sẽ đổi lớp nào làm việc này.
4. `_resize_patch_sequence` là chỗ Swin từng gây shape mismatch; notebook legacy 03
   từng vá nó bằng string replacement lúc runtime.
5. Sửa file này ảnh hưởng: Stage 1, Stage 2 mode `..._with_mhcac_prompt`, và mọi
   checkpoint cũ (shape state dict).

## Source relationships

- **Parent:** [`mhcac/_index.md`](_index.md)
- **Methods:** [`mhcac_12.py.methods/`](mhcac_12.py.methods/)
- **Related:** [`loss.py`](loss.py.doc.md) · [`shared_visual_tokens.py`](../vision_encoders/shared_visual_tokens.py.doc.md)

← [HOME](../../HOME.md)
