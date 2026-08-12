> Source: `mhcac/mhcac_12.py:354-448`
> Status: ✅ ACTIVE

# `AbnormalityClassificationModel.forward(...)`

## Located in

[`mhcac_12.py`](../../mhcac_12.py.doc.md)

## Purpose
`SharedVisualTokens` (+ tùy chọn text) → `logits [B,14,3]` và ba loss phụ.

## Signature
```python
def forward(self, shared_visual_tokens, text_embeddings=None,
            text_attention_mask=None, labels=None, sample_mask=None)
    -> (logits, attention_weights_list, contrastive_loss, orth_loss, sparsity_loss)
```

## ★ `text_embeddings` là ranh giới teacher/student
`None` → **student** (đường inference). Có giá trị → **teacher** (chỉ lúc train).

## ★ Một phép chiếu, nhiều view — docstring `:364`
> *"Spans are used only to give each encoder its own within-stream positional
> encoding and its own resize to `target_patch_count`; the projection to
> `embed_dim` happens once, on the merged tensor, so MHCAC and META-Former share
> one visual representation."*

## Execution flow
```text
Bước 1 validate: ndim==3 · dim==visual_dim · spans không rỗng   → ValueError
Bước 2 expert_tokens [14,D] → expand [B,14,D]
Bước 3 embedding_alignment(visual, text, expert)     ← MỘT phép chiếu
Bước 4 FOR name, span in sorted(spans, key=start):
          stream = visual_proj[:, span, :]
          biovil → cnn_downsampler   |  khác → _resize_patch_sequence
          → pos_enc(stream)                          ← pos-enc RIÊNG mỗi encoder
       image_patches = cat(streams, dim=1)
Bước 5 FOR i, layer in attention_layers (6 lớp):
          layer.expert_to_text_attention không None → truyền text   (TEACHER)
          i == num_layers-2 → expert += expert_token_norm(expert_tokens gốc)
          còn lại           → text_embeddings=None
Bước 6 expert_loss(...) → pooled, orth, contrastive, sparsity
Bước 7 14 × classifiers → stack → logits [B,14,3]
```

## Detailed logic
**Bước 4 — `sorted(spans, key=start)`** đảm bảo thứ tự stream ổn định, khớp với
thứ tự `SharedVisualTokenProjector` đã nối.

**Bước 5, lớp áp chót** — cộng lại expert token khởi tạo đã chuẩn hóa. Residual
chống việc token trôi mất bản sắc sau 6 lớp attention. ⚠ Đổi `num_layers` đổi lớp
nào làm việc này.

## Data / Tensor flow
`[B,ΣP,1408]` → `[B,ΣP,768]` → slice/resize/pos-enc → `[B,k·target,768]` →
6 lớp → `[B,14,768]` → `[B,14,3]`

`target_patch_count` ⚠ **cần runtime verification** (đọc từ `pos_enc`).

## Error handling
Ba `ValueError` ở bước 1, mỗi cái **nêu giá trị thật**. Cộng `ValueError` nếu
`image_streams` rỗng (`:407`).

## Config dependencies
`mhcac.text_dropout` · `mhcac.uncertain_policy` · `num_layers=6`

## Tests
`tests/test_stage1_objectives.py::test_mhcac_text_is_teacher_only_and_student_shape_matches_inference`

## Modification risk
⚠ Đặt default khác `None` cho `text_embeddings` sẽ **phá teacher/student separation**
mà không có lỗi nào — student im lặng nhận text.
