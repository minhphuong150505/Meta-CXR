> Source: `mhcac/mhcac_12.py:419-560`
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

## ★ Một phép chiếu, nhiều view — docstring `:426`
> *"Spans are used only to give each encoder its own within-stream positional
> encoding and its own resize to `target_patch_count`; the projection to
> `embed_dim` happens once, on the merged tensor, so MHCAC and META-Former share
> one visual representation."*

## ★ Hai nhánh xử lý stream
`self.stream_layouts` quyết định. Có layout → **native**: giữ nguyên số token,
tách token toàn cục, pos-enc riêng theo tên. Không có → **legacy**: mọi stream
bị đưa về `target_patch_count` và dùng chung một pos-enc.

## Execution flow
```text
Bước 1 validate: ndim==3 · dim==visual_dim · spans không rỗng   → ValueError
Bước 2 expert_tokens [14,D] → expand [B,14,D]
Bước 3 embedding_alignment(visual, text, expert)     ← MỘT phép chiếu
Bước 4 FOR name, span in sorted(spans, key=start):
          stream = visual_proj[:, span, :]
          IF stream_layouts.get(name)  ── NATIVE (cấu hình production) ──
            số token != layout.num_tokens             → ValueError
            capture bật → float32
            global_tokens = stream[:, :num_global]    ← CLS, không có toạ độ
            spatial       = stream[:, num_global:]
            stash (spatial, lưới vuông) nếu N chính phương  ← 14×14 / 7×7
            stream = cat([global_tokens, spatial])    ← ⚠ giữ spatial trên đồ thị
            → pos_enc[name](stream)                   ← pos-enc RIÊNG mỗi encoder
          ELSE  ───────────── LEGACY (swin/raddino bật) ─────────────
            biovil → float32 + stash trước cnn_downsampler → cnn_downsampler
            khác   → _resize_patch_sequence + float32 + stash
            → pos_enc(stream)                         ← MỘT pos-enc dùng chung
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

**Capture là opt-in và không detach.** `_last_cam_streams` được reset đầu mỗi
forward. Khi cờ tắt nó là `None`; khi bật nó chứa `(tensor, grid_hw)` để
`ExplanationLoss` lấy đạo hàm score theo đúng activation student.

## Data / Tensor flow
`[B,ΣP,1408]` → `[B,ΣP,768]` → slice/resize/pos-enc → `[B,k·target,768]` →
6 lớp → `[B,14,768]` → `[B,14,3]`

`target_patch_count` ⚠ **cần runtime verification** (đọc từ `pos_enc`).

## Error handling
Ba `ValueError` ở bước 1, mỗi cái **nêu giá trị thật**. Cộng `ValueError` nếu
`image_streams` rỗng (`:407`).

## Config dependencies
`mhcac.text_dropout` · `mhcac.uncertain_policy` · `num_layers=6`

`capture_streams` do `Blip2Qformer.forward` điều khiển theo lambda, epoch và mask;
không có config trực tiếp trong MHCAC.

## Tests
`tests/test_stage1_objectives.py::test_mhcac_text_is_teacher_only_and_student_shape_matches_inference` ·
`tests/test_explanation_loss.py::test_mhcac_only_keeps_cam_streams_while_capture_is_enabled`

## Modification risk
⚠ Đặt default khác `None` cho `text_embeddings` sẽ **phá teacher/student separation**
mà không có lỗi nào — student im lặng nhận text.

Không detach hoặc copy `_last_cam_streams`; Grad-CAM cần tensor gốc trong graph.
