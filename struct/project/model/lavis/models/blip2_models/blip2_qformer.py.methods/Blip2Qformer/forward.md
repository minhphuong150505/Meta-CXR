> Source: `model/lavis/models/blip2_models/blip2_qformer.py:878-1147`
> Status: ✅ ACTIVE

# `Blip2Qformer.forward(samples)`

## Located in

[`blip2_qformer.py`](../../blip2_qformer.py.doc.md)

## Purpose
Trái tim Stage 1. Một forward = encode + fusion + Q-Former (ITC/ITM/LM) + MHCAC ×3
+ explanation-aware Grad-CAM conditional + tổng hợp 12 loss.

## Signature
```python
def forward(self, samples) -> BlipOutput
```

## Parameters
### `samples`
- Type: `dict` từ `MIMIC_CXR_Dataset.collater` · Required: ✅
- Khóa: `image [B,3,448,448]`, `text_output list[str]`,
  `classification_labels [B,14]`, `classification_mask [B]`, `generation_mask [B]`,
  tùy chọn `explanation_mask [B,H,W]`, `explanation_mask_valid [B]`,
  và tùy chọn `aux_image [B,N,3,448,448]`, `aux_mask [B,N]`, `aux_view_ids [B,N]`,
  `anchor_view_id [B]`, `<enc>_feat`, `aux_<enc>_feat`.

## Returns
`BlipOutput` — `loss` (tổng có trọng số) + 12 thành phần + `classification_logits [B,14,3]`
+ `classification_mask [B]`.

## Local variables quan trọng
| Biến | Ý nghĩa |
|---|---|
| `shared_visual` | `SharedVisualTokens` — biểu diễn **duy nhất**, dùng bởi cả MHCAC và Q-Former |
| `image_embeds` | `shared_visual.tokens [B,ΣP,1408]` |
| `query_output` | Có `past_key_values` → tái dùng cho LM, không tính lại |
| `teacher_mask` | `classification_mask & generation_mask` — teacher cần **cả hai** |
| `student_logits` | Đường inference thật |
| `lambda_eff` | Trọng số explanation tại `current_epoch` |
| `cam_streams` | Activation student tạm; MHCAC state được clear trước khi đi tiếp |

## Execution flow
```text
1 gom cached / aux_cached từ samples
2 _encode_image_streams(...) → SharedVisualTokens
3 _batch_mask ×2 → classification_mask, generation_mask
4 Q-Former: query_tokens → bert(encoder_hidden_states=image_embeds, use_cache=True)
           tokenizer(text) → bert(text) → text_features
5 _image_text_contrastive → loss_itc, sim_i2t, sim_t2i
  _image_text_matching   → loss_itm
  _language_modeling     → loss_lm
  _update_itc_queue      (chỉ khi training)
6 tính lambda_eff; chỉ khi mask hợp lệ + study Positive mới bật capture
  mhcac(shared_visual, text_embeddings=None, labels, sample_mask)   ← STUDENT
  copy activation vào local rồi luôn tắt capture + clear _last_cam_streams
  cls_loss_fn(student_logits, ...)
7 IF local cam_streams có dữ liệu:
     ExplanationLoss(student_logits, labels, selected_streams, mask, valid)
8 IF teacher_mask.any() và (λ_teacher>0 hoặc λ_distill>0):
     mhcac(shared_visual, text_embeddings=<text>)                   ← TEACHER
     cls_loss_fn(teacher_logits, sample_mask=teacher_mask)
     soft_target_kl_loss(student, teacher, ...)
9 IF multi_view và aux_mask.any():
     mpc_loss_fn trên _last_prefusion_streams
     mhcac(anchor_shared, ...)  ← ANCHOR-ONLY  → view_consistency_loss
10 total = Σ λᵢ·lossᵢ + lambda_eff·L_exp  → BlipOutput
```

## Detailed logic
**Bước 4 — `use_cache=True` không phải tối ưu vặt.** `query_output.past_key_values`
được truyền thẳng sang `_language_modeling:810`. Query token được tính **một lần**
cho cả ITC và LM.

**Bước 6/8 — ranh giới teacher/student.** Khác biệt duy nhất là `text_embeddings`.
Đây là toàn bộ cơ chế giữ text khỏi đường inference.

**Explanation chỉ capture student.** Điều kiện được xác định trước lời gọi student
từ lambda, epoch, sự hiện diện của hai key mask, classification mask và nhãn
Positive. Teacher/anchor chạy sau khi cờ đã tắt. Nếu key chưa có (trạng thái Giai
đoạn 1), loss bằng zero nối graph và forward không crash.

**Bước 9 — nhánh view-consistency chạy MHCAC lần ba** trên `anchor_shared` (chỉ
anchor, không fuse). Nó tái dựng `SharedVisualTokens` từ `_last_prefusion_streams`,
áp `ln_vision` lại cho biovil (`:959`).

## Data / Tensor flow
Xem [DATA_FLOW.md §2.4](../../../../../../_meta/DATA_FLOW.md#24-trong-model).

## Side effects
⚠ Mutate `itc_image_queue` / `itc_text_queue` / `itc_queue_ptr` / `itc_queue_filled`
(chỉ khi `self.training`) · Reset `_last_prefusion_streams`, `_last_raddino_patches`

Capture CAM được clear trong `finally`, kể cả student forward raise; model không
giữ activation graph qua batch kế tiếp.

## Error handling
Batch không có mẫu hợp lệ → trả loss `0.0` **vẫn nối đồ thị autograd**
(`zero = x.sum()*0.0`), tránh DDP treo vì tham số không nhận gradient.

## Config dependencies
Mọi `loss.lambda_*`, `multi_view`, `mhcac.*`, `itc_queue_size`, `explanation.*`,
và epoch do runner cung cấp.

## Important conditions
```python
if teacher_mask.any() and (self.lambda_teacher_cls > 0 or self.lambda_distill > 0)  # :922
if self.multi_view and aux_mask is not None and aux_mask.any()                      # :946
if self.lambda_view_consistency > 0                                                 # :955
if self.explanation_loss_fn is not None                                             # gate cấu hình
if lambda_eff > 0 and explanation_mask is not None and mask_valid is not None        # gate batch
```
Các điều kiện này quyết định MHCAC chạy 1, 2 hay 3 lần và có dựng Grad-CAM hay không.

## Tests
`tests/test_stage1_objectives.py` (teacher-only) · `tests/test_multiview_losses.py` ·
`tests/test_explanation_loss.py` · `tests/test_loss_weight_gating.py`

## Modification risk
| Sửa | Ảnh hưởng |
|---|---|
| Truyền text vào lời gọi student | **Phá teacher/student separation** — text rò vào inference |
| Bỏ `zero = x.sum()*0.0` | DDP có thể treo |
| Đổi thứ tự `_update_itc_queue` | Queue chứa chính batch hiện tại → contrastive tự so với mình |
| Thêm loss mới | Phải thêm cả `lambda_` và trường trong `BlipOutput` |
| Không clear `_last_cam_streams` | Giữ graph qua batch → rò bộ nhớ |
| Gọi autograd khi lambda bằng 0 | Phá gate hiệu năng classification-only |
