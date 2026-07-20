# Prompt: Mở rộng pipeline META-CXR từ single-view sang multi-view

> Đưa toàn bộ nội dung dưới đây cho Claude Code. Đây là đặc tả thay đổi kiến trúc, không phải yêu cầu viết lại từ đầu — mục tiêu là **thêm khả năng multi-view mà vẫn giữ tương thích ngược 100% với checkpoint single-view hiện có**.

---

## 0. Bối cảnh & mục tiêu

Tôi có một pipeline sinh báo cáo X-quang ngực tên **META-CXR** đang chạy ổn định ở chế độ **single-view** (một ảnh một report). Kiến trúc hiện tại:

```
1 ảnh CXR
  → Combined Vision Encoder: 3 encoder frozen (VE_SWIN, VE_RES, VE_ViT),
    mỗi encoder qua một lớp FC riêng, rồi concat theo chiều sequence
    → 1 chuỗi token thị giác  V ∈ [B, P, D]   (P ≈ tổng số patch của 3 encoder, D = 768)
  → chuỗi này rẽ vào 2 nhánh:
      (a) MHCAC (multi-head cross-attention classification, dùng expert tokens) → phân loại abnormality (P/N/U)
      (b) META-Former (kiến trúc kiểu Q-Former: Self-Attn → Cross-Attn mỗi block xen kẽ → FFN, ×N, với learnable queries)
          → FC → cộng với soft-prompt abnormality → LM (frozen) → sinh report
```

**Mục tiêu:** cho phép một *study* gồm **nhiều ảnh** (anchor + 0..n auxiliary views, ví dụ PA/AP/lateral) được hợp nhất thành **một chuỗi token thị giác duy nhất** trước khi vào MHCAC và META-Former, sao cho:

- Study có 1 ảnh vẫn chạy y hệt như hiện tại (không regression).
- Study có nhiều ảnh được khai thác thông tin bổ sung qua cross-attention.
- **Không sửa** MHCAC, META-Former, LM, hay bất kỳ nhánh output nào — chúng chỉ nhận "một chuỗi token tốt hơn".
- Có thể **nạp checkpoint single-view đã train** và fine-tune tiếp, không train lại từ đầu.

---

## 1. Thành phần MỚI cần thêm: `ViewFusionModule`

Thêm một module mới, chèn vào **ngay sau bước concat của Combined Vision Encoder, trước khi chuỗi token rẽ vào MHCAC và META-Former**.

### 1.1. Chữ ký & I/O

```python
class ViewFusionModule(nn.Module):
    def __init__(self, dim=768, num_heads=8, ffn_ratio=4,
                 num_view_types=4, dropout=0.1, num_blocks=1):
        ...
    def forward(self, anchor, aux, aux_mask=None,
                anchor_view_id=None, aux_view_ids=None):
        """
        anchor:          [B, P, D]           token đã concat của ảnh chính
        aux:             [B, N*P, D]          token đã concat của N view phụ (đã pad tới N_max)
        aux_mask:        [B, N*P] bool        True = vị trí thật, False = padding
        anchor_view_id:  [B]                  loại view của anchor (0..3)
        aux_view_ids:    [B, N]               loại view của từng auxiliary
        return:          [B, P, D]            chuỗi hợp nhất, SHAPE GIỐNG HỆT anchor
        """
```

> **Ràng buộc bất biến quan trọng:** output PHẢI có shape `[B, P, D]` đúng bằng `anchor`. Đây là điều kiện để MHCAC/META-Former không phải sửa gì.

### 1.2. Cấu trúc bên trong mỗi block (theo thứ tự)

1. **View-type embedding.** Có `nn.Embedding(num_view_types, dim)` với 4 loại: `PA / AP / lateral / unknown`. Cộng embedding tương ứng vào token của anchor và vào token của từng auxiliary theo `view_ids`. (Broadcast embedding của mỗi view lên toàn bộ P token của view đó.)

2. **Multi-head cross-attention.** Anchor là Query, auxiliary là Key & Value. Dùng `W_Q, W_K, W_V` riêng biệt (không tie), 8 heads, head_dim = dim/num_heads. Áp `aux_mask` dưới dạng additive mask (`-inf` ở vị trí padding) trước softmax.

3. **Add & Norm (skip từ anchor):** `H = LayerNorm(anchor + Dropout(W_O @ attn_out))`.

4. **FFN:** `dim → dim*ffn_ratio → dim`, activation **GELU**, dropout.

5. **Add & Norm:** `out = LayerNorm(H + Dropout(FFN(H)))`.

Mặc định `num_blocks=1` (tối đa 2). Không stack sâu.

### 1.3. Ba chi tiết kỹ thuật BẮT BUỘC

- **Zero-init output projection `W_O`.** Khởi tạo trọng số (và bias) của `W_O` = 0. Nhờ đó ở bước forward đầu tiên, `attn_out` đóng góp = 0, module trở thành **identity** (`out ≈ LayerNorm(anchor)`). Đây là điều kiện để nạp checkpoint single-view và fine-tune mượt, không phá biểu diễn đã học.

- **Gate cho study 1 ảnh (n=0).** Nếu một sample không có auxiliary (toàn bộ `aux_mask` = False, hoặc `aux` rỗng), **bỏ qua hoàn toàn nhánh cross-attention** cho sample đó và trả về `anchor` nguyên vẹn (chỉ qua LayerNorm nếu muốn giữ nhất quán). TUYỆT ĐỐI không để softmax chạy trên một hàng toàn `-inf` → sẽ ra NaN. Xử lý theo batch: có thể tách sample n=0 ra, hoặc dùng safe-softmax (thay hàng toàn -inf bằng 0 sau softmax).

- **View dropout (chỉ khi training).** Với xác suất `p_view_drop=0.2`, ngẫu nhiên loại bỏ một số auxiliary view khỏi một study trong lúc train (cập nhật `aux_mask` tương ứng). Buộc model không phụ thuộc cứng vào multi-view, đảm bảo robust khi inference gặp study thiếu view.

---

## 2. Thay đổi ở Data Pipeline / Dataset

Hiện tại dataset trả về (ảnh, report, nhãn) theo từng ảnh. Cần đổi sang **gom theo study**.

### 2.1. Gom theo study

- Nhóm các ảnh cùng một `study_id` (MIMIC-CXR có sẵn `study_id`; mỗi study có metadata `ViewPosition`).
- Chọn **anchor** theo ưu tiên: `PA > AP > lateral > còn lại`. Nếu study có nhiều ảnh cùng loại ưu tiên nhất, chọn ảnh đầu tiên. Các ảnh còn lại là auxiliary.
- Report gắn với study (không phải với từng ảnh) — giữ nguyên report hiện có của study.

### 2.2. Collate function

- Pad số lượng auxiliary trong mỗi batch tới `N_max` của batch đó (không cần cố định toàn cục).
- Sinh `aux_mask` bool đánh dấu padding.
- Sinh `anchor_view_id` và `aux_view_ids` (map `ViewPosition` → {PA:0, AP:1, lateral:2, unknown:3}).
- Trả thêm các field này mà **không phá cấu trúc batch cũ** (các field cũ giữ nguyên tên).

### 2.3. Chế độ tương thích ngược

- Thêm cờ config `multi_view: bool`. Khi `multi_view=False`, dataset hoạt động **y hệt bản gốc** (mỗi ảnh một mẫu, không gom study), và `ViewFusionModule` bị bypass. Đây là chế độ để reproduce baseline và so sánh.

---

## 3. Thay đổi ở Forward của model chính

Trong `forward` của model tổng:

```python
# BƯỚC HIỆN CÓ (giữ nguyên): chạy Combined Vision Encoder trên MỌI ảnh của study,
# dùng chung trọng số (encoder đã frozen nên không tốn thêm param).
tok_anchor = combined_vision_encoder(anchor_img)          # [B, P, D]
tok_aux    = combined_vision_encoder(aux_imgs_flat)       # [B, N*P, D]  (chạy batched)

# BƯỚC MỚI: hợp nhất
if config.multi_view:
    fused = view_fusion(tok_anchor, tok_aux, aux_mask,
                        anchor_view_id, aux_view_ids)      # [B, P, D]
else:
    fused = tok_anchor

# BƯỚC HIỆN CÓ (giữ nguyên): rẽ vào 2 nhánh — KHÔNG SỬA GÌ
cls_out    = mhcac(fused, expert_tokens, ...)
report_out = meta_former_then_lm(fused, ...)
```

Lưu ý: encoder chạy trên auxiliary nên được gọi **một lần theo batch** (flatten `[B, N, 3, H, W] → [B*N, 3, H, W]`) để tận dụng GPU, sau đó reshape lại. Vì encoder frozen, có thể bọc trong `torch.no_grad()` như cách anchor đang làm.

---

## 4. Thay đổi ở Loss (tuỳ chọn, nên có)

Thêm hai loss phụ, mỗi loss có trọng số riêng trong config (mặc định nhỏ, có thể tắt = 0):

### 4.1. Multi-positive Contrastive (MPC) — cấp biểu diễn thị giác

- Trên output của Combined Vision Encoder (trước fusion), pool mỗi ảnh thành 1 vector.
- Với mỗi anchor, các auxiliary **cùng study** là positive, ảnh của study khác trong batch là negative.
- Loss = InfoNCE với nhiều positive (multi-positive), tự thích ứng số lượng positive thay đổi theo study.
- Trọng số config: `lambda_mpc` (mặc định 0.1).

### 4.2. View-consistency cho MHCAC — cấp phân loại (ĐIỂM RIÊNG của kiến trúc này)

- Cùng một study, các view khác nhau phải cho ra **cùng tập abnormality**.
- Chạy MHCAC riêng trên biểu diễn từng view (hoặc trên anchor-only vs fused), rồi phạt sự khác biệt giữa các phân phối dự đoán expert-token (ví dụ KL hoặc MSE giữa attention-pooling distribution / logit của các view).
- Trọng số config: `lambda_view_consistency` (mặc định 0.05).
- Chỉ áp dụng khi study có ≥ 2 view.

> Loss chính (report generation + classification hiện tại) **giữ nguyên**. Hai loss trên chỉ cộng thêm.

---

## 5. Config cần thêm

```yaml
multi_view: true            # false = chạy y hệt baseline single-view
view_fusion:
  dim: 768
  num_heads: 8
  ffn_ratio: 4
  num_blocks: 1
  num_view_types: 4
  dropout: 0.1
  p_view_drop: 0.2          # view dropout khi train
loss:
  lambda_mpc: 0.1
  lambda_view_consistency: 0.05
data:
  anchor_priority: [PA, AP, lateral]
  max_aux_views: 3          # cắt bớt nếu study có quá nhiều view
```

---

## 6. Checklist nghiệm thu (yêu cầu Claude Code tự kiểm)

1. `multi_view=False` → output số học **trùng khớp** bản gốc trên vài batch mẫu (regression test).
2. `ViewFusionModule` với `W_O` zero-init → ở step 0, `fused ≈ LayerNorm(anchor)` (sai khác < 1e-5 ngoài ảnh hưởng LN).
3. Study 1 ảnh (aux rỗng) → **không NaN**, không lỗi shape.
4. Study nhiều ảnh → output shape vẫn `[B, P, D]`.
5. Có thể `load_state_dict(strict=False)` từ checkpoint single-view (chỉ ViewFusionModule là param mới).
6. View dropout chỉ active khi `model.training == True`.
7. Encoder auxiliary chạy trong `no_grad` và batched (không loop từng ảnh).

---

## 7. Thứ tự triển khai đề xuất

1. `ViewFusionModule` + unit test (identity khi zero-init, không NaN khi n=0).
2. Sửa Dataset/Collate sang gom-theo-study + cờ `multi_view`.
3. Nối vào forward model chính, chạy regression test `multi_view=False`.
4. Thêm MPC loss, rồi view-consistency loss (bật dần bằng lambda).
5. Fine-tune từ checkpoint single-view; theo dõi metric hiện có (BLEU/ROUGE/RadGraph cho report, F1/AUC cho classification) — kỳ vọng không tụt ở step đầu, cải thiện dần trên tập study nhiều view.

---

## 8. Nguyên tắc xuyên suốt (nhấn mạnh cho Claude Code)

- **Không refactor thứ không liên quan.** Chỉ thêm/nối, giữ nguyên tên hàm, chữ ký, và hành vi của MHCAC / META-Former / LM.
- **Tương thích ngược là ưu tiên số 1.** Mọi thay đổi phải có đường về baseline qua config.
- **Ưu tiên đơn giản.** 1 block fusion, zero-init, gate cứng cho n=0 — không thêm cơ chế phức tạp trừ khi cần.
- Nếu có điểm mơ hồ (ví dụ tên biến/đường dẫn thực tế trong repo), **hỏi lại trước khi đoán**, hoặc nêu rõ giả định đã dùng.
