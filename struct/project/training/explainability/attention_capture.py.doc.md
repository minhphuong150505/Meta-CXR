> Source: `training/explainability/attention_capture.py` (~560 dòng)
> Status: ✅ ACTIVE — đã chạy trên GPU thật 2026-08-30
> Last verified against source: 2026-08-30

# `training/explainability/attention_capture.py`

## Purpose

Module **duy nhất** trong package này chạm vào model sống. Mọi module khác thuần
đủ để test trên máy CPU chính là để module này nhỏ lại và là chỗ duy nhất có
forward thật.

Nó không sửa model quá thời gian một lời gọi: hook gỡ và cờ khôi phục trong
`finally`.

## ⚠ Bốn sự thật về bộ nhớ — mỗi cái đổi bằng một lần OOM

Đo trên RTX 5060 Ti 16 GB, `google/medgemma-1.5-4b-it`, 2026-08-30:

| | |
|---|---|
| **Không bao giờ** truyền `output_attentions=True` | Nó lan xuống vision tower SigLIP: 27 lớp × 16 head × 4096² fp32 ≈ **27 GiB giữ lại**. Dùng hook lên module attention của language model, chọn **theo tên** để vision tower không lọt vào do vô tình. |
| Vision tower **phải** dùng `sdpa` | Bỏ `output_attentions` vẫn OOM: eager SigLIP giữ softmax của chính nó cho backward. `attn_implementation={"text_config":"eager","vision_config":"sdpa"}` — transformers 4.53 chấp nhận dạng dict. |
| **Đóng băng mọi tham số**, dùng `torch.autograd.grad` | `.backward()` cấp gradient cho ~4 tỉ tham số ≈ **8 GiB** không ai đọc. Khi weight đã đóng băng thì `inputs_embeds` mang `requires_grad` — đó là thứ giữ cho graph tồn tại. |
| Xin **ít hàng logit nhất có thể** | Tensor đầy đủ là `[1, S, 262208]`. |

Đủ cả bốn: **peak 9.9 GiB / 15.5**, attention `[34, 8, S, S]`, gradient tới đủ
**34/34 lớp**, nonzero ở cả 34.

➡ **NF4 không cần và không nên dùng ở đây.** Rủi ro "backward qua NF4" của Mục 7
không phát sinh trên phần cứng này, chứ không phải được giảm nhẹ.

## ⚠ Split `train` bị RAISE, không phải warning

Transform của split train áp `RandomAffine`, nên bản đồ tính ở đó gắn với một
hình học chỉ tồn tại cho đúng một lần lấy mẫu và không thể chồng lại lên ảnh.
Một warning trong log không phải rào cản với người sắp đem hình đi công bố.

## ⚠ Gradient checkpointing bị TẮT, và việc đó được báo cáo

Nếu bật, forward mà hook quan sát là forward **tính lại** — attention bắt được
không nhất thiết là attention đã sinh ra logit đang giải thích. Đường này chỉ
inference và peak mới 9.9/15.5 GiB, nên không có gì đáng đánh đổi.

## ⚠ Cổng triệt tiêu: đòi CI bootstrap loại trừ 0, không phải một ngưỡng

Đây không phải lo xa. Trên **12 study test thật** (frontal, findings 30–90
token, seed 16), MedGemma bf16:

| điều kiện | mean ΔNLL | 95% CI | study tệ đi | established |
|---|---:|---|---:|---|
| zero hoá embedding thị giác | +0.0535 | [-0.0283, +0.1314] | 7/12 | **không** |
| ảnh của study khác | **+0.1868** | **[+0.0761, +0.3045]** | 10/12 | **có** |

Phép zero **vượt ngưỡng 0.05 mà vẫn không phân biệt được với không hiệu ứng**.
Gate chỉ-so-ngưỡng gọi đó là pass; `score_ablation` thì không.

➡ **Điều kiện "ảnh của study khác" là phép thử sắc hơn và là cái nên trích.**
Vector 0 không phải "không có thông tin" — nó là một điểm ngoài phân phối. Trên
ảnh tổng hợp không phải X-quang, zero hoá còn làm target **dễ đoán hơn**
(−0.84). Thay bằng ảnh bệnh nhân khác mới hỏi đúng câu cần hỏi — *có dùng ảnh
NÀY không* — và ở trong phân phối.

Bootstrap lấy lại mẫu **theo study, không theo token**: token trong cùng một báo
cáo không độc lập, khoảng tin cậy theo token sẽ hẹp giả tạo.

## Main items

| Item | Vai trò |
|---|---|
| `assert_split_allowed` | RAISE trên `train` |
| `disable_gradient_checkpointing` | Tắt, trả về nó từng bật hay không |
| `language_attention_modules` | Chọn **theo tên**, sắp theo chỉ số lớp (lexicographic sẽ đặt lớp 10 giữa 1 và 2) |
| `capture_attention` | Context manager, hook luôn được gỡ kể cả khi thân lỗi |
| `stack_captured` | `{lớp: [B,H,S,S]}` → `[L,H,S,S]` **fp32** (bf16 cho tổng hàng 1.0014) |
| `locate_visual_tokens` | Đếm, liền kề, khớp lưới — đều fail-loud |
| `per_token_nll` | Dịch một bước; vị trí bị mask trả `nan`, **không phải 0.0** |
| `gradient_weighted_layers` | `autograd.grad`; fallback **mặc định tắt** |
| `score_ablation` / `assert_visual_tokens_matter` | Cổng triệt tiêu có CI |
| `load_medgemma_for_explanation` | import transformers **lazy** |
| `build_visual_inputs` | Vision dưới `no_grad`, scatter vào embedding; `visual_features` để thay ảnh study khác |
| `teacher_forced_forward` | KV cache tắt, attention bắt qua hook |
| `qformer_cross_attention` | **Interface, RAISE** — xem [`_index.md`](_index.md) |

## Calls / Called by

- Calls: `torch`; `training.explainability.{rollout,projection}`; `transformers`
  **chỉ bên trong hàm**, nên module vẫn import được trên máy CPU.
- Called by: chưa có runner nào — chưa viết.

## Related tests

`tests/explainability/test_attention_capture.py` — 43 test, chạy CPU với một
transformer thay thế. Model giả **có một vision tower mồi kiểu sdpa**, nên "bắt
nhầm vision tower" là một test đỏ chứ không phải một lần OOM. Attention của nó
thật sự sinh ra output (`weights @ hidden`), vì bản trước tính weights song song
với output và `autograd` báo unused — đó là tính chất của model giả, không phải
của attention thật.

Tầng GPU (4 góc ô vuông tổng hợp, triệt tiêu trên study thật) **chưa tự động
hoá**; xem bảng số ở trên.
