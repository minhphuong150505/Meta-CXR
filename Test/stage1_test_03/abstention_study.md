# Abstention / selective prediction — nghiên cứu khả thi

Đo trên `run_20260820_ft/checkpoint_best.pth` (epoch 9, mô hình cuối cùng), test 3,269 study,
`study_presence` + `marginal_presence`, ngưỡng calibrate trên val. 12 nhãn (bỏ meta).

**Kết luận: abstention dựa trên độ tin cậy của model KHÔNG dùng được ở điểm vận hành
theo từng nhãn. Nhưng nghiên cứu này tìm ra một cải tiến khác, miễn phí: hiệu chuẩn
điểm số theo từng nhãn.**

---

## 1. Đường cong risk–coverage (bỏ qua theo từng ô)

| coverage | precision | recall hiệu dụng | precision nếu bỏ ngẫu nhiên |
|---|---:|---:|---:|
| 100% | 0.2931 | 0.5373 | 0.2931 |
| 80% | 0.3476 | 0.3866 | 0.2949 |
| 60% | 0.3909 | 0.2754 | 0.2937 |
| 40% | 0.4387 | 0.1891 | 0.2878 |
| 10% | 0.5037 | 0.0905 | 0.2827 |

Hơn bỏ ngẫu nhiên, nhưng precision 0.5037 ở coverage 10% **trùng với trần 0.52 đã đo
được từ phép quét sàn precision ngày 2026-08-20** — vì đây là cùng một phép toán.
Bỏ qua một ô làm nó thành âm tính, nên "chỉ gọi dương khi xa ngưỡng" = nâng ngưỡng.

## 2. Bỏ qua theo cả study — gần như vô dụng

| coverage | precision | recall | ngẫu nhiên |
|---|---:|---:|---:|
| 100% | 0.2931 | 0.5373 | 0.2931 |
| 60% | 0.3176 | 0.4766 | 0.2916 |
| 30% | 0.3347 | 0.3748 | 0.2871 |

Bỏ 40% ca khó nhất chỉ nâng precision 0.293 → 0.318.

## 3. Sáu tín hiệu độ tin cậy — không cái nào thắng nâng ngưỡng

Precision@K macro, K = tỉ lệ số ca được gọi dương tính ở ngưỡng calibrate.

| quy tắc | K=50% | K=25% | K=15% |
|---|---:|---:|---:|
| **`s = m × q_pos`** (≡ nâng ngưỡng) | **0.3582** | **0.4007** | 0.4129 |
| `q_pos` một mình | 0.3190 | 0.3561 | 0.3656 |
| `m` một mình | 0.3514 | 0.3903 | 0.4070 |
| `s × (1 − entropy(q))` | 0.3510 | 0.3946 | 0.4046 |
| `s` trừ phạt entropy | 0.3529 | 0.3919 | **0.4201** |
| `min(m, q_pos)` | 0.3548 | 0.3983 | 0.4099 |

`m` và `q` là hai đầu ra của cùng một mạng đọc cùng đặc trưng — "bất đồng" giữa
chúng không phải bằng chứng độc lập.

## 4. Bộ dự đoán lỗi có huấn luyện (logistic regression, fit trên val)

Mục tiêu: dự đoán *"lời gọi dương tính này có đúng không"*. 9,884 ô được gọi dương
tính trên test, 36.9% đúng. Đặc trưng: `s`, `m`, `q[3]`, entropy, margin, cộng bối
cảnh study (thống kê trên 11 nhãn còn lại).

**AUROC gộp chung trên test:**

| | AUROC |
|---|---:|
| chỉ điểm số `s` | 0.6693 |
| LR, đặc trưng của ô | **0.7316** |
| LR + bối cảnh study | 0.7309 |

Trông như thành công. **Không phải.** Truy nguyên theo từng nhãn:

| nhãn | n gọi | theo `s` | LR | Δ |
|---|---:|---:|---:|---:|
| Pleural Effusion | 1,364 | 0.7460 | 0.7470 | +0.0011 |
| Cardiomegaly | 1,851 | 0.6524 | 0.6530 | +0.0006 |
| Enlarged Cardiomediastinum | 429 | 0.5814 | 0.6034 | +0.0219 |
| Pneumothorax | 188 | 0.6153 | 0.5665 | −0.0488 |
| Fracture | 297 | 0.5818 | 0.5607 | −0.0211 |
| **trung bình 12 nhãn** | | **0.6149** | **0.6078** | **−0.0071** |

**Trong từng nhãn, bộ dự đoán lỗi còn tệ hơn chính điểm số** — thua trên 8/12 nhãn.
Toàn bộ mức tăng AUROC gộp đến từ việc xếp hạng **giữa các nhãn**.

## 5. Và mức tăng đó chỉ là hiệu chuẩn

Precision@K **gộp chung** (một điểm vận hành duy nhất cho cả 12 nhãn):

| quy tắc | K=50% | K=35% | K=25% | K=15% |
|---|---:|---:|---:|---:|
| `s` thô | 0.4646 | 0.5114 | 0.5617 | 0.6217 |
| LR đầy đủ, 12 đặc trưng | 0.5229 | 0.5721 | 0.6119 | 0.6871 |
| **`s` hiệu chuẩn theo từng nhãn** | **0.5239** | **0.5776** | **0.6305** | **0.7020** |

AUROC gộp: `s` thô 0.6693 · LR 12 đặc trưng 0.7316 · **`s` hiệu chuẩn 0.7392**.

Một hồi quy logistic **một chiều trên chính `s`**, fit riêng cho từng nhãn, **thắng**
mô hình 12 đặc trưng. Không có thông tin nào ngoài điểm số. Vấn đề đơn thuần là 12
nhãn không nằm trên cùng một thang đo.

---

## Kết luận

**Với điểm vận hành theo từng nhãn — cái đang dùng — abstention không mua được gì.**
Lỗi của model không dự đoán được từ chính đầu ra của nó. Muốn abstention có giá trị,
`g(x)` phải đọc thông tin `f(x)` không có: đầu chọn huấn luyện riêng (SelectiveNet),
lấy mẫu hậu nghiệm trên trọng số (MC dropout), hoặc mô hình hoá bác sĩ (learning to
defer). Kỳ vọng nên thấp: một LR 12 đặc trưng có cả bối cảnh study đã rút ra được
**đúng 0** thông tin trong từng nhãn.

**Nhưng có một kết quả dùng được ngay: hiệu chuẩn theo từng nhãn.** Miễn phí (fit
trên val, không train lại), và với bất kỳ quy trình nào xếp hạng phát hiện **xuyên
nhãn** — ưu tiên danh sách đọc, phân loại mức khẩn — nó nâng precision ở coverage 25%
từ **0.5617 lên 0.6305**. Đó là ứng dụng lâm sàng thực tế nhất mô hình này hỗ trợ
được ở mức hiệu năng hiện tại: không phải chẩn đoán tự động, mà là **sắp thứ tự ưu tiên**.

Tệp số liệu: `abstention_risk_coverage.json`.
