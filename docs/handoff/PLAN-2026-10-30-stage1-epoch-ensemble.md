# PLAN 2026-10-30: gộp epoch của cùng một lần train pha 1b (snapshot ensemble), phép thử CPU trên val

**Trạng thái:** ĐÃ CHẠY 2026-10-30 — **DỪNG** theo quy tắc đã chốt (xem Execution report). User duyệt
2026-10-30 ("mình đồng ý"). Bước 1 chỉ dùng CPU, đọc các file dự đoán val có
sẵn. Không sửa code, không sửa YAML, không train.

## 1. Vì sao

- Audit 2026-10-29: dời logit hay dời ngưỡng không đổi gì dưới cutpoint. Muốn tăng recall mà precision giữ
  nguyên thì phải xếp hạng tốt hơn.
- ASL, mở khóa encoder (U5 và L42) đều làm macro AP giảm. Gộp nhiều checkpoint của cùng một lần train là
  cách phổ biến để xếp hạng ổn định hơn mà không đổi kiến trúc. Dạng triển khai cuối, nếu đi tiếp, là trung
  bình trọng số (SWA): vẫn một mô hình META-CXR, chi phí suy luận như cũ.
- **Bằng chứng ngược chiều:** run cũ 2026-08-20 (cách chấm nhị phân, 10 epoch, val gần như phẳng) gộp epoch
  4, 6, 9 không hơn epoch 9. Vì vậy chưa rõ có lợi, phải thử.

## 2. Arm (chỉ val, n = 1.808)

| Run | Cơ sở (best theo `macro_recall`) | Bản gộp |
|---|---|---|
| C42 = `run_20261005_paper/phase1b` | `val_predictions_epoch_best.npz` (epoch 2) | trung bình xác suất ba lớp của epoch 2, 3, 4 |
| L42 = `unf_full_20261030/L42/phase1b` | `val_predictions_epoch_best.npz` (epoch 4) | trung bình xác suất ba lớp của epoch 2, 3, 4 |

Tập epoch {2, 3, 4} chốt trước, không dò. Hai run là hai lần kiểm tra độc lập.

## 3. Quy tắc đọc, chốt TRƯỚC khi chạy

- **Chỉ số quyết định:** macro AP "có bệnh" (P hoặc U so với N, trên `s = p_pos/(p_pos+p_neg)`), trung bình
  14 finding. Δ = bản gộp − cơ sở, tính riêng mỗi run. Bootstrap ghép cặp theo study, 1.000 lần, seed 16.
- **ĐI TIẾP sang bước 2** khi đạt đủ:
  1. Δ > 0 ở cả hai run;
  2. trung bình Δ của hai run > +0,0042;
  3. rào chắn ở cả hai run: Δ wF1 ≥ −0,0027 và Δ F1-5 ≥ −0,0060 (cutpoint cross-fit).
- **Mọi trường hợp còn lại:** dừng, ghi kết quả âm.
- **Chỉ để mô tả:** AUROC, RP12, recall/precision P/U dưới cutpoint và argmax, AP theo finding hiếm; C42 best
  gộp với L42 best (hai mô hình, mốc trên, không báo cáo).
- Cùng một mô hình nên NB seed không áp dụng trực tiếp; vẫn dùng 0,0042 làm ngưỡng để thận trọng.

## 4. Bước 2 (chỉ khi ĐI TIẾP, hỏi user riêng)

Trung bình trọng số `checkpoint_best` (epoch 2) và `checkpoint_4` của C42 (chỉ hai checkpoint còn lưu), chấm
val, khoảng 10 phút GPU. Trước khi chấm test một lần phải giải quyết: mô hình đang báo cáo là bản sau pha 1c.

## 5. Giới hạn

- Chỉ val; tập epoch cố định; dự đoán gộp (3 lần suy luận) chỉ là phép thử thay cho trung bình trọng số.
- Không cho biết gì về pha 1c.

## Execution report

**Kết luận theo quy tắc đã chốt: DỪNG.** Δ macro AP trung bình −0,0003, không đạt điều kiện 1 và 2. Không làm
bước 2.

**Chạy:** `~/ens_probe.py` trên CPU của máy train (checkout `~/Meta-CXR` @ `4338f16`), 1 phút, log
`~/ens_probe_20261030.log`. Cùng file REF (pha 1c best) cho precision mốc như các probe trước.

**Bản gộp epoch 2–4 − best, val n = 1.808:**

| Chỉ số | C42 | L42 |
|---|---|---|
| **macro AP "có bệnh"** (quyết định) | **+0,0001** [−0,0050; +0,0048] | **−0,0007** [−0,0027; +0,0015] |
| AUROC "có bệnh" | +0,0011 | +0,0004 |
| RP12 | +0,0136 | +0,0162 |
| wF1 (rào chắn) | +0,0012 | −0,0008 |
| F1-5 (rào chắn) | +0,0007 | −0,0000 |
| Recall / precision P/U dưới cutpoint | +0,0037 / +0,0036 | +0,0128 / −0,0122 |
| Recall / precision P/U dưới argmax | −0,0032 / −0,0021 | −0,0076 / +0,0066 |

Trung bình Δ macro AP hai run: −0,0003 [−0,0028; +0,0022] (bootstrap ghép cặp theo study, 1.000 lần, seed 16).
Rào chắn đạt.

**Chỉ mô tả:**

- AP theo finding dao động hai chiều và không nhất quán giữa hai run. Ví dụ Lung Lesion +0,012 (C42) nhưng
  −0,018 (L42); Pleural Other −0,045 (C42) nhưng +0,001 (L42).
- RP12 tăng ở cả hai run (+0,014 / +0,016), nhưng đó là một điểm trên đường PR, AP thì không đổi.
- Mốc trên C42 best + L42 best (hai mô hình): macro AP −0,0035, không hơn C42 một mình.

**Đọc kết quả:** các epoch 2–4 của cùng một lần train pha 1b xếp hạng gần như giống nhau, nên gộp chúng không
cải thiện xếp hạng. Giống kết quả 2026-08-20 ở run cũ. Trung bình trọng số (bước 2) không có lý do để thử.

**Giới hạn:** chỉ val; một tập epoch cố định; dự đoán gộp thay cho trung bình trọng số.
