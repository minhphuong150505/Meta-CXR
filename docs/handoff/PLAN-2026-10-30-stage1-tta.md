# PLAN 2026-10-30: test-time augmentation (TTA) cho Stage 1, phép thử trên val

**Trạng thái:** ĐÃ CHẠY 2026-10-31 — **DỪNG** theo quy tắc đã chốt (Δ trung bình +0,0035 < +0,0042; xem
Execution report). User duyệt 2026-10-30 ("ok"). Chỉ suy luận trên val, không train, không sửa YAML.

## 1. Vì sao

- Bốn hướng làm mô hình xếp hạng tốt hơn đều thất bại (ASL, mở khóa encoder ở pilot và ở budget đầy đủ, gộp
  epoch). Gộp epoch thất bại vì các epoch của cùng một lần train dự đoán gần như giống nhau. TTA tạo khác biệt
  ở phía ảnh đầu vào, là một nguồn biến thiên khác.
- Chỉ thay đổi lúc suy luận. Mô hình vẫn là META-CXR, vẫn đủ ba lớp.
- Ghép điểm theo phân cấp nhãn đã được kiểm tra trên nhãn val và bị loại: nhãn không tuân theo phân cấp (388/436
  study có Cardiomegaly lại có Enlarged Cardiomediastinum là Negative).
- Kỳ vọng: lợi ích thường nhỏ. Chưa rõ, nên phải thử.

## 2. Thay đổi code (opt-in, mặc định tắt)

- `model/lavis/data/tta.py` (chỉ PIL): `resolve_tta_affine`, `apply_tta_affine`. Một phép xoay + phóng cố định
  quanh tâm, một lần lấy mẫu bilinear, góc ngoài ảnh tô 0.
- `MIMIC_CXR_Dataset`: đọc `datasets.mimic_cxr.vis_processor.eval.tta: {degrees, scale}`; áp lên ảnh gốc TRƯỚC
  tiền xử lý của mọi encoder. Khoá vắng → giống từng byte. Từ chối split `train`, `run.feature_cache_dir`, và
  khi có `explanation_mask`.
- `tests/test_tta_affine.py` (12 test). Cập nhật `struct/`, CLAUDE.md, README.
- Trên máy train chạy từ một bản clone riêng `~/Meta-CXR-tta` (từ `~/Meta-CXR` @ `4338f16`, chép ba file trên),
  để `~/Meta-CXR` giữ nguyên cho đến khi user cho commit.

## 3. Mô hình và lượt chấm (val, n = 1.808)

| Mô hình | Checkpoint |
|---|---|
| C42 | `run_20261005_paper/phase1b/.../checkpoint_best.pth` (cơ sở của mọi pilot) |
| R1C | `run_20261005_paper/phase1c/.../checkpoint_best.pth` (mô hình đang báo cáo) |

Năm lượt mỗi mô hình, chốt trước, không dò: ảnh gốc (`id`), xoay +5° và −5°, phóng 0,95 và 1,05. Không lật ngang
(sai giải phẫu). TTA = trung bình xác suất ba lớp của năm lượt.

**Cổng G1:** lượt `id` của mỗi mô hình phải giống từng bit file `val_predictions_epoch_best.npz` đã có (sau khi
sắp theo `sample_key`). Không khớp → dừng, báo user. Log phải có đúng một dòng `Loading evaluation weights from`.

**Cổng G2:** mỗi lượt có biến đổi phải khác lượt `id` (nếu giống hệt, khoá `tta` không được đọc) → dừng.

## 4. Quy tắc đọc, chốt TRƯỚC khi chạy

- **Chỉ số quyết định:** macro AP "có bệnh" (P hoặc U so với N, trên `s = p_pos/(p_pos+p_neg)`), trung bình 14
  finding. Δ = TTA − lượt `id`, mỗi mô hình. Bootstrap ghép cặp theo study, 1.000 lần, seed 16.
- **ĐI TIẾP** (chấm test một lần, hỏi user riêng) khi đạt đủ:
  1. Δ > 0 ở cả hai mô hình;
  2. trung bình Δ > +0,0042;
  3. rào chắn ở cả hai: Δ wF1 ≥ −0,0027 và Δ F1-5 ≥ −0,0060 (cutpoint cross-fit).
- **Mọi trường hợp còn lại:** dừng, ghi kết quả âm.
- **Chỉ để mô tả:** AUROC, RP12, recall/precision P/U dưới cutpoint và argmax, AP theo finding hiếm, macro AP
  của từng lượt đơn lẻ.
- Cùng một mô hình nên NB seed không áp dụng trực tiếp; vẫn dùng 0,0042 làm ngưỡng để thận trọng.

**Chi phí:** 10 lượt chấm val, ước tính 30–50 phút GPU.

## 5. Giới hạn

- Chỉ val; một bộ biến đổi cố định; TTA tốn 5 lần suy luận.
- Biến đổi áp lên ảnh gốc nên Swin (pad thành vuông) thấy góc tô 0 khi xoay.

## Execution report

**Kết luận theo quy tắc đã chốt: DỪNG.** Δ macro AP dương ở cả hai mô hình và khoảng tin cậy nằm trên 0, nhưng
trung bình +0,0035 không vượt ngưỡng +0,0042 (điều kiện 2). Không chấm test.

**Chạy:**

- Lần 1 (2026-10-30 23:53) dừng ở lượt `C42 r+5` sau 11 giây, do cổng an toàn trong code: YAML vẫn trỏ
  `model.explanation.mask_cache_dir` tới cache mask (loss giải thích là 0 từ 2026-08-17), nên dataset nạp mask
  và `_row_visual` từ chối TTA khi có mask. G1 của lần 1 đã qua. Thư mục được giữ ở `~/tta_20261030_abort1`.
- Khi đánh giá, đường "có mask" và đường thường cho cùng một ảnh (resize 512, cắt giữa 448, không augmentation).
  User duyệt ("ok") thêm `model.explanation.mask_cache_dir=""` vào cả 10 lượt, không sửa code.
- Lần 2: chuỗi `~/tta_chain.sh` chạy đúng một lần, 23:59:19 → 00:13:02 ngày 2026-10-31, từ `~/Meta-CXR-tta`
  (`4338f16` + ba file của plan này). Log ghi `mask_cache_dir: null`.
- **G1 qua:** lượt `id` của C42 và R1C giống từng bit file `val_predictions_epoch_best.npz` (max|Δp| = 0). Mỗi
  log có đúng một dòng `Loading evaluation weights from`. **G2 qua:** mọi lượt có biến đổi khác lượt `id`.
- Mỗi lượt khoảng 1–1,5 phút, khoảng 4,7 GB. Không có file test, không còn file `epoch_provided` trong
  `run_20261005_paper`.

**TTA (trung bình 5 lượt) − lượt `id`, val n = 1.808:**

| Chỉ số | C42 | R1C (mô hình đang báo cáo) |
|---|---|---|
| **macro AP "có bệnh"** (quyết định) | 0,4329 → 0,4361, **+0,0032** [+0,0001; +0,0063] | 0,4271 → 0,4309, **+0,0038** [+0,0003; +0,0069] |
| AUROC "có bệnh" | −0,0010 | −0,0007 |
| RP12 | +0,0056 | −0,0045 |
| wF1 (rào chắn) | −0,0005 | −0,0005 |
| F1-5 (rào chắn) | −0,0045 | −0,0010 |
| Recall P/U dưới cutpoint | **−0,0169** | **−0,0103** |
| Precision P/U dưới cutpoint | +0,0015 | +0,0023 |
| Recall / precision P/U dưới argmax | −0,0025 / +0,0012 | −0,0017 / +0,0013 |

Trung bình hai mô hình: **+0,0035** [+0,0004; +0,0064] (bootstrap ghép cặp theo study, 1.000 lần, seed 16).
Rào chắn đạt.

**Macro AP từng lượt đơn lẻ (chỉ mô tả):** C42 `id` 0,4329, r+5 0,4308, r−5 0,4315, s095 0,4296, s105 0,4319;
R1C `id` 0,4271, r+5 0,4278, r−5 0,4276, s095 0,4245, s105 0,4276. Không lượt biến đổi nào tốt hơn rõ rệt;
lợi ích chỉ đến từ việc lấy trung bình.

**AP theo finding hiếm (chỉ mô tả), C42 / R1C:** Lung Lesion +0,019 / +0,015, Pneumothorax +0,004 / +0,016,
Enlarged Cardiomediastinum +0,004 / +0,004, Fracture +0,002 / −0,000, Consolidation −0,000 / +0,000, Pleural
Other −0,000 / −0,002. Ngoài ra Pneumonia +0,012 / +0,009; Lung Opacity −0,005 / −0,001.

**Đọc kết quả:**

- TTA nâng macro AP một chút và đều ở cả hai mô hình (khoảng tin cậy trên 0 ở cả hai), nhưng dưới ngưỡng đã
  chốt. AUROC không tăng (−0,001).
- **Không giúp mục tiêu recall:** dưới cutpoint (fit lại trên xác suất TTA), recall P/U GIẢM −0,017 / −0,010,
  precision tăng nhẹ. Giả thuyết (chưa đo): trung bình 5 lượt làm xác suất "mềm" hơn, nên cutpoint tối ưu wF1
  dời về phía ít gọi dương tính hơn.
- Tốn 5 lần suy luận cho +0,0035 AP. Không đề xuất chấm test.

**Giới hạn:** chỉ val; một bộ biến đổi cố định, không dò; cùng một mô hình nên không có NB seed.
