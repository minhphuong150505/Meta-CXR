# PLAN 2026-11-01: điểm vận hành thiên recall cho Stage 1 (cutpoint F2)

**Trạng thái:** user duyệt 2026-11-01 ("1 đồng ý; 2 bạn tăng Beta lên đi tối ưu recall và precision"). Code đã viết
(mục 5). Chỉ CPU, không train, không sửa YAML.

**Diễn giải yêu cầu về β:** không cố định β = 2. B0 chạy lưới β ∈ {1; 1,5; 2; 3; 4} trên val (cross-fit); user chọn β có
đánh đổi recall–precision ưng ý nhất dựa trên B0, TRƯỚC khi chấm test. Test chỉ chấm một lần, với β đã chọn.

## 1. Vì sao

- Mục tiêu của user: Stage 1 bỏ sót ít finding hơn.
- Các phép đo 2026-10-29 → 10-31 cho thấy khả năng xếp hạng (macro AP "có bệnh" khoảng 0,43) gần như không đổi được
  bằng loss, encoder, đầu vào, TTA hay ensemble. Vậy recall chỉ còn đổi được bằng **điểm vận hành**.
- Quy tắc headline hiện tại (cutpoint fit để tối đa weighted F1 từng finding) tự chọn điểm ít báo bệnh. Audit
  2026-10-29, val cross-fit: recall "có bệnh" (P hoặc U) 0,556 / precision 0,641; argmax 0,732 / 0,500; Eq. 22 0,801 /
  0,357. Trên test với file headline: 0,531 / 0,608.
- Weighted F1 của sklearn tính trọng số theo số ca, mà lớp Negative chiếm đa số, nên tiêu chí này thưởng việc gọi
  Negative. Đó là lý do recall thấp, chứ không phải do mô hình.

## 2. Quy tắc mới (opt-in, file headline giữ nguyên)

Vẫn ba lớp, vẫn hai cutpoint trên `s = p_pos/(p_pos+p_neg)`: Negative < t1 ≤ Uncertain < t2 ≤ Positive. Fit trên val,
từng finding, hai bước:

1. **t1** (ranh giới "có bệnh" / Negative, quyết định số ca bỏ sót): tối đa **F_β của "có bệnh"** (P hoặc U so với N),
   **β = 2** (recall nặng gấp 4 lần precision trong trung bình điều hòa). Lưới ứng viên như hiện tại (121 phân vị của
   chính điểm `s` của finding đó).
2. **t2** (chia Positive / Uncertain trong nhóm "có bệnh"): với t1 đã chọn, tối đa weighted F1 ba lớp trên các
   `t2 ≥ t1`. Finding có dưới 5 ô Uncertain: `t2 = t1`, như hiện tại.

Bước 2 không đổi recall "có bệnh", chỉ đổi cách chia P/U. β = 2 là đề xuất; user có thể chọn giá trị khác TRƯỚC khi
chạy (β = 1 là F1 "có bệnh"; β lớn hơn thiên recall hơn).

## 3. Đo

- **B0, chỉ val (không đụng test):** cross-fit 2 nửa × 10 lần, seed 16 (giống audit): fit trên một nửa, chấm nửa kia.
  So các quy tắc: cutpoint headline (wF1), cutpoint F-β với β ∈ {1; 1,5; 2; 3; 4}, argmax. (Eq. 22 không chạy lại; số
cross-fit của nó đã có trong audit 2026-10-29: 0,801 / 0,357.) Báo recall / precision "có bệnh", wF1, F1-5,
  macro recall, số finding P/U mỗi study, recall từng finding (nhất là các finding bỏ sót nhiều: Lung Opacity,
  Pneumonia, Atelectasis, Enlarged Cardiomediastinum, Consolidation, Pneumothorax).
- **Quyết định, chốt trên B0 (trước khi đọc test):** user chọn β, và chọn F-β là dòng bổ sung bên cạnh headline, hoặc
  thay headline, hoặc bỏ.
- **B1, test một lần:** fit F-β (β đã chọn) trên toàn bộ val (1.808), ghi
  `configs/stage1_cutpoints/run_20261005_paper_recall_f<β>.json`,
  chấm test (3.269) một lần, ghép cặp với file headline: bootstrap theo study, 1.000 lần, seed 16, cho mọi chỉ số ở B0.
- Dữ liệu: `run_20261005_paper` phase1c `val/test_predictions_epoch_best.npz` trên máy train. Chạy CPU trên máy train;
  chỉ số tổng hợp mang về.

## 4. Cách báo cáo, chốt TRƯỚC khi chạy

- Đây **không** phải mô hình tốt hơn: AUROC và AP không đổi. Mọi con số F2 ghi là "điểm vận hành thiên recall (cutpoint
  F2 fit trên val), không phải giao thức bài báo", luôn đặt cạnh headline wF1 và argmax.
- Báo cả những chỉ số giảm (precision, wF1, F1-5), không chỉ recall.
- Recall "có bệnh" trên test sẽ gần như chắc chắn tăng; đó không phải bằng chứng gì ngoài việc dời ngưỡng.
- **Stage 2 không đổi.** Cue của Stage 2 vẫn dùng file headline. Pilot 2026-10-29 cho thấy cue thiên recall (Eq. 22) làm
  Stage 2 tệ hơn (BERTScore −0,0063 trung bình 2 seed), nên không đưa F2 vào cue nếu không có pilot riêng.

## 5. Code (đã viết 2026-11-01)

- `training/evaluation/threshold_calibration.py`: thêm objective `present_fbeta` (tham số `beta`) vào
  `fit_severity_cutpoints`; mặc định `weighted_f1` giữ nguyên từng bit.
- `scripts/calibrate_thresholds.py`: `--objective {weighted_f1,present_fbeta}` và `--beta` cho `--rule cutpoints`;
  metadata của file ghi objective và β.
- `scripts/compare_stage1_operating_points.py` (mới): `--crossfit` trên val (B0) và so ghép cặp trên test (B1) giữa
  các quy tắc; chỉ in số tổng hợp; từ chối fit trên test.
- `tests/test_recall_operating_point.py` (6 test): F-β đúng số tính tay; β lớn hơn không làm t1 cao hơn; t2 không đổi các ô
  đã gọi "có bệnh"; mặc định không đổi; cross-fit không bao giờ chấm trên nửa đã fit; recall/precision "có bệnh" đúng số
  tính tay. Kiểm tra thêm trên máy train trước B0: fit lại headline từ npz val của R1C phải ra đúng
  `configs/stage1_cutpoints/run_20261005_paper.json`.
- Cập nhật `struct/`, CLAUDE.md, README cùng commit.

## 6. Chi phí

Chỉ CPU, vài phút. Không GPU, không ổ đĩa đáng kể.

## 7. Giới hạn

- Cùng một mô hình; chỉ dời ngưỡng theo đường precision–recall sẵn có.
- β là lựa chọn giá trị (bỏ sót quan trọng hơn báo nhầm bao nhiêu lần), không phải điều dữ liệu quyết định.
- Finding xếp hạng kém (Enlarged Cardiomediastinum, Fracture, Pleural Other) có thể bị gọi "có bệnh" rất nhiều để đạt
  recall; số P/U mỗi study sẽ được báo để thấy điều đó.
- Nhãn tự động: 25% ca "bỏ sót" trên test có nhãn bác sĩ là Negative (audit 2026-10-29), nên recall đo theo labeler
  thấp hơn recall thật.
