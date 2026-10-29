# Stage 1: kiểm tra đầu vào và recall (2026-10-29)

**Trạng thái:** đã đo xong, chỉ đọc file và chạy CPU. Không train, không sửa code hay YAML.
Kết quả tổng hợp nằm trên host tại `~/input_audit_20261029/` (`input_audit.json`, `rule_xfit.log`,
`la_posthoc.log`). Các script nằm cạnh đó (`~/input_audit.py`, `~/rule_xfit.py`, `~/la_posthoc.py`) và
chưa đưa vào repo.

**Yêu cầu của user:** tăng recall của Stage 1 để mô hình không bỏ sót finding, nhưng vẫn phải là phát
triển tiếp bài báo gốc chứ không phải một pipeline khác. Trước hết phải kiểm tra xem đầu vào có phải là
nguyên nhân hay không.

**Dữ liệu dùng:** `run_20261005_paper` phase1c `checkpoint_best`, gồm `val_predictions_epoch_best.npz`
(1.808 study) và `test_predictions_epoch_best.npz` (3.269 study), cùng cutpoint
`configs/stage1_cutpoints/run_20261005_paper.json`.

**Định nghĩa:** một ô bị coi là "bỏ sót" khi nhãn tự động là Positive hoặc Uncertain (blank tính là
Negative, D-018) nhưng quyết định là Negative. Đây cũng chính là định nghĩa cue âm tính sai của Stage 2.

## 1. Recall theo bệnh (cutpoint headline)

- **Toàn bộ:** recall P/U trên val là 0,572 (precision 0,651); trên test là 0,531 (precision 0,608).
  Với argmax: val 0,732 / 0,500, test 0,719 / 0,484.
- **Bệnh mất nhiều ca nhất trên test:** Lung Opacity 625 ca (15%), Pneumonia 465, Atelectasis 445,
  No Finding 359, Enlarged Cardiomediastinum 354, Edema 348, Cardiomegaly 329.
- **Recall gần 0 ngay cả với argmax:** Enlarged Cardiomediastinum 0,00, Fracture 0,00,
  Pleural Other 0,04 và Lung Lesion 0,11. Consolidation đạt 0,09 với cutpoint và 0,48 với argmax.
  Pneumothorax đạt 0,28 và 0,52.

## 2. Các giả thuyết về đầu vào: không giả thuyết nào là nguyên nhân chính

- **View (chuẩn hóa theo cơ cấu bệnh của mỗi split):** tỷ lệ bỏ sót ở AP là 0,44 trên val và 0,47 trên
  test; ở PA là 0,50 và 0,51. Ảnh AP không bị bỏ sót nhiều hơn PA.
- **Số view:** study 1 view và 2 view có tỷ lệ bỏ sót 0,45 / 0,49 trên val và 0,48 / 0,50 trên test.
- **Phổi bị cắt khỏi khung (mask phổi CheXmask, có cho 4.774 trên 5.077 ảnh anchor, aspect khớp 100%).**
  - BioViL (resize 512 rồi cắt 448) mất trung bình 2,5–3,0% diện tích phổi, trung vị 0,7–1,1%. Khoảng
    18–22% số ảnh mất hơn 5%. Phần mất nằm ở hai bên (1,5–1,8%) và phía trên (khoảng 1%); phía dưới
    gần như không mất (0,07–0,10%).
  - PubMedCLIP mất 0,3–0,4%. Swin đệm thành ảnh vuông nên không mất gì.
  - Tính chung mọi bệnh, tỷ lệ bỏ sót theo nhóm mất <0,5% / 0,5–2% / 2–5% / >5% là:
    PA 0,50 / 0,50 / 0,49 / 0,45; AP 0,43 / 0,45 / 0,44 / 0,42. Tức là không tăng theo phần bị cắt.
  - Ở Pleural Effusion trên phim PA, tỷ lệ tăng 0,20 → 0,39, nhưng mỗi nhóm chỉ 36–54 ô và hiện tượng
    không xuất hiện trên phim AP. Hiện chưa đủ để coi là nguyên nhân.
- **Nhãn, đối chiếu với 676 study test có nhãn bác sĩ (MIMIC-CXR-JPG 2.1.0).**
  - Chỉ 76 trên 6.908 ô blank (1,1%) được bác sĩ ghi là P/U.
  - Trong 999 ô bỏ sót, **249 ô (25%) bác sĩ ghi là Negative**, nghĩa là lỗi nằm ở nhãn tự động. Các
    bệnh chiếm nhiều: No Finding 31/42, Lung Opacity 51/136, Pneumonia 31/116,
    Enlarged Cardiomediastinum 24/83, Pleural Other 23/43, Support Devices 22/40.
  - Bác sĩ gán nhãn bằng cách đọc báo cáo, nên phép đo này không thấy được những bệnh có trên phim mà
    báo cáo không viết ra.

## 3. Quy tắc quyết định: chỉ val, cross-fit 2 nửa × 10 lần

| Quy tắc | Recall P/U | Precision | wF1 | F1-5 | Macro recall | Cue P/U mỗi study |
|---|---:|---:|---:|---:|---:|---:|
| Cutpoint tối ưu wF1 (đang dùng) | 0,556 | 0,641 | 0,858 | 0,808 | 0,465 | 1,96 |
| Argmax (giao thức của bài báo) | 0,732 | 0,500 | 0,829 | 0,749 | 0,491 | 3,30 |
| Eq. 22 (ô bị bỏ phiếu trắng tính là không có) | 0,801 | 0,357 | 0,767 | 0,736 | 0,545 | 5,06 |
| Cutpoint tối ưu macro recall | 0,862 | 0,319 | 0,715 | 0,699 | 0,534 | 6,11 |

Chỉ cần đổi ngưỡng là recall tăng, nhưng precision giảm theo đúng đường precision–recall hiện có.
Trong khi đó Stage 2 chép theo khoảng 50% cue dương tính sai (PLAN-2026-10-28-stage2-cue-share).

## 4. Logit adjustment kiểu PromptMRG KHÔNG thay đổi gì dưới cutpoint

PromptMRG (AAAI 2024; code `jhb86253817/PromptMRG`, `models/blip.py` và `modules/trainer.py`) làm như sau:

- Khi train, cộng `log(pi_d)` vào logit Positive. `pi_d` là trung bình p_pos trên các ca dương tính của
  tập val, chia cho giá trị lớn nhất.
- Khi suy luận thì không cộng gì và dùng argmax.
- Bài báo báo cáo CE recall +0,034, precision −0,013.

Phép thử hậu kiểm trên val (cross-fit, `pi` tính trên nửa dùng để fit):

- Dùng cutpoint fit lại: quyết định giống hệt nhau ở **100% số ô**, recall vẫn 0,556 và precision vẫn
  0,641. Lý do là điểm `s = p_pos/(p_pos+p_neg)` chỉ dời một hằng số theo từng bệnh.
- Dùng argmax: recall 0,732 → 0,814, precision 0,500 → 0,383. Đây là dời ngưỡng, giống hệt hàng Eq. 22
  ở bảng trên.

Phiên bản áp dụng lúc train chỉ xấp xỉ phiên bản hậu kiểm, vì đầu phân loại có bias tự do cho từng
bệnh và từng lớp. Vì vậy nhiều khả năng mức tăng recall của PromptMRG là do dời điểm vận hành, mà điểm
vận hành thì cutpoint đã tự chọn. **Không đề xuất pilot hướng này.**

## 5. Hệ quả cho việc tăng recall

Muốn tăng recall mà không mất precision, mô hình phải xếp hạng tốt hơn, tức là có đường precision–recall
tốt hơn. Quy tắc đọc kết quả cho mọi pilot sau này cũng phải dựa trên chỉ số không phụ thuộc ngưỡng, hoặc
so sánh recall ở cùng một mức precision. Các hướng còn lại đang chờ user chọn:

1. **Asymmetric loss** (Ridnik ICCV 2021): giảm trọng số các ca âm tính dễ.
   - Trên MS-COCO, mAP (không phụ thuộc ngưỡng) tăng từ 84,0 (CE) lên 85,1 (focal) và 86,6 (ASL). Đó là
     kết quả train end-to-end.
   - Chưa tìm được kết quả nào trên CXR có khoảng tin cậy.
   - Áp vào CE ba lớp có trọng số của MHCAC.
2. **Mở khóa encoder nông ngay từ pha 1b** (biểu diễn ảnh được thích nghi).
   - Bằng chứng tốt nhất trong repo: `run_20260820_ft`, AUROC +0,020 và tăng ở 14/14 bệnh (đo theo cách
     chấm nhị phân cũ, 10 epoch).
   - Arm C của pilot 2026-10-11 thua (−0,036 AUROC), nhưng arm đó gộp thêm bố cục patch BioViL nên
     không tách được tác động riêng.

Recall gần 0 ở các tổn thương nhỏ và hiếm (Fracture, Lung Lesion, Pleural Other) khớp với tài liệu về độ
phân giải (Wollek 2023: nodule AUC 0,780 ở 512 px, 0,803 ở 1024 px). Luồng 448 px của BioViL hiện gần như
không được dùng (CLAUDE.md, "BioViL stream is mis-read").
