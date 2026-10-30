# PLAN 2026-10-29: mở khóa encoder nông ngay từ pha 1b, để tăng recall bằng cách xếp hạng tốt hơn

**Trạng thái:** user duyệt bước GPU 2026-10-29 ("ok"). **ĐÃ CHẠY: U5 TỆ HƠN**: macro AP −0,0093
[−0,0143; −0,0045], không chạy full. Xem Execution report.
User đã chọn hướng này sau khi ASL thất bại
(`PLAN-2026-10-29-stage1-asymmetric-loss.md`). Không cần sửa code: chỉ dùng override cấu hình, cùng
cơ chế mà arm C của pilot 2026-10-11 đã dùng. Không sửa YAML.

**Mục tiêu.** Tăng recall ở cùng mức precision. Muốn vậy mô hình phải xếp hạng tốt hơn
(`PLAN-2026-10-29-stage1-recall-input-audit.md`).

## 1. Vì sao chọn hướng này, và bằng chứng ngược chiều

**Bằng chứng ủng hộ:**

- `run_20260820_ft` mở khóa đúng 5 pattern nông: ResNet50 `layer4` và projector của BioViL, CLIP block
  10–11 và `post_layernorm` của PubMedCLIP. Kết quả: AUROC tăng ở 14/14 finding, macro +0,020
  [+0,014; +0,026].
  - Mức tăng lớn nhất rơi đúng vào các finding đang có recall gần 0: Pleural Other +0,080,
    Lung Lesion +0,039, Pneumothorax +0,030.
  - Lưu ý: run đó train 10 epoch, dùng cách chấm nhị phân cũ, kiến trúc cũ (hai encoder, chưa có pha),
    và gộp chung với việc đổi kappa.
- Tài liệu: encoder đóng băng là trần thấp.
  - Tayebi Arasteh et al., arXiv 2510.07191: fine-tune toàn phần hơn đặc trưng đóng băng.
  - arXiv 2406.10050: linear probe đạt khoảng 0,80–0,82 AUROC, fine-tune khoảng 0,87–0,88.
  - R2GenGPT: fine-tune Swin nâng CE F1 từ 0,325 lên 0,389.

**Bằng chứng ngược chiều, ở đúng budget pilot.** Đo lại hôm nay trên CPU từ các file của pilot
2026-10-11 (`~/s1pilot_20261011`), với chỉ số quyết định của plan này, một seed:

- B = bố cục patch cộng projector được train. C = B cộng mở khóa đúng 5 pattern trên.
- **C − B: macro AP −0,0142, AUROC −0,0169.** Mở khóa làm xếp hạng kém đi, và mức giảm vượt NB 0,0042.
- B − A: macro AP −0,0092.
- Cần đọc kết quả này một cách thận trọng:
  - C đi kèm bố cục patch, nên không tách được tác động của riêng việc mở khóa.
  - C được chọn ở epoch 0 theo `macro_recall`, còn B ở epoch 1.
  - Chỉ có một seed.
- Arm A của pilot đó trùng từng bit với P15_s42, control của plan này.
- Pha 1c của mô hình đang báo cáo cũng mở khóa 5 pattern này (1 epoch, cùng với ITC/ITM/LM) mà không
  tăng AUROC trên test (+0,0037 [−0,0012; +0,0095]).

**Kết luận trước khi chạy.** Chưa ai thử mở khóa trên bố cục token cũ (cấu hình đang ship) ở pha 1b.
Bằng chứng gần nhất ở budget pilot lại nghiêng về phía xấu hơn. Pilot này rẻ (khoảng 2 giờ) và trả lời
đúng câu hỏi đó. Nếu nó không đạt, mở khóa ở budget pilot coi như đã thử xong. Việc mở khóa có giúp ở
budget lớn hay không (10 epoch như `run_20260820_ft`) là một câu hỏi khác, tốn khoảng 6 giờ trở lên mỗi
run, và cần user quyết định riêng.

## 2. Bước GPU

**Công thức pilot giống P15:**

- Chỉ chạy pha 1b, trên 60.000 study train đầu tiên, 2 epoch, warm-up 150 update.
- Khởi tạo từ `run_20261005_paper/checkpoint_phase1a.pth`.
- Đặt `test_splits=[]`, nên không chấm test.
- Chạy nối tiếp, một card, mỗi arm có thư mục mới dưới `~/unf1b_20261029/`.

**Override cho arm U5.** Các pattern giống hệt pha 1c và `run_20260820_ft`. LR của encoder lấy
`init_lr_enc` 1e-5 có sẵn trong YAML; runner tự đưa các tham số này vào nhóm `encoder_*`.

```
P=run.phases.phase1b
PAT=[visual_encoder.encoder.encoder.layer4,visual_encoder.projector,pubmedclip.model.vision_model.encoder.layers.10,pubmedclip.model.vision_model.encoder.layers.11,pubmedclip.model.vision_model.post_layernorm]
$P.unfreeze_encoder_blocks=true $P.model.encoder_finetune.enabled=true
$P.model.encoder_finetune.patterns=$PAT $P.model.encoder_finetune.keep_batchnorm_eval=true
```

| Arm | Seed | Control so sánh |
|---|---|---|
| U5_s42 | 42 | P15_s42 (`~/auxdrop_pilot_20261025/P15`) |
| U5_s17 | 17 | P15_s17 (`~/seednoise_20261026/P15_s17`) |

**Các cổng kiểm tra (dừng ngay nếu một cổng không qua):**

- **G0.** Checkout trên host phải đồng bộ với lịch sử đã viết lại (`928f0cb`, cùng tree với `a8d6936`).
  Nếu host còn ở `a8d6936` thì vẫn chạy được vì tree giống hệt; không reset gì nếu user chưa đồng ý.
  GPU rảnh, ổ là `ntfs3`, `/home` còn ít nhất 12 GB, thư mục đích chưa tồn tại.
- **G1.** Smoke U5 (2.000 study, 1 epoch): log phải in đúng
  `unfroze 69 parameters (31.85M) across 5 patterns; batchnorm_eval=True`. Dòng trainable phải có
  `visual_encoder 17.67M` và `pubmedclip 14.18M`. Không Traceback, không OOM, loss hữu hạn.
- **Mỗi arm thật:** rc=0, seed đúng, có `val_predictions_epoch_best.npz`, không có file test nào.

Không cần cổng so sánh bit với code cũ: code không đổi so với lần chạy ASL, nơi G1 đã xác nhận đường P15
giống hệt.

**Chi phí:** khoảng 47–55 phút mỗi arm (arm C cũ mất 46m55s, 6,8 GB) cộng smoke khoảng 10 phút.
**Tổng khoảng 2 giờ GPU.**

## 3. Quy tắc đọc kết quả, chốt TRƯỚC khi chạy

Giống hệt plan ASL, để hai hướng so sánh được với nhau. Script chấm là `~/asl_probe.py`, đã chạy thật
một lần.

- **Chỉ số quyết định:** macro AP "có bệnh" (P hoặc U so với N, trên `s = p_pos/(p_pos+p_neg)`),
  trung bình 14 finding. NB = 0,0042, nên NB/√2 = 0,0030.
- Gọi Δ_s = U5_s − P15_s. Khoảng tin cậy của trung bình hai seed lấy bằng bootstrap ghép cặp theo study,
  1.000 lần, seed 16.

**U5 TỐT HƠN** khi đạt đủ bốn điều kiện:

1. Δ macro AP > 0 ở **cả hai** seed.
2. Trung bình hai seed > 0,0030.
3. Cận dưới khoảng tin cậy > 0.
4. Rào chắn: trung bình Δ wF1 ≥ −0,0019 và trung bình Δ F1-5 ≥ −0,0042 (cutpoint cross-fit).

**U5 TỆ HƠN** khi Δ macro AP < 0 ở cả hai seed và trung bình < −0,0030, hoặc khi rào chắn không đạt.

**Mọi trường hợp còn lại:** KHÔNG XÁC LẬP.

**Chỉ để mô tả:** AUROC "có bệnh"; recall ở precision mốc; recall và precision P/U dưới cutpoint;
các số argmax; AP theo từng finding (nhất là các finding hiếm).

**Nếu U5 TỐT HƠN:** không chạy full ngay. Hỏi user chạy thêm seed 7 và 23, giữ nguyên quy tắc này.
Nếu 4 seed cùng đạt, khi đó mới đề xuất một run 1b đầy đủ có mở khóa, kèm cutpoint mới và một lần chấm
test.

**Nếu không xác lập hoặc tệ hơn:** ghi lại kết quả. Mở khóa ở budget pilot coi như đã thử xong. Các
hướng còn lại (mở khóa ở budget lớn, mở khóa Swin, độ phân giải cao hơn) đều tốn kém hơn và cần user
quyết định.

## 4. Giới hạn đã biết trước

- Chỉ là pilot (1b, 60.000 study, 2 epoch), 2 seed, chỉ val.
- Mức lợi của mở khóa có thể chỉ xuất hiện sau nhiều epoch hơn.
- Swin (MedCLIP), luồng đang mang phần lớn tín hiệu, không nằm trong 5 pattern, vì nó chưa bao giờ được
  đo khi mở khóa.
- Encoder thay đổi thì đầu vào của Q-Former (đóng băng sau 10% đầu 1b) bị lệch, nên soft token cho
  Stage 2 có thể kém đi. Điều này cần đo riêng nếu U5 thắng.

## Execution report

**Kết quả: U5 TỆ HƠN.** Kết luận theo quy tắc đã chốt từ trước khi chạy. Không chạy full.

**Thông tin chạy:**

- Chuỗi `~/unf.sh` chạy trên host từ 15:43 đến 17:25 ngày 2026-10-29, khởi chạy đúng một lần.
- Code `6a438ac`: so với `928f0cb` chỉ đổi một dòng chú thích trong test và `build.py` của `Eval/`, không
  đụng code train. YAML không đổi kích thước hay mtime.
- Log ở `~/unf.log`; các arm và `probe.log` ở `~/unf1b_20261029/`.

**Các cổng kiểm tra và arm thật:**

- **G1 qua.** Smoke in đúng `unfroze 69 parameters (31.85M) across 5 patterns; batchnorm_eval=True`.
  Dòng trainable có `visual_encoder 17.67M` và `pubmedclip 14.18M`. Fusion là đường early, ASL tắt.
- **Hai arm thật:** U5_s42 train 53m35s, U5_s17 train 44m05s, đỉnh khoảng 6,8 GB. Seed ghi trong log
  đúng, không có NaN, không có file test nào được tạo.

Kết quả trên val (1.808 study). Δ = U5 − P15.

| | Δ seed 42 | Δ seed 17 | Trung bình | NB/√2 |
|---|---:|---:|---:|---:|
| **macro AP "có bệnh" (quyết định)** | **−0,0005** | **−0,0181** | **−0,0093 [−0,0143; −0,0045]** | 0,0030 |
| AUROC "có bệnh" | −0,0083 | −0,0185 | −0,0134 | 0,0086 |
| Recall ở precision mốc (12 finding) | +0,0166 | −0,0571 | −0,0202 | 0,0157 |
| wF1, cutpoint cross-fit (rào chắn) | +0,0001 | −0,0013 | −0,0006 | 0,0019 |
| F1-5, cutpoint cross-fit (rào chắn) | −0,0005 | −0,0032 | −0,0019 | 0,0042 |
| Recall P/U dưới cutpoint | +0,0265 | −0,0076 | +0,0095 | 0,0272 |
| Precision P/U dưới cutpoint | −0,0092 | −0,0049 | −0,0070 | — |
| Recall P/U với argmax | −0,0005 | −0,0248 | −0,0127 | — |
| Precision P/U với argmax | +0,0083 | +0,0118 | +0,0101 | — |
| wF1 với argmax | +0,0034 | +0,0063 | +0,0049 | — |

AP theo từng finding (trung bình hai seed), so với P15:

- **Giảm** ở 11/14 finding. Giảm mạnh nhất: Enlarged Cardiomediastinum −0,027, Pneumonia −0,026,
  Edema −0,021, Consolidation −0,015, Atelectasis −0,013, Lung Lesion −0,012, Pneumothorax −0,011.
- **Tăng nhẹ:** Support Devices +0,007, No Finding +0,002, Fracture +0,002.

**Cách đọc:**

- Quy tắc TỆ HƠN đạt: Δ âm ở cả hai seed, trung bình −0,0093 vượt NB/√2, và khoảng tin cậy nằm hẳn
  dưới 0.
- Hai seed lệch nhau rõ: seed 42 gần như bằng 0, seed 17 giảm mạnh. Riêng ở seed 42 thì đây là "không
  xác lập"; hướng chung qua hai seed là xấu hơn.
- Chiều này khớp với arm C cũ (C − B −0,0142).
- Mở khóa **không** giúp các finding hiếm ở budget này, dù đó chính là nơi `run_20260820_ft` (10 epoch)
  từng thấy mức tăng lớn nhất.
- Với argmax, precision và wF1 tăng nhẹ còn recall giảm nhẹ. Đó là điểm vận hành dịch chuyển, không phải
  xếp hạng tốt hơn.

**Giới hạn:** chỉ là pilot (1b, 60.000 study, 2 epoch), 2 seed, chỉ val. Lần này không đo được trường
hợp encoder cần nhiều epoch mới thích nghi có lợi.
