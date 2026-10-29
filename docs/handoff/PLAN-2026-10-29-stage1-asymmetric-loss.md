# PLAN 2026-10-29: asymmetric loss cho MHCAC, để tăng recall bằng cách xếp hạng tốt hơn

**Trạng thái:**

- Plan: user duyệt 2026-10-29.
- **Bước 2 (sửa code và chạy test CPU): đã duyệt và XONG 2026-10-29.**
  - Test trên máy dev: `tests/test_asymmetric_loss.py`, `tests/test_logit_adjustment.py` và
    `tests/test_stage1_objectives.py` đều xanh.
  - Bộ test CPU đầy đủ trên host, chạy trong worktree tạm từ `20a6ee1` có áp thay đổi: không lỗi nào,
    2 skip.
  - OmegaConf: khóa đặt qua phase1b đọc ra (4,0, 0,05); YAML shipped đọc ra (0, 0).
- Bước 3 (GPU): chờ user duyệt. Chưa chạy.

Thứ tự các bước, mỗi bước chờ user duyệt riêng:
(1) duyệt plan này, (2) sửa code và chạy test CPU, (3) chạy GPU.

**Bối cảnh:** `PLAN-2026-10-29-stage1-recall-input-audit.md`.

- Đầu vào (view, khung bị cắt, số view, nhãn blank) không phải nguyên nhân chính của việc bỏ sót.
- Chỉ đổi ngưỡng thì recall tăng nhưng precision rơi theo đúng đường precision–recall hiện có.
- Logit adjustment kiểu PromptMRG không làm đổi quyết định nào dưới cutpoint.
- Vậy chỉ một thay đổi khiến mô hình **xếp hạng tốt hơn** mới tính là tăng recall thật.

## 1. Thay đổi

**Nguồn:** Ridnik et al., *Asymmetric Loss For Multi-Label Classification*, ICCV 2021
(code `Alibaba-MIIL/ASL`, `src/loss_functions/losses.py`, lớp `AsymmetricLoss`).

- Họ báo cáo trên MS-COCO, mAP (không phụ thuộc ngưỡng): CE 84,0, focal 85,1, ASL 86,6.
- Tham số khuyến nghị: gamma_neg = 4, gamma_pos = 0, clip m = 0,05.
- Trọng số focusing không truyền gradient (`disable_torch_grad_focal_loss=True` trong `AsymmetricLoss`).
- Họ train cả backbone. Ở đây encoder gần như đóng băng trong 1b, nên mức tăng có thể nhỏ hơn.

**Áp vào MHCAC.** MHCAC có loss CE ba lớp có trọng số theo từng finding (bài báo Eq. 10,
`mhcac.loss.ClassificationLoss`). "Âm tính" của ASL ở đây là ô có nhãn Negative. Với mỗi ô hợp lệ
(nhãn khác -100):

- Nhãn Positive hoặc Uncertain: `l = -log p_y`. Giữ nguyên CE (gamma_pos = 0).
- Nhãn Negative: đặt `q = min(p_neg + m, 1)` và `w = stopgrad((1 - q) ** gamma_neg)`. Khi đó
  `l = -w * log(q)`.
  - Một ô âm tính dễ có `p_neg >= 1 - m`: loss bằng 0 và không có gradient. Đây là ngưỡng cứng của ASL.
  - Một ô âm tính càng giống dương tính (`p_neg` càng thấp) thì càng được giữ trọng số.
  - Với hai lớp, cách viết này trùng hẳn với ASL gốc, vì `1 - p_m = min(p_neg + m, 1)` với
    `p_m = max(p_present - m, 0)`.
- Trọng số lớp của bài báo (`class_weights`, Eq. 10) **giữ nguyên**: mỗi ô vẫn nhân `w_c[y]`, và cách
  chuẩn hóa giữ nguyên như `nn.CrossEntropyLoss(weight=...)`, tức chia cho `sum w_c[y]` trên các ô hợp
  lệ. Như vậy, so với recipe hiện tại, chỉ có phần focusing trên ô Negative là thay đổi.
- Vẫn là ba lớp, không gộp lớp nào (D-023). Argmax, xác suất xuất ra và mọi chỉ số đều đọc logit gốc.

**Cấu hình:** khóa mới `model.mhcac.asymmetric_loss: {gamma_neg, clip}`.

- Thiếu khóa này thì hành vi y như cũ, đến từng bit.
- Khóa này không ghi vào YAML, vì sửa YAML làm đổi mtime (bẫy `oracle_mix` của Stage 2). Đặt qua
  phase: `run.phases.phase1b.model.mhcac.asymmetric_loss.gamma_neg=4` và `...clip=0.05`.
- Từ chối chạy khi đi cùng `logit_adjustment` hoặc khi `label_smoothing > 0`. Shipped là 0,0 (D-026).

**File code dự kiến sửa:**

- `mhcac/loss.py` (`ClassificationLoss`).
- `model/lavis/models/blip2_models/blip2_qformer.py` (đọc khóa và truyền vào).
- Test mới `tests/test_asymmetric_loss.py`.
- Tài liệu: `struct/` (trang của `loss.py` và `blip2_qformer.py`), `CLAUDE.md`, `README.md`.

## 2. Test CPU (phải xanh trước khi chạy GPU)

1. `gamma_neg = 0` và `clip = 0` (cũng như khi thiếu khóa): loss và gradient bằng hệt CE có trọng số
   hiện tại, trên logit ngẫu nhiên, có cả ô -100 và có `sample_mask`.
2. Một ví dụ nhỏ tính tay: hai finding, ba ô, đủ cả ba nhãn.
3. Ô Negative có `p_neg >= 1 - m`: loss bằng 0 và gradient bằng 0.
4. Ô P/U: loss bằng đúng thành phần CE có trọng số tương ứng.
5. Trọng số focusing không có gradient. Gradient đo được phải khớp công thức tách `w`.
6. Từ chối khi đi cùng `logit_adjustment`, khi `label_smoothing > 0`, khi `gamma_neg < 0`, và khi
   `clip` nằm ngoài [0, 1).
7. Đường cấu hình: thiếu khóa thì không bật ASL; có khóa thì giá trị đến được `cls_loss_fn`.
8. Bộ test CPU đầy đủ trên host: chỉ được có đúng các lỗi baseline đã biết.

## 3. Bước GPU (chờ duyệt riêng sau bước 2)

**Công thức pilot giống các pilot trước:**

- Chỉ chạy pha 1b, trên 60.000 study train đầu tiên, 2 epoch, warm-up 150 update.
- Khởi tạo từ `run_20261005_paper/checkpoint_phase1a.pth`.
- Đặt `run.phases.phase1b.run.test_splits=[]`, nên **không chấm test**.
- Chạy nối tiếp, một card. Mỗi arm có thư mục mới dưới `~/asl_pilot_20261029/` (`test ! -e` trước mỗi arm).

| Arm | Thay đổi | Seed |
|---|---|---|
| ASL_s42 | gamma_neg 4, clip 0,05 | 42 |
| ASL_s17 | như trên | 17 |
| P15_s42 (đã có) | không | `~/auxdrop_pilot_20261025/P15` |
| P15_s17 (đã có) | không | `~/seednoise_20261026/P15_s17` |

P15 là recipe đang ship (multi-view ON) ở budget pilot, tức đúng mô hình đang báo cáo, thu nhỏ.

**Các cổng kiểm tra (dừng ngay nếu một cổng không qua):**

- **G0.** GPU rảnh, ổ là `ntfs3`, không có job khác, `/home` còn ít nhất 12 GB. Checkout sạch đúng commit
  đã định.
- **G1.** Smoke P15 (2.000 study, 1 epoch, seed 42) chạy ở commit `f4674ea` (code của P15_s17) và ở
  commit mới với ASL tắt. Checkpoint và npz val phải giống hệt (`~/latefusion_ckpt_cmp.py`), để bảo đảm
  control cũ còn dùng được.
- **G2.** Smoke ASL (2.000 study, 1 epoch): rc=0, log có cấu hình ASL đúng giá trị, loss hữu hạn, không
  Traceback, không OOM.
- **Mỗi arm thật:** rc=0, seed đúng, không có NaN, có `val_predictions_epoch_best.npz`, không có file
  test nào.

**Chi phí:** khoảng 51 phút mỗi arm (P15_s17 mất 50m59s) cộng ba smoke khoảng 30 phút. **Tổng khoảng
2 giờ 15 phút GPU.**

## 4. Quy tắc đọc kết quả, chốt TRƯỚC khi chạy

**Định nghĩa:**

- "Có bệnh" nghĩa là Positive hoặc Uncertain, đúng nghĩa cue của Stage 2.
- Điểm số là `s = p_pos / (p_pos + p_neg)`, chính là điểm mà cutpoint dùng.
- Mỗi arm lấy epoch tốt nhất theo cách chọn của chính pha đó (`macro_recall`), như các pilot trước.
- Mọi phép đo chỉ trên val, 1.808 study.
- Script đo: `~/recall_metrics.py`, sẽ đưa vào repo cùng code nếu user muốn.

**Chỉ số quyết định:** macro AP, tức AP của `s` cho "có bệnh" so với Negative, trung bình trên 14 finding.
Đây là chỉ số không phụ thuộc ngưỡng, cùng loại với mAP mà ASL báo cáo.

**Ngưỡng nhiễu (NB), đã đo hôm nay** trên 5 cặp seed có sẵn ở budget pilot: P15 42/17, OFF 42/17,
OFF 7/23, TK 42/17, TK 7/23. NB = 2 × RMS các chênh lệch giữa seed.

| | macro AP | AUROC "có bệnh" | Recall ở precision mốc | wF1 (cutpoint cross-fit) | F1-5 | Recall P/U (cutpoint) |
|---|---:|---:|---:|---:|---:|---:|
| NB | 0,0042 | 0,0122 | 0,0222 | 0,0027 | 0,0060 | 0,0385 |
| NB/√2 | 0,0030 | 0,0086 | 0,0157 | 0,0019 | 0,0042 | 0,0272 |

P15_s42 tham khảo: AP 0,4114, AUROC 0,7801, wF1 0,8531, F1-5 0,7990, recall P/U 0,533, precision 0,626.

Gọi Δ_s = ASL_s − P15_s cho mỗi seed s ∈ {42, 17}, và Δ̄ là trung bình hai seed. Khoảng tin cậy của Δ̄
lấy bằng bootstrap ghép cặp theo study, 1.000 lần, seed 16.

**ASL TỐT HƠN** khi đạt đủ cả bốn điều kiện:

1. Δ macro AP > 0 ở **cả hai** seed.
2. Δ̄ macro AP > NB/√2 = 0,0030.
3. Cận dưới khoảng tin cậy của Δ̄ macro AP > 0.
4. Rào chắn chỉ số headline: Δ̄ wF1 ≥ −0,0019 và Δ̄ F1-5 ≥ −0,0042 (cutpoint cross-fit theo
   `sample_key`, seed 16, như các pilot trước).

**ASL TỆ HƠN** khi Δ macro AP < 0 ở cả hai seed và Δ̄ < −0,0030, hoặc khi điều kiện 4 không đạt.

**Mọi trường hợp còn lại:** KHÔNG XÁC LẬP.

**Chỉ để mô tả, không dùng để quyết định:**

- AUROC "có bệnh".
- Recall ở precision mốc. Mốc là precision theo từng finding của mô hình đang báo cáo
  (`run_20261005_paper`), cutpoint cross-fit trên val. Chỉ tính trên 12 finding có mốc lớn hơn 0
  (Pleural Other và Fracture có mốc bằng 0). Chỉ số này rất nhiễu, NB 0,022.
- Recall và precision P/U dưới cutpoint.
- Các số argmax (giao thức của bài báo). Lưu ý ASL dời điểm vận hành về phía dương tính, nên argmax
  sẽ đổi mạnh mà không nói gì về xếp hạng.
- AP theo từng finding hiếm: Enlarged Cardiomediastinum, Fracture, Pleural Other, Lung Lesion,
  Consolidation, Pneumothorax.
- Macro recall và AUROC ba lớp.

**Nếu ASL TỐT HƠN:** không chạy full ngay. Bài học từ TK (pass ở pilot nhưng không lặp lại) là phải
hỏi user chạy thêm seed 7 và 23, giữ nguyên quy tắc này cùng NB. Full run và Stage 2 chỉ bàn sau khi
4 seed cùng đạt.

**Nếu không xác lập hoặc tệ hơn:** ghi lại kết quả, dừng hướng ASL. Hướng tiếp theo cần user quyết
định là mở khóa encoder nông ngay từ pha 1b.

## 5. Giới hạn đã biết trước

- Chỉ là pilot: 1b, 60.000 study, 2 epoch, 2 seed, chỉ val.
- Một bộ tham số theo khuyến nghị của ASL. Không dò tham số, vì dò sẽ cần thêm nhiều arm và làm loãng
  quy tắc đọc.
- Nhãn tự động sai khoảng 25% số ca "bỏ sót" (theo audit), đó là trần chung của mọi arm.
- Tăng AP chưa có nghĩa là Stage 2 tốt hơn. Câu hỏi đó cần một pilot Stage 2 riêng.

## Execution report

(chưa chạy)
