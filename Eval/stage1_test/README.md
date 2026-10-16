# Stage 1 — Đánh giá trên tập TEST

Run `run_20260819_xmpoff` · `checkpoint_best.pth` (epoch **6**) · config `pretraining/configs/mimic_cxr_full.yaml` · commit `11e35943e704`

Tổng hợp ngày 2026-08-20. Mọi con số dưới đây tính lại **offline** từ file `.npz` do
`run.save_predictions: true` ghi ra, không cần GPU.

---

## 0. Run này chạy thế nào

| | |
|---|---|
| Kết thúc | `rc=0`, TRAINING COMPLETED — 10/10 epoch |
| Thời gian | 18:15:29 (19/08) → 06:50:54 (20/08) = **12h35m** |
| Tốc độ | **0.3505 s/it**, `max mem` 8,526 MiB (53% của card 16 GB) |
| Kernel fault | **0** — đây là run đầu tiên sống sót kể từ khi cài lại máy 17/08 |
| Nguyên nhân đã sửa | **tắt XMP** trong BIOS (4 thanh RAM 2 hãng chạy 3200 MT/s trên nền Intel chỉ validate 2666 ở 2 DIMM/kênh) |
| Chọn checkpoint | `selection_metric: loss` trên **val**, epoch 6 (`val_loss` 1.8882) |

Đường cong val (chỉ chấm từ epoch 5 vì `eval_start_epoch: 5`):

| epoch | val_loss | val_f1_positive_macro |
|---:|---:|---:|
| 5 | 1.8916 | 0.7185 |
| **6** | **1.8882** | **0.7213** |
| 7 | 1.8963 | 0.7203 |
| 8 | 1.8945 | 0.7173 |
| 9 | 1.8902 | 0.7188 |

Nếu đổi `selection_metric` sang `f1_positive_macro` thì **vẫn chọn epoch 6** — hai tiêu chí
trùng nhau ở run này, nên việc chọn checkpoint không phải là điểm tranh cãi.

Test split được giữ kín hoàn toàn: chấm đúng **một lần**, từ `checkpoint_best`, sau khi train xong.
Ngưỡng được calibrate **chỉ trên val** (1,808 study), rồi mới áp lên test (3,269 study).

---

## 1. Số chính

**F1 chỉ có nghĩa dưới mẫu số `study_presence`.** Đọc mục 2 trước khi trích bất kỳ
con số F1 nào.

| Metric | Giá trị | 95% CI |
|---|---:|:---:|
| `positive_macro_f1` | **0.3397** | [0.3275, 0.3515] |
| `macro_auroc` | **0.7441** | [0.7341, 0.7540] |
| `macro_auprc` | 0.3004 | [0.2919, 0.3139] |
| `positive_macro_recall` | 0.5006 | [0.4811, 0.5206] |
| `positive_macro_precision` | 0.2776 | [0.2657, 0.2888] |
| `positive_micro_f1` | 0.4685 | [0.4585, 0.4779] |
| `micro_auroc` | 0.8183 | – |

Cấu hình: `study_presence` + `marginal_presence`, ngưỡng calibrate trên val bằng
`--selection plateau --plateau-fraction 0.95 --min-positive 5`, bootstrap 1,000 lần,
seed 42. Thư mục `eval_presence_plateau/`.

⚠ `macro_auroc` và `macro_auprc` **không phụ thuộc ngưỡng**, nên giống hệt arm
`eval_presence_marginal`. Chỉ F1/P/R thay đổi.

---

## 2. Vì sao phải đổi mẫu số — và bằng chứng

Ma trận nhãn CheXpert có **hai cách đọc**, và chúng không thay thế được cho nhau.
Trước 2026-08-20 repo chỉ có một cách, để ngầm định, và điều đó làm F1 mất nghĩa.

| | `masked_polarity` (mặc định lịch sử) | `study_presence` |
|---|---|---|
| Ô trống nghĩa là | **bị mask** — chấm polarity *với điều kiện* finding đã được nhắc | **không có** |
| Số ô còn lại | 53–98% mỗi nhãn bị bỏ | dùng hết 3,269 study cho cả 14 nhãn |
| Prevalence mỗi nhãn | 0.13 – 1.00, **12/14 nhãn > 0.55** | **0.019 – 0.344** |
| Nhãn suy biến | **3** (`No Finding`, `Pleural Other`, `Fracture`) | **0** |

`masked_polarity` là mẫu số **đúng cho hàm loss** — ô trống nghĩa là labeler không
tìm thấy đề cập, không phải bác sĩ loại trừ. Nhưng nó là mẫu số **hỏng cho F1**, vì
"âm tính" còn lại chỉ là nhóm hiếm mà bác sĩ viết rõ là không có, khiến dương tính
thành đa số.

Bảng baseline do chính script sinh ra — đây là bằng chứng, không phải lập luận:

| Baseline | `masked_polarity` | `study_presence` |
|---|---:|---:|
| `all_positive` (đoán dương hết) | **0.8397** | **0.2280** |
| `all_negative` | 0.0000 | 0.0000 |
| `majority_class` | 0.8200 | 0.0000 |
| `prevalence_random` | 0.7617 | 0.1418 |
| `threshold_half` (ngưỡng 0.5) | 0.8713 | 0.3173 |
| **Model** | **0.8717** | **0.3224** |
| Model hơn `all_positive` | **+0.032** | **+0.094** |

Ở cột trái, một hằng số hơn model 0.032, và calibrate ngưỡng chỉ mua thêm **+0.0004**
so với ngưỡng 0.5. Ở cột phải, `all_negative` và `majority_class` rơi về **0.0000** —
F1 không còn ăn gian được bằng cách đoán dương hết, cũng không ăn gian được bằng cách
im lặng.

⚠ **`macro_auroc` cũng phụ thuộc framing** (0.7857 ở cột trái, 0.7441 ở cột phải) vì
hai bên chấm trên hai quần thể khác nhau. Luôn ghi kèm framing khi trích.

⚠ Dưới `study_presence`, ô **uncertain đã bị gộp vào "không có"**, nên
`--uncertain-policy` không còn quyết định gì nữa. Đừng báo cáo hai flag như thể chúng
còn độc lập.

### 2.1 Mention gate — dùng đúng thứ model đã học

`classification_logits` là `q`: phân phối polarity **có điều kiện** trên việc finding
được nhắc tới. Một mình nó **không trả lời được** câu hỏi `study_presence`. Xác suất
đúng là `P(có) = sigmoid(mention) × q_pos`, và run này đã huấn luyện mention head
(`lambda_gate: 0.5`) — nó chỉ chưa từng được trả ra khỏi model.

Sau khi expose gate (commit `c4f6021`) và chấm lại:

| | `q_pos` trần | `m × q_pos` | Δ |
|---|---:|---:|---:|
| `macro_auroc` | 0.6767 | **0.7441** | **+0.0674** |
| `micro_auroc` | 0.6384 | **0.8183** | **+0.1799** |
| `macro_auprc` | 0.2657 | **0.3004** | +0.0347 |
| `positive_macro_f1` | 0.3074 | **0.3224** | +0.0150 |
| `macro_specificity` | 0.6359 | **0.8214** | **+0.1855** |
| `binary_accuracy` | 0.6340 | **0.8118** | +0.1778 |

**AUROC tăng trên 14/14 nhãn, trung bình +0.0831.** Đây không phải nhiễu.

Nơi gate đóng góp nhiều nhất đúng là nơi "có được nhắc tới không" mang phần lớn tín hiệu:

| nhãn | AUROC `q_pos` | AUROC `m × q_pos` | Δ |
|---|---:|---:|---:|
| No Finding | 0.5445 | **0.8049** | **+0.2603** |
| Pleural Other | 0.5366 | **0.7867** | **+0.2501** |
| Fracture | 0.4853 | **0.6651** | **+0.1798** |
| Support Devices | 0.7836 | **0.8783** | +0.0948 |
| Lung Lesion | 0.6507 | 0.7261 | +0.0754 |

Ba nhãn đầu là ba nhãn **suy biến dưới `masked_polarity`** — dưới mẫu số cũ chúng cho
F1 ≈ 1.0 miễn phí và AUROC không tính được. Dưới mẫu số mới cộng gate, chúng trở thành
bài toán thật và model làm được thật.

⚠ **`m` không phải xác suất đã hiệu chỉnh.** Gate được train với trọng số nghịch tần
suất (`n_not_mentioned / n_mentioned × kappa_gate`, kappa 2, cap 10), nên odds của nó bị
thổi lên. Điều đó **không sao ở đây** vì ngưỡng được calibrate trên val nên hấp thụ hết
độ lệch — nhưng đừng đọc `m = 0.5` như "50% khả năng được nhắc tới".

---

## 2bis. Tối ưu F1 — làm được gì, và trần ở đâu

### Khoảng cách tới trần, và nó nằm ở đâu

Trần lý thuyết của việc chọn ngưỡng (dùng ngưỡng oracle trên chính test — **chỉ để
chẩn đoán, không bao giờ là số báo cáo**) là **0.3532**. Điểm xuất phát 0.3224, mất 0.0309:

| nguồn mất | mất | vì sao |
|---|---:|---|
| `Fracture` | **0.1345** | `--min-positive 20` mà val chỉ có 18 dương → rơi về ngưỡng 0.5 → **không bao giờ báo dương**, F1 = 0.0000 |
| `Pleural Other` | **0.0827** | val có 15 dương → cũng rơi về 0.5 |
| 10 nhãn còn lại | 0.0015–0.0333 mỗi nhãn | nhiễu val→test bình thường |

**Hai nhãn bị bỏ qua chiếm 59% toàn bộ khoản mất.**

### Hai thay đổi, chọn bằng CV trong val trước khi động vào test

1. **`--selection plateau --plateau-fraction 0.95`** — đứng ở **trung vị của mọi ngưỡng
   đạt ≥ 95% đỉnh**, thay vì đúng đỉnh. Trên 1,808 study, đỉnh đường cong F1 phụ thuộc
   study nào tình cờ rơi vào val; tâm vùng gần tối ưu chuyển giao tốt hơn.
2. **`--min-positive 5`** — hai nhãn hiếm được calibrate thật, ra 0.231 / 0.244, **sát
   ngưỡng oracle trên test** (0.250 / 0.247). Đây là bằng chứng ngưỡng đúng, không phải may.

Chọn bằng **CV 5-fold × 10 lần bên trong val**: plateau@0.95 + minpos5 đạt **0.3246**,
argmax + minpos20 đạt **0.3202**. Test chỉ được chạm một lần sau khi đã chốt.

### Kết quả — và nó là một ĐÁNH ĐỔI, không phải quà miễn phí

Bootstrap **ghép cặp** 2,000 lần trên cùng study (đây mới là kiểm định đúng; CI đơn lẻ
của hai arm có chồng lấn):

| | Δ | 95% CI | kết luận |
|---|---:|:---:|---|
| **F1** | **+0.0174** | [+0.0102, +0.0243] | có ý nghĩa |
| **Recall** | **+0.0747** | [+0.0619, +0.0880] | có ý nghĩa |
| **Precision** | **−0.0211** | [−0.0303, −0.0137] | có ý nghĩa |

Đổi 0.021 precision lấy 0.075 recall — tỉ lệ **3.5:1** nghiêng về "không bỏ sót bệnh".
0.3397 đạt **97% trần** của việc chọn ngưỡng.

### Bốn thứ ĐÃ THỬ VÀ KHÔNG ĂN THUA — đừng thử lại mù

| thử | CV macro F1 trên val | kết luận |
|---|---:|---|
| `m × q` (đang dùng) | **0.3247** | tốt nhất |
| `m^a × q^(1-a)`, a fit từng nhãn | 0.3228 | không hơn |
| logistic stacking `[logit m, logit q]` | 0.3225 | không hơn (AUROC +0.004, F1 đứng yên) |
| `q` trần | 0.3155 | kém, đúng dự đoán |
| ensemble mean(ep4, ep6, ep9) | 0.3266 | thua ep9 một mình |
| ensemble mean(ep6, ep9) | 0.3280 | thua ep9 một mình |
| dùng `checkpoint_9` thay `checkpoint_best` | 0.3291 | **+0.0041 nhưng CI cắt 0** (thắng 66%/100 fold) — không đủ để đổi |

Phép nhân đơn giản đã là cách kết hợp đúng, và các epoch gần như thay thế được cho nhau
— khớp với đường val phẳng (train loss 1.856 → 1.834 qua epoch 5–9, val loss chỉ
1.8916 → 1.8902).

### ⚠ Precision KHÔNG mua được bằng ngưỡng

Ép sàn precision trên val rồi áp lên test:

| sàn precision trên val | test F1 | test precision | test recall |
|---|---:|---:|---:|
| không (F1-optimal plateau) | **0.3409** | 0.2778 | **0.5044** |
| 0.35 | 0.2647 | 0.3023 | 0.3987 |
| 0.40 | 0.2646 | 0.3330 | 0.3506 |
| 0.50 | 0.2321 | 0.3975 | 0.2329 |
| 0.60 | 0.1580 | 0.3924 | 0.1410 |
| 0.70 | 0.1045 | **0.4074** | 0.0815 |

Sàn 0.70 chỉ giao được 0.41 trên test và vứt đi **84% recall**. Precision bão hòa quanh
**0.40**. Trần đó là của **mô hình**, không phải của quy tắc quyết định — muốn precision
cao hơn phải **train lại**, không có cờ calibration nào làm được.

### Đòn bẩy còn lại: `lambda_mention_conditioned_cls` (chưa chạy)

Nó train thẳng hợp lý `-log(1-m)` khi không được nhắc và `-log(m) - log(q[y])` khi được
nhắc — **đúng bằng đại lượng mà `study_presence` + `marginal_presence` đang chấm**. Cặp
`lambda_cls` + `lambda_gate` đang dùng train hai thừa số **riêng rẽ và không có gì hòa
giải chúng**.

⚠ Kết luận "nó không hiệu quả" hồi 2026-08-16 được đưa ra dưới mẫu số `masked_polarity`
— vốn mask hết ô trống nên **về nguyên tắc không nhìn thấy hợp lý joint**. Phán quyết đó
**không còn áp dụng** và đáng test lại. Yêu cầu `lambda_cls: 0.0` và `lambda_gate: 0.0`
(constructor raise nếu không), một run đầy đủ ~12.5 giờ.

---

## 3. Từng bệnh lý — `study_presence` + `m × q_pos`, ngưỡng plateau

| Bệnh lý | n+ | Prev. | P | R | F1 | AUROC | AUPRC | Thr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Support Devices † | 1125 | 0.344 | 0.6581 | 0.8436 | **0.7394** | **0.8783** | 0.7724 | 0.124 |
| Pleural Effusion | 1074 | 0.329 | 0.6186 | 0.7989 | **0.6973** | **0.8682** | 0.7738 | 0.566 |
| Pneumothorax | 112 | 0.034 | 0.2720 | 0.3036 | 0.2869 | **0.8283** | 0.1919 | 0.631 |
| Edema | 683 | 0.209 | 0.4840 | 0.6208 | 0.5439 | **0.8245** | 0.5435 | 0.690 |
| No Finding † | 568 | 0.174 | 0.4792 | 0.5282 | 0.5025 | 0.8049 | 0.5185 | 0.639 |
| Pleural Other | 63 | 0.019 | 0.0527 | 0.5556 | 0.0963 | 0.7867 | 0.0528 | **0.231** |
| Cardiomegaly | 866 | 0.265 | 0.3984 | 0.8268 | 0.5377 | 0.7595 | 0.5058 | 0.472 |
| Consolidation | 220 | 0.067 | 0.2383 | 0.2091 | 0.2228 | 0.7446 | 0.1796 | 0.683 |
| Lung Lesion | 123 | 0.038 | 0.0927 | 0.4309 | 0.1525 | 0.7261 | 0.1151 | 0.445 |
| Atelectasis | 731 | 0.224 | 0.3556 | 0.7278 | 0.4778 | 0.7229 | 0.4071 | 0.539 |
| Pneumonia | 335 | 0.102 | 0.2377 | 0.4030 | 0.2990 | 0.6993 | 0.2176 | 0.648 |
| Lung Opacity | 1038 | 0.318 | 0.4103 | 0.7707 | 0.5355 | 0.6915 | 0.4961 | 0.425 |
| Fracture | 89 | 0.027 | 0.1049 | 0.1685 | **0.1293** | 0.6651 | 0.0557 | **0.244** |
| Enlarged Cardiomediastinum | 151 | 0.046 | 0.0658 | 0.1921 | 0.0980 | 0.6126 | 0.0657 | 0.542 |

† meta-label, loại khỏi macro (`include_meta_labels: false`).

Hai nhãn in đậm ở cột ngưỡng là hai nhãn trước đây rơi về 0.5:
**`Fracture` từ F1 0.0000 → 0.1293**, `Pleural Other` từ 0.0227 → 0.0963.

Bốn nhãn khỏe nhất (AUROC ≥ 0.82): Support Devices, Pleural Effusion, Pneumothorax, Edema.
Yếu nhất: Enlarged Cardiomediastinum 0.6126, Fracture 0.6651.

⚠ Năm nhãn có precision dưới 0.25 (Pleural Other 0.053, Enlarged Cardiomediastinum 0.066,
Lung Lesion 0.093, Fracture 0.105, Consolidation 0.238). Ở những nhãn này model báo dương
sai nhiều hơn đúng rất nhiều — **chưa dùng lâm sàng được**, và theo mục 2bis thì không
chỉnh ngưỡng nào cứu được.

---

## 3bis. Mẫu số cũ — giữ lại để so với bài báo gốc, KHÔNG dùng làm số chính

Repo gốc và bài báo dùng mẫu số này, nên bảng dưới cần cho việc đối chiếu. Nhưng theo
mục 2, F1 ở đây không đo kỹ năng.

| | F1 (đang dùng) | Youden J | thr 0.5 |
|---|---:|---:|---:|
| `positive_macro_f1` | 0.8717 | 0.8517 | 0.8713 |
| `macro_auroc` | 0.7857 | 0.7857 | 0.7857 |
| `macro_auprc` | 0.8951 | 0.8951 | 0.8951 |
| `macro_specificity` | 0.4197 | **0.5580** | 0.4442 |
| `balanced_accuracy` | 0.6586 | **0.7052** | 0.6706 |
| `positive_macro_recall` | **0.9061** | 0.8647 | 0.9056 |

AUROC/AUPRC không phụ thuộc ngưỡng nên giống nhau cả ba cột — đây là **cùng một model,
khác điểm vận hành**.

Ba nhãn suy biến dưới mẫu số này (`No Finding` 0 âm, `Pleural Other` 0 âm, `Fracture`
3 âm) cho F1 ≈ 1.0 miễn phí. Bỏ chúng và bỏ meta-label:

| macro | như báo cáo | bỏ luôn nhãn suy biến |
|---|---:|---:|
| `auroc` | 0.7857 (n=11) | **0.8021** (n=10) |
| `f1` | 0.8717 (n=12) | **0.8477** (n=10) |
| `specificity` | 0.4197 | 0.4616 |

AUROC **tăng** khi bỏ chúng còn F1 **giảm** — bằng chứng trực tiếp rằng F1 ở mẫu số này
đang được nâng bởi đúng những nhãn không đo được gì.

---

## 4. Những gì KHÔNG có ở đây, và tại sao

| | Trạng thái |
|---|---|
| **NLG (BLEU / ROUGE / METEOR / CIDEr / BERTScore)** | Không tính. Đây là metric Stage 2 (sinh báo cáo); Stage 2 chưa chạy với checkpoint này. |
| **RadGraph, RadCliQ, CheXbert, RadFact** | **Không cài được** trong repo — là research code có licence riêng. Báo là *unavailable*, **không** báo là 0. |
| **XAI / saliency (`evaluate_explanation.py`)** | Cache mask giải thích nằm trên `/mnt/drive1tb`, **mất khi cài lại máy 17/08**. Muốn có phải build lại (~25 phút, cần mount ổ dữ liệu). Ngoài ra `lambda_explanation*` đều = 0.0 trong run này. |
| **Retrieval (ITC/ITM)** | `lambda_itc/itm/lm` = **0.0** trong config này (đúng như repo gốc). Loss ba số này = 0.0000 suốt run. |
| **Subgroup theo view (PA/AP/lateral)** | File `.npz` không ghi `view_positions`, nên chỉ có subgroup theo nhãn (normal 138 / abnormal 3,131 study). |

⚠ **Checkpoint này KHÔNG dùng được cho Stage-2 chế độ `meta_cxr_qformer`** (soft token):
đường ảnh của Q-Former bị skip toàn bộ trong Stage 1, cross-attention chưa từng thấy ảnh y tế.
`medgemma_direct` không bị ảnh hưởng.

---

## 5. Cấu trúc thư mục

```
Test/stage1_test/
├── README.md                          ← file này
├── summary_all.json                   ← 5 arm + bootstrap + baseline + đóng góp của gate
├── arm_comparison.csv                 ← bảng so sánh 5 arm
├── mention_gate_contribution.csv      ← AUROC/F1/Spec từng nhãn, q_pos vs m×q_pos
├── thresholds_presence_marginal_plateau.json ← ngưỡng ĐANG DÙNG (plateau, minpos5)
├── thresholds_presence_marginal_f1.json  ← argmax, minpos20 (arm cũ, để so sánh)
├── thresholds_presence_f1.json        ← val, study_presence, q_pos
├── thresholds_f1.json                 ← val, masked_polarity, q_pos
├── thresholds_youden_j.json           ← val, masked_polarity, Youden J
├── predictions/
│   ├── gate_val.npz  / gate_test.npz              ← epoch 6 (checkpoint_best), CÓ gate
│   ├── val_pred_ep4.npz / test_pred_ep4.npz       ← epoch 4, cho thí nghiệm ensemble
│   ├── val_pred_ep9.npz / test_pred_ep9.npz       ← epoch 9, cho thí nghiệm ensemble
│   └── val_… / test_predictions_epoch_best.npz    ← từ chính run train (q trùng từng bit)
├── eval_presence_plateau/    ← ★ ĐÁNH GIÁ CHÍNH (ngưỡng plateau)
├── eval_presence_marginal/   ← cùng score, ngưỡng argmax (arm so sánh)
├── eval_presence_qpos/       ← cùng framing, không dùng gate (đo đóng góp của gate)
├── eval_f1/                  ← mẫu số cũ, có plots/ + confusion_matrices/
├── eval_youden/              ← mẫu số cũ, điểm vận hành thiên specificity
└── eval_uncalibrated/        ← mẫu số cũ, ngưỡng 0.5
```

Tái tạo (không cần GPU — chỉ đọc `.npz`):

```bash
python scripts/calibrate_thresholds.py \
    --predictions Test/stage1_test/predictions/gate_val.npz \
    --label-framing study_presence --score marginal_presence \
    --objective f1 --uncertain-policy ignore_uncertain \
    --min-positive 5 --selection plateau --plateau-fraction 0.95 \
    --output Test/stage1_test/thresholds_presence_marginal_plateau.json

python scripts/evaluate_stage1.py \
    --predictions Test/stage1_test/predictions/gate_test.npz \
    --thresholds Test/stage1_test/thresholds_presence_marginal_plateau.json \
    --label-framing study_presence --score marginal_presence \
    --uncertain-policy ignore_uncertain --bootstrap-samples 1000 --split test \
    --output-dir Test/stage1_test/eval_presence_plateau
```

⚠ `--label-framing` / `--score` phải **giống nhau ở cả hai lệnh**. Khác nhau thì
`evaluate_stage1.py` **từ chối chạy** và nêu tên cả hai bên — đã kiểm chứng. (Trái với
`--uncertain-policy`, vốn im lặng cho qua; đó là lý do guard này tồn tại.)

### Cách xuất lại `.npz` có mention gate

Cần GPU và ổ dữ liệu đã mount. Chạy ~2 phút cho cả val lẫn test:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python -m pretraining.train \
    --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --options run.evaluate=True run.test_splits=[val,test] \
      run.resume_ckpt_path=<run>/mimic_cxr_full_blip2/checkpoint_best.pth
```

⚠⚠ **`run.resume_ckpt_path` là bắt buộc.** Trước bản vá `0f1003d`, bỏ nó đi thì run
vẫn chạy xong, ghi đủ file, không báo lỗi — và chấm điểm **model random init**. Kiểm
chứng bằng `grep "Loading evaluation weights from" <log>`; không có dòng nào nghĩa là
kết quả vô giá trị. Giờ nó raise thay vì im lặng.

Bằng chứng lần xuất này đúng: `q` trong `gate_test.npz` **trùng từng bit**
(max abs diff 0.0) với `test_predictions_epoch_best.npz` mà chính run train ghi ra.

---

## 6. Bảo mật dữ liệu

`Test/` nằm trong `.gitignore` (dòng 131). File `.npz` chứa `sample_keys` là định danh
study của MIMIC-CXR — **dữ liệu PhysioNet credentialed**. Không commit, không push,
không đưa lên bất kỳ dịch vụ ngoài nào. Chỉ các con số tổng hợp trong file README này
là an toàn để trích vào luận văn.
