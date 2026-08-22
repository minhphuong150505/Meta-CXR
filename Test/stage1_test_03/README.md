# Stage-1 test evaluation — phiên bản 03

**Kết luận: 5 epoch huấn luyện thêm KHÔNG cải thiện mô hình. Giữ nguyên phiên bản 02 (epoch 9) làm mô hình cuối cùng.**

| | |
|---|---|
| Run | `run_20260821_ext` — resume từ `run_20260820_ft`, `run.max_epoch` 10 → 15 |
| Thời gian | 2026-08-21, 08:13:53 → 15:45:08 (**7h32m**), `rc=0`, 0 restart, 0 kernel fault |
| Checkpoint được chấm | `checkpoint_14.pth` (epoch cuối) |
| `checkpoint_best` theo `val_loss` | vẫn là **epoch 9** — không epoch nào trong 10–14 vượt được |
| Test split | 3,269 study, chấm **đúng một lần** |
| Đối chứng | phiên bản 02 = `run_20260820_ft/checkpoint_best.pth` (epoch 9) |

---

## 1. Vì sao lại chấm epoch 14 chứ không phải `checkpoint_best`

`checkpoint_best.pth` của run này **chính là epoch 9 của phiên bản 02** — chấm lại nó chỉ ra đúng số cũ. Câu hỏi thật là: 5 epoch thêm có tạo ra mô hình nào tốt hơn không?

`val_loss` nói **không**:

| ep | train | val_loss | gap | val_cls |
|---|---|---|---|---|
| 9 ★ | 1.8480 | **1.8843** | +0.0363 | 0.9660 |
| 10 | 1.8450 | 1.8909 | +0.0459 | 0.9696 |
| 11 | 1.8470 | 1.8877 | +0.0407 | 0.9696 |
| 12 | 1.8410 | 1.8921 | +0.0511 | 0.9719 |
| 13 | 1.8380 | 1.8843 | +0.0463 | 0.9692 |
| 14 | 1.8360 | 1.8877 | **+0.0517** | 0.9716 |

Train đi xuống đều, val đứng yên, khoảng cách nới từ +0.036 lên +0.052 — dấu hiệu overfit nhẹ.

Nhưng `val_loss` là tổng có trọng số **bị các nhãn phổ biến chi phối** (điểm yếu đã ghi trong `CLAUDE.md` về `selection_metric: loss`). Chấm lại 6 epoch trên val bằng thước thật (`study_presence` + `marginal_presence`, threshold-free):

| ep | macro AUROC | micro AUROC | macro AUPRC | macro AUROC (bỏ gate) |
|---|---|---|---|---|
| 9 ★ | 0.7846 | **0.8461** | 0.3068 | **0.7354** |
| 10 | 0.7846 | 0.8339 | 0.3060 | 0.7332 |
| 11 | 0.7805 | 0.8420 | 0.3069 | 0.7197 |
| 12 | 0.7834 | 0.8416 | 0.3012 | 0.7201 |
| 13 | 0.7865 | 0.8419 | 0.3062 | 0.7254 |
| **14** | **0.7927** | 0.8406 | **0.3101** | 0.7304 |

Bootstrap ghép cặp trên 1,808 study val, 2,000 lần lấy mẫu:
**Δ macro AUROC = +0.0082, 95% CI [+0.0015, +0.0148], P(ep14 > ep9) = 0.995.**

CI không cắt 0, nên **epoch 14 được chọn dựa trên validation** và mang sang test.

⚠ Đây là **lựa chọn hậu nghiệm**: tiêu chí chọn checkpoint được đổi từ `val_loss` sang macro AUROC *sau khi* nhìn kết quả val. Không có rò rỉ test — mọi quyết định vẫn nằm trong val — nhưng phải khai báo đúng như vậy khi viết báo cáo.

---

## 2. Kết quả test — lợi ích trên val KHÔNG chuyển sang test

`study_presence` + `marginal_presence`, ngưỡng calibrate trên val epoch 14 bằng quy tắc plateau (`--selection plateau --plateau-fraction 0.95 --min-positive 5`).

| chỉ số | run02 (ep9) | run03 (ep14) | Δ, 95% CI | có ý nghĩa? |
|---|---:|---:|:---:|:---:|
| `macro_auroc` | **0.7643** | 0.7638 | −0.0005 [−0.0043, +0.0031] | không |
| `positive_macro_f1` | **0.3542** | 0.3474 | −0.0067 [−0.0161, +0.0024] | không |
| `positive_macro_precision` | 0.2931 | 0.2944 | +0.0015 [−0.0063, +0.0096] | không |
| `positive_macro_recall` | **0.5373** | 0.4544 | **−0.0826 [−0.0978, −0.0675]** | **có** |
| `macro_specificity` | 0.8020 | **0.8264** | **+0.0244 [+0.0217, +0.0272]** | **có** |
| `micro_auroc` | **0.8166** | 0.8056 | −0.0110 | — |
| `macro_auprc` | **0.3203** | 0.3179 | −0.0024 | — |

Bootstrap ghép cặp trên cùng 3,269 study test, 2,000 lần lấy mẫu. Chi tiết: `bootstrap_vs_run02.json`.

**Đọc bảng này:**

- **Ba chỉ số đo chất lượng mô hình đều có CI cắt 0.** AUROC, F1 và precision không thay đổi. Lợi ích +0.0082 đo được trên val **không lặp lại** trên test (−0.0005).
- **Hai chỉ số có ý nghĩa thống kê chỉ là dịch điểm vận hành**, không phải mô hình tốt hơn: ngưỡng calibrate lại trên val epoch 14 ra dè dặt hơn, nên recall tụt 0.083 và specificity tăng 0.024. Đánh đổi 3.4:1 nghiêng về specificity — ngược hướng với cái ta cần.
- **AUROC chỉ cải thiện trên 6/14 nhãn** (`per_label_vs_run02.csv`), tức xấp xỉ tung đồng xu. So sánh: lần mở băng encoder cải thiện **14/14**. Đó là khác biệt giữa một thay đổi thật và nhiễu.

---

## 3. Vì sao val nói có mà test nói không

Ba lý do, đều đã kiểm chứng được từ chính số liệu trên:

1. **Cỡ mẫu.** Val chỉ có 1,808 study và CI của nó là [+0.0015, +0.0148] — cận dưới sát 0. Một hiệu ứng vừa đủ vượt ngưỡng ý nghĩa trên tập nhỏ là loại hiệu ứng dễ biến mất nhất khi đổi tập.
2. **Macro và micro đi ngược nhau.** Trên val, macro AUROC tăng nhưng micro AUROC giảm (0.8461 → 0.8406). Nghĩa là phần "cải thiện" nằm ở vài nhãn hiếm — nơi macro cho trọng số bằng nhau còn micro thì không. Nhãn hiếm có ít dương tính nên ước lượng AUROC của chúng nhiễu nhất.
3. **Toàn bộ chênh lệch đi qua mention gate.** Bỏ gate ra (cột cuối bảng mục 1), epoch 14 **thua** epoch 9 (0.7304 vs 0.7354). Đầu phân loại `q` không hề tốt lên; chỉ có tích `m × q` may mắn hơn trên val.

---

## 4. Nhánh `masked_polarity` — giữ để đối chiếu, KHÔNG dùng làm số chính

| chỉ số | run02 (ep9) | run03 (ep14) | Δ |
|---|---:|---:|---:|
| `positive_macro_f1` | 0.8841 | 0.8815 | −0.0027 |
| `macro_auroc` | 0.7985 | 0.7833 | −0.0151 |
| `macro_specificity` | 0.4536 | 0.4633 | +0.0097 |

Mẫu số này bỏ ~80% ô nhãn trống, đẩy prevalence lên 0.773, khiến "đoán tất cả đều dương" đạt F1 0.8397 — chỉ kém mô hình 0.044. **Không trích dẫn con số 0.88 ở bất kỳ đâu.** Lý do đầy đủ ở `../stage1_test_02/README.md` mục về label framing.

---

## 5. Kết luận và khuyến nghị

**Mô hình cuối cùng: phiên bản 02, `run_20260820_ft/checkpoint_best.pth` (epoch 9).**
macro AUROC 0.7643 · positive macro F1 0.3542 · precision 0.2931 · recall 0.5373 · specificity 0.8020.

**Chi phí của thí nghiệm này: 7h32m GPU, thu về con số 0.** Nhưng nó trả lời dứt điểm một câu hỏi mở từ 2026-08-20 — "`checkpoint_best` là epoch cuối và val loss vẫn đang giảm, vậy train thêm có giúp không?" Câu trả lời là **không**, và đã đo bằng bootstrap trên test chứ không đoán.

Kèm theo hai bài học phương pháp đáng ghi:

- **Warm restart không phải continuation.** `LinearWarmupCosineLRScheduler` không có trạng thái, nên nâng `max_epoch` khi resume làm LR nhảy 2.03× (2.0e-5 → 4.06e-5, đo được ở epoch 10 iter 0). Đây là SGDR restart. Nó không hỏng, nhưng phải gọi đúng tên.
- **Một kết quả val sát ngưỡng ý nghĩa trên 1,808 mẫu thì đừng tin.** Nếu lần sau gặp CI dạng [+0.0015, +0.0148], hãy coi đó là "chưa kết luận được" chứ không phải "có cải thiện".

**Hướng tiếp theo — cái nào cũng đáng giá hơn train thêm epoch:**

1. `model.loss.lambda_mention_conditioned_cls` — huấn luyện trực tiếp đại lượng `m × q_pos` mà `study_presence` đang chấm, thay vì huấn luyện hai đầu rời rồi nhân lại. Cần `lambda_cls: 0.0` và `lambda_gate: 0.0`, ~12.5 h.
2. Ablation tách riêng đóng góp của **kappa** và của **mở băng encoder** — hai thứ này vào cùng một run nên chưa tách được nguyên nhân.
3. Mở băng sâu hơn (`layer3` + CLIP block 8–9) — mở băng nông đã cho +0.0201 AUROC có ý nghĩa trên 14/14 nhãn, đây là hướng duy nhất đã chứng minh dịch được trần precision.
4. Cơ chế **abstention** — model precision 0.29 trên 100% ca thì không dùng được; cùng model đó precision cao trên phần nó tự tin, phần còn lại chuyển bác sĩ, thì dùng được ngay.

---

## 6. Tệp trong thư mục

| tệp | nội dung |
|---|---|
| `eval_presence_plateau/` | **kết quả chính** — `study_presence` + `marginal_presence`, epoch 14 |
| `eval_masked_polarity/` | nhánh đối chiếu, không dùng làm số chính |
| `bootstrap_vs_run02.json` | bootstrap ghép cặp test, 2,000 lần, 5 chỉ số |
| `version_comparison.csv` | run02 vs run03, cả hai framing, 8 chỉ số |
| `per_label_vs_run02.csv` | so sánh từng nhãn trong 14 nhãn |
| `summary_all.json` | gộp tất cả + metadata |
| `predictions/` | `.npz` val epoch 9–14 và test epoch 14 |
| `thresholds_*.json` | ngưỡng calibrate trên **val epoch 14**, không bao giờ trên test |

⚠ Thư mục `Test/` nằm trong `.gitignore`. Các tệp `.npz` chứa `sample_keys` (định danh study) — **không bao giờ commit**.
