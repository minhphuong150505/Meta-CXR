# Evaluator validation

Ngày chạy: 2026-07-22 · Branch `feature/complete-evaluator` · Python 3.12, numpy 2.5.1

Mọi con số dưới đây là **output thật** của code trong repo, chạy trên dữ liệu tổng hợp nhỏ do
chính tài liệu này định nghĩa. **Không có số nào của mô hình thật** — chưa có lần inference nào
trên MIMIC-CXR được thực hiện. Mục đích là chứng minh evaluator hành xử đúng, không phải báo cáo
chất lượng mô hình.

Tái lập:

```bash
python -m pytest tests/test_classification_metrics.py tests/test_threshold_calibration.py \
                 tests/test_generation_metrics.py tests/test_evaluation_integration.py \
                 tests/test_stage1_eval_hook.py tests/test_evaluation_config.py
```

---

## Phần A — Classification

### A1. Accuracy cao trong khi positive F1 bằng 0

200 study × 3 pathology, prevalence 5%. Mô hình dự đoán **toàn negative**.

```
binary_accuracy    = 0.9617
positive_macro_f1  = 0.0000
positive_macro_recall = 0.0000
```

✅ **Đạt.** Đây là lý do tồn tại của bảng baseline: 96,17% accuracy trông rất tốt trong khi mô
hình không phát hiện được một bất thường nào. Hàng `all_negative` trong
`baseline_table()` khiến điều này lộ ra ngay trong report.

Lưu ý thêm: `positive_macro_precision` ở tình huống này là **`nan`** (undefined), không phải 0 —
mô hình chưa từng dự đoán positive nên precision không xác định. Nhưng `f1` vẫn là **0**, không
phải `nan`, vì recall xác định và bằng 0. Phân biệt này được ép trong `_positive_f1()` và có test
riêng; nếu F1 bị trả `nan` ở đây thì pathology đó sẽ bị loại khỏi macro và một mô hình dự đoán
rỗng lại đạt macro hoàn hảo.

### A2. Macro F1 khác Micro F1

P0: 8 study, phân loại đúng hết. P1: 2 study, bỏ sót cả hai positive.

```
positive_macro_f1  = 0.5000     # (1.0 + 0.0) / 2
positive_micro_f1  = 0.6667     # gộp tp=2, fp=0, fn=2 -> P=1.0, R=0.5
```

✅ **Đạt.** Hai giá trị khác nhau và khớp với tính tay. Macro cho pathology hiếm trọng số ngang
bằng; micro để study nhiều hơn chi phối.

### A3. AUPRC phản ánh mất cân bằng, AUROC thì không

200 mẫu, đúng 2 positive (prevalence 1%), một positive nằm giữa bảng xếp hạng.

```
macro_auroc = 0.7525
macro_auprc = 0.5100     (prevalence = 0.0100)
```

✅ **Đạt.** AUROC 0,75 nghe khả quan; AUPRC 0,51 cho thấy thực tế kém hơn nhiều. Với positive
hiếm, AUPRC là chỉ số cần đọc. AUPRC được implement theo **step-wise average precision**
(giống `sklearn.average_precision_score`), không dùng nội suy hình thang — hình thang lệch lạc
quan trên dữ liệu mất cân bằng.

### A4. Threshold calibrated khác 0.5 khi cần

6 study, toàn bộ score nằm **dưới** 0.5.

```
threshold 0.5   -> positive_macro_f1 = 0.0000
calibrated 0.350 -> positive_macro_f1 = 1.0000
```

✅ **Đạt.** Ở ngưỡng mặc định không có gì được dự đoán positive. Calibration tìm được cutoff
0,350 và tách hoàn hảo. Đây là trường hợp `argmax` 3 lớp của evaluator cũ **không thể** xử lý,
vì nó không có tham số nào để điều chỉnh.

### A5. Uncertain policy làm thay đổi metric đúng như kỳ vọng

4 study: 1 positive, 1 uncertain, 2 negative; cả positive lẫn uncertain đều có score 0,9.

| Policy | positive_macro_f1 | tp | fp | n_valid |
|---|---:|---:|---:|---:|
| `three_class` | 0.6667 | 1 | 1 | 4 |
| `uncertain_as_positive` | 1.0000 | 2 | 0 | 4 |
| `uncertain_as_negative` | 0.6667 | 1 | 1 | 4 |
| `ignore_uncertain` | 1.0000 | 1 | 0 | **3** |

✅ **Đạt.** Cả bốn policy cho kết quả đúng như định nghĩa:
- U-Ones biến mẫu uncertain thành true positive.
- U-Zeros biến nó thành false positive.
- U-Ignore **loại hẳn** mẫu đó (`n_valid` giảm từ 4 xuống 3) — đây là điểm phân biệt duy nhất
  giữa `ignore_uncertain` và `uncertain_as_negative` ở đây.
- `three_class` trùng `uncertain_as_negative` trên metric nhị phân, đúng như đã ghi trong
  docstring của `uncertain_policy.py`.

### A6. Metric legacy không đổi

`tests/test_stage1_eval_hook.py` chạy `ImageTextPretrainTask.evaluation` với model giả:

- P0 dự đoán hoàn hảo và có positive; P1 không có positive nào.
- `f1_positive_macro` (legacy) = **0.5000** — giữ nguyên hành vi cũ, kể cả khuyết điểm.
- `f1_positive_macro_defined_only` (mới) = **1.0000** — loại pathology không có positive.

✅ **Đạt.** Selection metric giữ nguyên bit-for-bit, nên checkpoint cũ và log cũ vẫn so sánh
được. Metric đã sửa được báo cáo song song dưới tên khác.

---

## Phần B — Report generation

### B1. Report trùng khớp hoàn toàn

```
BLEU-1 = 1.0000   BLEU-4 = 1.0000   ROUGE-L = 1.0000
```

✅ **Đạt.**

### B2. Sai negation nhưng lexical overlap vẫn cao ⚠️ **quan trọng nhất**

- Reference: `There is no pneumothorax.`
- Generated: `There is a pneumothorax.`

```
BLEU-1  = 0.8000
ROUGE-L = 0.8000
flags   = ['possible_false_positive_finding', 'possible_negation_error']
negation_mismatches = ['Pneumothorax']
```

✅ **Đạt, và đây là luận điểm trung tâm của toàn bộ tầng đánh giá.** Hai báo cáo **đối lập hoàn
toàn về lâm sàng** vẫn đạt BLEU-1 và ROUGE-L bằng 0,80. Không có ngưỡng nào trên metric lexical
phát hiện được lỗi này. Chỉ phân tích ở mức mệnh đề mới bắt được.

Hệ quả cần nhớ khi viết khóa luận: **một bảng chỉ có BLEU/ROUGE/METEOR/CIDEr/BERTScore không
chứng minh được tính đúng đắn lâm sàng.**

### B3. Report rỗng

```
bleu_4 = 0.0000   rouge_l = 0.0000   (không crash)
flags  = ['empty_output', 'possible_omitted_finding']
```

✅ **Đạt.** Điểm 0 ở đây là điểm 0 **thật** cho một output rỗng, khác hoàn toàn với điểm 0 do
exception mà `compute_nlg` cũ tạo ra.

### B4. Temporal hallucination

Generated: `The effusion is unchanged from the prior study.`

```
không có prior context -> flags = ['possible_temporal_hallucination']
                           temporal_phrases = ['prior study', 'unchanged']
có prior context       -> flags = []
```

✅ **Đạt.** Cờ bật khi input không có prior, và tự tắt khi context có prior. Tên cờ là
`possible_*` vì heuristic không thể chắc chắn.

### B5. Dependency thiếu KHÔNG trở thành điểm 0

Yêu cầu `bleu, meteor, cider, bertscore` trong môi trường không có nltk/pycocoevalcap/bert_score:

```
corpus keys      = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']
unavailable keys = ['bertscore', 'cider', 'meteor']
```

Kèm log:
```
metric 'meteor' needs the 'nltk' package, which is not installed.
Install it with: pip install nltk && python -m nltk.downloader wordnet.
The evaluator does not substitute a different implementation and does not report 0.0.
```

✅ **Đạt.** Ba metric thiếu **không xuất hiện** trong `corpus` (nên không thể lọt vào bảng kết
quả như số 0), mà nằm trong `unavailable` kèm câu lệnh cài đặt. Đây chính là lỗi S2-A trong
`evaluator_audit.md` đã được đóng.

---

## Phần C — Tính toàn vẹn của pipeline

| Kiểm tra | Kết quả |
|---|---|
| Chạy lại evaluator từ file prediction, không cần model/GPU/dataset | ✅ `test_full_stage1_pipeline_runs_without_a_model` |
| Calibration trên test bị từ chối (CLI trả exit code 2) | ✅ `test_calibration_cli_refuses_a_test_split` |
| Threshold fit trên test bị từ chối khi load (exit code 2) | ✅ `test_evaluator_refuses_thresholds_calibrated_on_test` |
| `metrics.json` không chứa token `NaN`/`Infinity`, parse được bằng `json.loads` | ✅ `test_metrics_json_is_parseable_and_has_no_nan_token` |
| Giá trị undefined ghi là `null` kèm lý do trong `skipped` | ✅ `auroc = null`, `skipped.Fracture = ["no_positive_samples"]` |
| Hai lần chạy cùng seed cho kết quả trùng khớp | ✅ `test_reevaluating_the_same_file_is_deterministic` |
| Text báo cáo không bị ghi ra trừ khi có `--include-text` | ✅ `test_full_stage2_pipeline_runs_without_a_model` |
| Sinh đủ PNG (ROC, PR, confusion ×N, bar ×2, histogram, reliability) | ✅ smoke run, 6 confusion + 6 plot |

---

## Những gì KHÔNG được kiểm chứng ở đây

- **Không có metric nào của mô hình thật.** Chưa chạy inference trên MIMIC-CXR; toàn bộ số trên
  là dữ liệu tổng hợp.
- **METEOR, CIDEr, BERTScore chưa từng chạy** trong môi trường này (thiếu package). Đường dẫn
  code của chúng chỉ được kiểm tra ở nhánh "dependency thiếu".
- **CheXbert / RadGraph / RadCliQ / RadFact chưa được implement.** `training/evaluation/clinical.py`
  raise `NotImplementedError` kể cả khi package đã cài — đây là chủ ý, xem mục F của báo cáo.
- **Các cờ `possible_*` chưa được bác sĩ xác nhận.** Chúng là heuristic từ vựng để sàng lọc.
- **Chưa chạy trên GPU.** Hook lưu prediction trong `image_text_pretrain.py` chỉ được test bằng
  model giả trên CPU.
