# Evaluator audit

Ngày: 2026-07-22 · Branch: `feature/complete-evaluator` · Base commit: `9b8e27f`

Tài liệu này ghi lại **hiện trạng đã đọc từ source**, không phải từ tài liệu cũ. Mọi shape và
mapping bên dưới đều được xác nhận trong code, kèm `file:line`.

---

## 1. Sự thật nền tảng (đã xác minh)

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Số abnormality | **14** | `training/train_eval_figure9_llm_variants_200.py:160` (`ABNORMALITIES_14`), `model/lavis/models/blip2_models/blip2_qformer.py:43` (`chexpert_cols`) |
| Số class / abnormality | **3** | `CLASS_MAP = {"negative": 0, "positive": 1, "uncertain": 2}` — `train_eval_figure9_llm_variants_200.py:176` |
| Shape logits | **`[B, 14, 3]`** | `model/lavis/tasks/image_text_pretrain.py:41-45` (validate tường minh) |
| Shape labels | **`[B, 14]`**, dtype long, giá trị là **class index** | cùng chỗ trên; `ReportDataset.py:704` |
| Uncertain gốc | CheXpert `-1` → **`2`** | `ReportDataset.py:311-315` |
| Missing label | CheXpert `NaN` → **`0` (negative)** | `ReportDataset.py:314` (`.fillna(0)`) |
| Mask mức sample | `classification_mask` (bool, 1 giá trị / study) | `ReportDataset.py:705-707` |
| Selection metric | `f1_positive_macro` | `pretraining/configs/mimic_cxr_full_l4.yaml:117` |

**Hai abnormality không phải bệnh lý:** `No Finding` (index 0) và `Support Devices` (index 13).
Chúng đang được gộp vào mọi macro average hiện tại.

---

## 2. Kiến trúc evaluator hiện tại

### Stage 1 — `model/lavis/tasks/image_text_pretrain.py` (129 dòng, toàn bộ evaluator)

```
for batch in data_loader:
    output = model(batch)
    logits [B,14,3] → argmax(dim=-1) → predictions [B,14]
    valid = sample_mask[:,None] & (labels >= 0) & (labels < 3)
    → bincount → confusion [14, 3, 3]        # (abnormality, true, pred)
→ all_reduce (DDP)
→ precision = TP / predicted.clamp_min(1)
   recall   = TP / support.clamp_min(1)
   f1       = 2PR/(P+R)
→ stats: accuracy, precision/recall/f1_macro, f1_weighted,
         precision/recall/f1_positive_macro  (= *[:, 1].mean())
```

Không có gì khác. Không lưu prediction, không probability, không AUROC/AUPRC, không threshold.

### Stage 2 — `training/train_eval_figure9_llm_variants_200.py:1298` `compute_nlg()`

```
tokenize(t) = re.findall(r"\w+", t.lower())
BLEU-1..4  : nltk corpus_bleu, SmoothingFunction
METEOR     : nltk meteor_score, trung bình theo sample
ROUGE-L    : pycocoevalcap Rouge
CIDEr      : pycocoevalcap Cider
BERTScore  : bert_score, chỉ F1, rescale_with_baseline=False, device="cpu"
```

---

## 3. Điểm ĐÚNG (giữ nguyên, không viết lại)

1. **Validate shape tường minh** (`image_text_pretrain.py:41`) — raise thay vì đoán. Tốt.
2. **Confusion matrix dựng bằng `bincount` trên GPU** — chính xác và rẻ; `all_reduce` đúng cho DDP.
3. **`sample_mask` được tôn trọng** — dòng không có nhãn CheXpert không vào confusion.
4. **`labels >= 0 & labels < num_classes`** — chặn label rác.
5. **Tách `f1_positive_macro` khỏi `f1_macro`** — ý định đúng: tránh F1 bị negative áp đảo.
6. **`threshold.json` không bao giờ được load ngầm** (`train_eval_figure9_llm_variants_200.py:178`) —
   quyết định đúng, vì file đó không có provenance.
7. Stage 2 dùng package chuẩn (nltk, pycocoevalcap, bert_score) chứ không tự implement.

---

## 4. Điểm SAI hoặc thiếu

### 🔴 S1-A — Macro average bị pha loãng bởi pathology không có mẫu positive

```python
recall = true_positive / support.clamp_min(1.0)   # image_text_pretrain.py:102
...
"f1_positive_macro": f1[:, 1].mean().item()       # :125
```

Nếu một pathology **không có mẫu positive nào** trong split, `support[:,1] == 0`, nên
`recall = 0/1 = 0` → `f1 = 0`. Giá trị 0 này **vẫn được đưa vào `.mean()`** trên đủ 14 phần tử.

Đây chính xác là trường hợp spec yêu cầu phải xử lý. Hậu quả: `f1_positive_macro` **thấp giả**, và
tệ hơn — nó **thay đổi theo phân bố của split**, nên không so sánh được giữa val và test.

`Pleural Other` và `Fracture` là các pathology hiếm, rất dễ rơi vào trường hợp này ở val (2.963 dòng).

### 🔴 S1-B — Precision = 0 khi model không bao giờ dự đoán positive

```python
precision = true_positive / predicted.clamp_min(1.0)   # :101
```

`predicted[:,1] == 0` (model chưa từng đoán positive cho pathology đó) → precision = 0/1 = 0.
Về mặt thống kê precision lúc này là **không xác định**, không phải 0. sklearn có
`zero_division` để phân biệt; ở đây thì không. Lại tiếp tục vào `.mean()`.

### 🔴 S1-C — Hoàn toàn không có AUROC / AUPRC

Evaluator chỉ dùng `argmax`. **Probability bị vứt đi ngay sau argmax** và không bao giờ được lưu.
Với dữ liệu mất cân bằng nặng như MIMIC-CXR, đây là thiếu sót nghiêm trọng nhất:
- AUPRC là metric phù hợp nhất cho positive class hiếm.
- Không có probability thì **không thể** calibrate threshold, không thể vẽ ROC/PR curve,
  không thể tính reliability diagram.

### 🔴 S1-D — Không lưu prediction

Không có file nào được ghi. Muốn tính lại bất kỳ metric nào cũng phải **chạy lại inference toàn bộ
split trên GPU**. Đây là lý do trực tiếp khiến việc phân tích sau huấn luyện gần như bất khả thi.

### 🟠 S1-E — `accuracy` là element-wise, bị negative áp đảo

```python
"accuracy": (true_positive.sum() / confusion.sum().clamp_min(1.0))   # :112
```

Đây là accuracy trên **toàn bộ 14×N phần tử flatten**. Với prevalence positive thấp, một model dự
đoán **toàn negative** vẫn đạt accuracy rất cao. Không có baseline nào để phát hiện chuyện này.

### 🟠 S1-F — `argmax` 3 lớp không tương đương threshold trên positive

`argmax` trên 3 logit là quy tắc quyết định cứng, **không có tham số nào để điều chỉnh**
trade-off precision/recall. Không thể calibrate. Spec yêu cầu threshold riêng cho từng pathology —
hiện tại không có chỗ nào để cắm vào.

### 🟠 S1-G — `No Finding` và `Support Devices` nằm trong macro

Index 0 là `No Finding` — một meta-label, **phủ định của 13 cái còn lại**, không phải bệnh lý.
Index 13 là `Support Devices` — thiết bị, không phải bệnh lý. Gộp cả hai vào
`f1_positive_macro` làm lệch con số so với cách báo cáo chuẩn trong y văn (thường 13 hoặc 5 bệnh lý).

### 🟠 S1-H — Missing label bị âm thầm biến thành negative

`ReportDataset.py:314` `.fillna(0)`. Một study có **một số** cột NaN (không phải tất cả) sẽ:
- `_has_chexpert_label_raw = True` → **không bị mask**,
- các cột NaN → `0 = negative`.

CheXpert coi ô trống là "không đề cập", và quy ước phổ biến là gán negative — nhưng ở đây quy ước
đó bị **nướng cứng vào dữ liệu**, và mask chỉ ở **mức sample**, không ở **mức (sample, pathology)**.
Evaluator vì thế **không thể** phân biệt "âm tính thật" với "không đề cập". Không sửa được ở tầng
evaluator; cần mask 2 chiều. Ghi nhận là hạn chế đã biết.

### 🔴 S2-A — `except Exception` biến metric lỗi thành điểm 0

```python
except Exception:
    rouge_l = 0.0        # :1320-1321
except Exception:
    cider = 0.0          # :1325-1326
```

Một lỗi dependency, một input rỗng, hay một bug bất kỳ đều trở thành **điểm 0 hợp lệ trên bảng kết
quả**. Không phân biệt được "model kém" với "metric không chạy". METEOR còn có **ba** tầng
try/except lồng nhau (`:1308-1314`) cùng kết cục. Đây là lỗi nguy hiểm nhất của Stage 2 vì nó tạo
ra con số **trông như đã đo**.

### 🟠 S2-B — Tokenizer phá cấu trúc câu

`tokenize = re.findall(r"\w+", lower())` — bỏ toàn bộ dấu câu và lowercase.
- Điểm tốt: các từ y khoa quan trọng (`no`, `without`, `new`, `left`, `right`) **vẫn được giữ** vì
  chúng là `\w+`. Spec lo ngại chuyện này — kiểm tra cho thấy hiện **chưa** bị mất.
- Điểm xấu: mất ranh giới câu → ROUGE-L (dựa trên LCS) và CIDEr bị ảnh hưởng; và tokenization này
  **không khớp** với tokenizer nội bộ của pycocoevalcap, nên `gts`/`res` truyền vào `Rouge()` là
  chuỗi **gốc** chứ không phải chuỗi đã tokenize (`:1315-1316`) → **hai metric dùng hai
  tokenization khác nhau** trong cùng một hàm mà không ghi chú.

### 🟠 S2-C — BERTScore chỉ báo F1, không rescale

Chỉ `f1.mean()` được trả về (`:1353`); precision và recall bị vứt (`_p`, `_r`).
`rescale_with_baseline=False` → giá trị tuyệt đối nằm trong dải hẹp ~0.8–0.9 kể cả với text ngẫu
nhiên, rất dễ bị đọc nhầm là "tốt".

### 🟠 S2-D — Không có per-sample score → không thể error analysis

`compute_nlg` chỉ trả về **corpus-level**. Không có BLEU/ROUGE/BERTScore theo từng sample, nên
không thể tìm mẫu tệ nhất, không thể phân nhóm lỗi, không thể bootstrap.

### 🟠 S2-E — Làm tròn 4 chữ số ngay trong hàm tính

`round(..., 4)` tại `:1346-1353`. Mất precision trước khi có cơ hội bootstrap hay tổng hợp.

### 🔴 S2-F — Không có clinical metric nào

Không có CheXbert, RadGraph, RadCliQ, RadFact. Điều này đã được ghi nhận trung thực trong
`docs/STAGE2_PIPELINE_MODES.md` và `training/evaluation/clinical.py` (module này raise
`MissingOptionalDependency` thay vì trả số giả — **thiết kế đúng, giữ nguyên**).

### 🟠 CHUNG — Không có bootstrap CI, không baseline, không subgroup

Không có phần nào của repo tính confidence interval, all-negative baseline, hay phân tích theo
view/subgroup.

---

## 5. Rủi ro metric cao giả — tổng hợp

| Rủi ro | Cơ chế | Mức độ |
|---|---|---|
| `accuracy` cao trong khi không phát hiện được positive nào | S1-E, không có baseline đối chứng | 🔴 cao |
| `BERTScore` ~0.85 bị đọc là "tốt" | S2-C, không rescale baseline | 🔴 cao |
| ROUGE-L / CIDEr = 0 do exception, bị ghi vào bảng như kết quả thật | S2-A | 🔴 cao |
| `f1_positive_macro` **thấp** giả, và không so sánh được giữa các split | S1-A, S1-B | 🟠 vừa |
| Macro bị lệch do gộp `No Finding` + `Support Devices` | S1-G | 🟠 vừa |
| "Không đề cập" bị tính là "âm tính đúng" | S1-H | 🟠 vừa (không sửa được ở tầng eval) |

**Lưu ý về hướng lệch:** S1-A và S1-B khiến `f1_positive_macro` **thấp hơn** thực tế, không phải
cao hơn. Nó vẫn là bug — vì đây là **selection metric** chọn checkpoint, và một metric nhiễu theo
phân bố split sẽ chọn sai checkpoint.

---

## 6. Kế hoạch nâng cấp

Nguyên tắc: **không sửa `image_text_pretrain.py` theo hướng đổi giá trị metric đang có.**
`f1_positive_macro` hiện là selection metric của các run đang chạy; đổi công thức tại chỗ sẽ làm
checkpoint cũ không so sánh được. Thay vào đó:

- Giữ nguyên các key hiện có (backward-compatible).
- **Thêm** key mới với hậu tố rõ ràng (ví dụ `f1_positive_macro_valid_only`).
- Toàn bộ metric mới nằm trong package `evaluation/` độc lập, chạy **offline từ file prediction**.

| # | Hạng mục | Ghi chú |
|---|---|---|
| 1 | `evaluation/schemas.py` + serialization prediction (npz/jsonl) | Nền tảng cho mọi thứ còn lại |
| 2 | Hook lưu logits/probs/labels/ID vào Stage-1 eval | Sửa tối thiểu, có cờ bật/tắt |
| 3 | `evaluation/classification_metrics.py` | Đầy đủ macro/micro/weighted + positive-focused + clinical-style + AUROC/AUPRC per-pathology, có `valid_mask` |
| 4 | `evaluation/uncertain_policy.py` | 4 policy; mặc định giữ hành vi hiện tại |
| 5 | `evaluation/threshold_calibration.py` + `scripts/calibrate_thresholds.py` | Chỉ trên validation; 5 objective |
| 6 | `evaluation/bootstrap.py` | Resample **theo study**, có seed |
| 7 | `evaluation/baselines.py` | all-negative, majority, prevalence-random, 0.5 |
| 8 | `evaluation/generation_metrics.py` | Thay `compute_nlg`; bỏ `except → 0.0`; per-sample; BERTScore P/R/F1 |
| 9 | `evaluation/clinical_metrics.py` | Interface; tái dùng `training/evaluation/clinical.py` đã có |
| 10 | `evaluation/error_analysis.py` | Gồm temporal-hallucination heuristic, đặt tên `possible_*` |
| 11 | `evaluation/subgroup_analysis.py` | Theo view/PA/AP/multi-view/độ dài report |
| 12 | `evaluation/visualization.py` | ROC/PR/confusion/bar; tắt được bằng CLI |
| 13 | `evaluation/report_writer.py` | `evaluation_report.md` + json/csv |
| 14 | 4 script CLI | `evaluate_stage1/2`, `calibrate_thresholds`, `generate_evaluation_report` |
| 15 | Unit test + integration test | 26 case theo spec; không phá 328 test cũ |

### Quyết định thiết kế cần chốt

1. **`No Finding` / `Support Devices`**: đề xuất mặc định **loại khỏi macro** nhưng vẫn báo cáo
   per-pathology, kèm cờ `--macro-include-meta-labels` để giữ hành vi cũ.
2. **Probability cho AUROC ở bài toán 3 lớp**: dùng `softmax(logits)[..., 1]` (xác suất class
   positive). Đây là lựa chọn chuẩn khi quy về nhị phân positive-vs-rest.
3. **`f1_positive_macro` cũ**: giữ nguyên bit-for-bit làm selection metric; metric mới song song.
