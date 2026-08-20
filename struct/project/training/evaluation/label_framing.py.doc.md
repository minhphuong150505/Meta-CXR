> Source: `training/evaluation/label_framing.py` (178 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-20

# `label_framing.py`

## Purpose

Quyết định **câu hỏi mà metric đang trả lời** — và do đó quyết định F1 có nghĩa
hay không.

## Why it exists

Ma trận nhãn CheXpert có hai cách đọc hợp lệ, và chúng **không thay thế được cho
nhau**. Để mặc định ngầm đã làm `positive_macro_f1` mất khả năng diễn giải.

Đo trên test split của run `run_20260819_xmpoff` (3,269 study, 2026-08-20):

| | `masked_polarity` | `study_presence` |
|---|---|---|
| Prevalence mỗi nhãn | 0.13 – 1.00 (12/14 nhãn > 0.55) | 0.019 – 0.344 |
| `all_positive` đạt macro F1 | **0.8397** | **0.2280** |
| `all_negative` đạt macro F1 | 0.0000 | 0.0000 |
| `threshold_half` đạt macro F1 | 0.8713 | 0.2797 |
| Model | 0.8717 | 0.3074 |

Ở `masked_polarity`, một hằng số hơn model 0.03 — metric đó không đo kỹ năng.
53–98% ô mỗi nhãn là ô trống và bị mask, nên "âm tính" còn lại chỉ là nhóm hiếm
mà bác sĩ **viết rõ là không có**, khiến dương tính thành đa số ở 12/14 nhãn.

Đây là cùng một vấn đề mà [`uncertain_policy.py`](uncertain_policy.py.doc.md)
giải quyết cho lớp Uncertain: làm lựa chọn **tường minh, có tên, ghi vào mọi file
kết quả**, thay vì chôn trong một dòng code.

## Status

```text
✅ ACTIVE — mặc định `masked_polarity` (giữ nguyên hành vi lịch sử)
             `study_presence` là framing duy nhất nên trích dẫn F1
```

## Main items

| Tên | Vai trò |
|---|---|
| `FRAMINGS`, `DEFAULT_FRAMING` | `masked_polarity` (mặc định) / `study_presence` |
| `SCORES`, `DEFAULT_SCORE` | `conditional_positive` (mặc định) / `marginal_presence` |
| `frame_labels(labels, framing)` | ★ Viết lại nhãn class-index |
| `presence_scores(preds, score)` | ★ `q_pos` hoặc `mention × q_pos` |
| `apply_framing(preds, framing, score)` | ★ Trả về `ClassificationPredictions` mới |
| `validate_framing` / `validate_score` | Raise nếu tên lạ |
| `UnknownFramingError`, `ScoreUnavailableError` | |

## Design decisions

**Viết lại thành một `ClassificationPredictions` mới, không phải nhánh `if` trong
từng metric.** Nhờ đó calibration, AUROC, bootstrap, subgroup đều tiếp tục đọc
`probabilities[..., POSITIVE]` và **không cần sửa gì**.

**Dưới `study_presence`, phân phối trả về là hai điểm `[1-p, p, 0]`** — "uncertain"
không nằm trong tập câu trả lời mà framing này thừa nhận. Tổng vẫn bằng 1, nếu không
AUROC và vòng tìm ngưỡng sẽ đọc một cột không mang ý nghĩa nó tự nhận.

⚠ **Framing này đã nuốt luôn quyết định của `uncertain_policy`**: ô uncertain bị gộp
vào "không có". Phải nói rõ điều đó khi báo cáo, đừng để người đọc tưởng hai flag
còn độc lập.

⚠ **`marginal_presence` cần `mention_probabilities`** trong file `.npz`. Thiếu thì
raise `ScoreUnavailableError` chứ **không** âm thầm rơi về `conditional_positive` —
rơi ngầm sẽ cho ra một con số trông hợp lý mà sai nguồn gốc.

Chấm `study_presence` bằng `conditional_positive` là hợp lệ nhưng **thiệt cho model**:
`q` là xác suất có điều kiện trên việc finding được nhắc tới, nên mention gate chính
là thừa số bị bỏ. Con số thu được là **cận dưới**, không phải ước lượng.

## Calls / Called by

Gọi: `numpy`, `evaluation.schemas` (`NEGATIVE`, `POSITIVE`, `ClassificationPredictions`).
Được gọi: `scripts/calibrate_thresholds.py`, `scripts/evaluate_stage1.py`.
Test: `tests/test_label_framing.py` (9 test).

## Parent

[`training/evaluation/`](_index.md)
