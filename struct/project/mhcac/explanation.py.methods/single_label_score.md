> Source: `mhcac/explanation.py`
> Status: ✅ ACTIVE

# `single_label_score(logits, label_index, sample_selector)`

## Located in

[`explanation.py`](../explanation.py.doc.md)

## Purpose

Score Logit-Difference-Squared cho **một** bệnh, cộng trên các study được chọn.
Đây là đầu vào của Grad-CAM riêng theo bệnh (strong term).

## Vì sao tồn tại

[`logit_difference_squared`](logit_difference_squared.md) cộng **mọi** bệnh dương
vào một số duy nhất, nên gradient của nó cho ra một CAM trả lời "vùng nào giải
thích bất kỳ bệnh nào của study này". Với mask phổi giải phẫu thì đó là câu hỏi
đúng. Với một box chuyên gia vẽ quanh **một** phát hiện có tên thì đó là câu hỏi
sai — box đó chỉ có thể phán quyết được câu "mô hình có nhìn vào bệnh NÀY không".

## Execution flow

```text
kiểm tra logits [B,A,C], C >= 2, 0 <= label_index < A
   ↓
difference = logits[:, label_index, 1] - logits[:, label_index, 0]   (FP32)
   ↓
trả về Σ_study (difference² × selector)     ← scalar
```

## ★ Một backward cho tất cả study cùng bệnh

`sample_selector` gộp mọi study trong batch có box cho bệnh đó, nên
`ExplanationLoss._strong_term` chỉ cần **một** `autograd.grad` cho mỗi *bệnh
phân biệt* có mặt — thường 1–2 — chứ không phải một lần cho mỗi study, và cũng
không phải 14 lần.

## Parameters

| Tên | Nghĩa |
|---|---|
| `logits` | `[B, A, C]`, C >= 2; lớp 1 là Positive, lớp 0 là Negative |
| `label_index` | Chỉ số cột CheXpert; phải trùng thứ tự `chexpert_cols` |
| `sample_selector` | `[B]` bool — study nào có box cho bệnh này |

## Error / edge cases

| Tình huống | Hành vi |
|---|---|
| `logits` không phải 3 chiều hoặc C < 2 | `ValueError` |
| `label_index` ngoài `[0, A)` | `ValueError` |
| `sample_selector` sai độ dài | `ValueError` |
| Không study nào được chọn | Trả về 0 (scalar), vẫn nối với graph |

## Called by

`ExplanationLoss._strong_term`

## Modification risk

Đổi thứ tự nhãn ở bất kỳ đâu (`chexpert_cols`, `CHEXPERT_LABELS` trong
`build_explanation_masks.py`) mà không đổi đồng bộ sẽ khiến hàm này giám sát
**sai bệnh**, và không có gì báo lỗi.
