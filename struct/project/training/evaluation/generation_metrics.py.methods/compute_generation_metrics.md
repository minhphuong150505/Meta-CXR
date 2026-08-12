> Source: `training/evaluation/generation_metrics.py:308-437`
> Status: ✅ ACTIVE

# `compute_generation_metrics(predictions, references, ...)`

## Purpose

Tính lexical generation metrics được yêu cầu và ghi provenance implementation/
version; metric optional thiếu dependency được báo unavailable thay vì score 0.

## Behavior

- Số prediction/reference lệch hoặc list rỗng → `ValueError`.
- Metric name lạ → `ValueError` liệt kê supported set.
- BLEU/ROUGE có implementation nội bộ; METEOR/CIDEr/BERTScore import optional.
- `strict=False`: ghi `suite.unavailable`; `strict=True`: raise
  `MissingMetricDependency`.
- Prediction rỗng sau tokenize được đếm trong provenance và nhận lexical score 0.

## Returns / caveat

`MetricSuite(corpus, per_sample, unavailable, provenance)`. Đây là lexical
similarity, không được gọi là clinical accuracy.

## Tests

`tests/test_generation_metrics.py`, `tests/test_clinical_metrics.py` cho semantics
unavailable liên quan.

← [`generation_metrics.py`](../generation_metrics.py.doc.md) · [HOME](../../../../HOME.md)
