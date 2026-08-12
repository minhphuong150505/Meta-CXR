> Source: `training/evaluation/generation_metrics.py` (437 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `generation_metrics.py`

## Purpose
Chỉ số sinh ngôn ngữ. **BLEU và ROUGE-L tự implement** (chỉ cần stdlib+numpy);
METEOR/CIDEr/BERTScore ủy quyền cho package tùy chọn.

## Why it exists
Docstring `:3` ghi file này sinh ra để sửa **hai lỗi** trong `fig9.compute_nlg`.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `compute_generation_metrics(...)` | 308 | [📄](generation_metrics.py.methods/compute_generation_metrics.md) ★ Điểm vào |
| `corpus_bleu(...)` | 103 | ★ Tự implement |
| `rouge_l(prediction, reference, beta=1.2)` | 185 | ★ Tự implement |
| `rouge_n(...)` | 207 | |
| `_lcs_length(a, b)` | 169 | Longest common subsequence |
| `normalize(text)` / `tokenize(text)` | 83, 88 | ★ Dùng chung với `error_analysis` |
| `_meteor` / `_cider` / `_bertscore` | 226, 239, 252 | Ủy quyền package ngoài |
| `MetricSuite` | 283 | Gói chỉ số |
| `_package_version(name)` | 299 | ★ Ghi version vào kết quả |
| `LEXICAL_METRICS` | — | Danh sách chỉ số từ vựng |
| `MissingMetricDependency` | 69 | |

## Calls / Called by
Gọi: `collections.Counter`, `numpy`; lazy: `nltk`/`pycocoevalcap`/`bert_score`.
Được gọi: `scripts/evaluate_stage2.py:41`; `error_analysis.py:36`; `config.py:19`;
`tests/test_generation_metrics.py:26`.

## Side effects
Không.

## Error / edge cases
Thiếu package → `MissingMetricDependency` **nêu tên package**, không trả 0.

## Related tests
`tests/test_generation_metrics.py` (314 dòng)

## Developer notes
⚠ **BLEU/ROUGE không phải độ chính xác lâm sàng.** Chúng đo trùng lặp từ ngữ. Một
báo cáo sai hoàn toàn về mặt y khoa vẫn có thể đạt BLEU cao nếu dùng đúng từ vựng.
`_package_version` được ghi vào kết quả để sau này so sánh được giữa các lần chạy.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md)

← [HOME](../../../HOME.md)
