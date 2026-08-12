> Source: `training/evaluation/clinical.py` (185 dòng)
> Status: ✅ ACTIVE — luôn báo unavailable
> Last verified against source: 2026-08-12

# `clinical.py`

## Purpose
Adapter cho chỉ số lâm sàng — và **cố ý từ chối bịa số**.

## ⚠ Quy tắc không thương lượng
CheXbert, RadGraph, CheXpert labeler **không phải extras cài được**. Chúng là
research code sau license riêng, không phải pin tái lập được.

Module raise một trong hai:
- `MissingOptionalDependency` — **nêu tên package** thiếu
- `NotImplementedError` — package có nhưng adapter **chưa validate** với điểm tham
  chiếu công bố

> **Chỉ số lâm sàng thiếu được báo là "unavailable", KHÔNG BAO GIỜ là điểm 0.**
> Điểm 0 nghĩa là "model sai hoàn toàn". "Unavailable" nghĩa là "chưa đo được".
> Nhầm hai thứ này là bóp méo kết quả nghiên cứu.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `build_metric(name)` | 134 | ★ Factory |
| `resolve_metrics(...)` | 142 | |
| `validate_selection_metric(...)` | 173 | |
| `RadGraphMetric` | 100 | |
| `CheXbertMetric` | 109 | |
| `CheXpertLabelMetric` | 118 | |
| `_DependencyBackedMetric` | 61 | Lớp cơ sở |
| `ClinicalMetric` (Protocol) | 50 | Interface cho adapter thật |
| `MissingOptionalDependency` | 34 | |

⚠ RadCliQ và RadFact được nhắc trong tài liệu nhưng **không có class** ở đây.

## Calls / Called by
Gọi: `importlib` (dò package).
Được gọi: `scripts/evaluate_stage2.py:159` (**import trễ**); `config.py:184`;
`tests/test_clinical_metrics.py`.

## Side effects
Không (chỉ dò import).

## Related tests
`tests/test_clinical_metrics.py` — canh **không trả 0**

## Developer notes
Muốn thêm adapter thật: implement `ClinicalMetric` Protocol, và **validate với
điểm tham chiếu đã công bố** trước khi bỏ `NotImplementedError`. Nếu không, số
sinh ra không so sánh được với bất kỳ bài báo nào.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md)

← [HOME](../../../HOME.md)
