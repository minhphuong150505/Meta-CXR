> Source: `training/evaluation/counterfactual.py` (268 dòng)
> Status: ❓ UNKNOWN — chỉ test import
> Last verified against source: 2026-08-12

# `counterfactual.py`

## Purpose
Kiểm tra model **thật sự nhìn ảnh**: làm hỏng ảnh đầu vào rồi xem báo cáo có đổi
không. Nếu không đổi, model đang đọc từ language prior chứ không từ pixel.

## ⚠ Chưa được nối
Chỉ `tests/test_counterfactual.py`. Không script nào bắt đầu chuỗi này.
[D-001](../../_meta/DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline).

## Status
```text
❓ UNKNOWN
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `CounterfactualEvaluator` | 118 | ★ |
| `ReportGenerationBackend` (Protocol) | 44 | Cắm model thật vào |
| `ClinicalChangeBackend` (Protocol) | 59 | |
| `CounterfactualConfig` | 83 | |
| `AuditSample` | 74 | |
| `lexical_change(original, perturbed)` | 103 | Mức thay đổi từ vựng |
| `privacy_violations(payload, path)` | 241 | ★ Quét dữ liệu nhạy cảm |
| `assert_shareable(payload)` | 261 | ★ **Chặn xuất bản ghi có PHI** |
| `tokenize(text)` | 99 | |

⚠ `:36` mirror `SENSITIVE_EVAL_KEYS` từ `training/stage2_utils.py`. **Hai nơi phải
khớp nhau** — comment ghi rõ điều đó.

## Calls / Called by
Gọi: `evaluation.perturbations` (`:32`), `torch`.
Được gọi: **chỉ** `tests/test_counterfactual.py`.

## Side effects
Chạy inference (nếu backend được cắm).

## Related tests
`tests/test_counterfactual.py` (320 dòng)

## Developer notes
Đây là **kiểm chứng mạnh nhất** cho câu hỏi "model có nhìn ảnh không" — mạnh hơn
`capabilities.py` (chỉ kiểm cấu trúc model). Nếu nối được, nên nối.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md)

← [HOME](../../../HOME.md)
