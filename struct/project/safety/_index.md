> Source: `safety/` (5 file, ~1.163 LOC)
> Status: hỗn hợp — 1 file ✅ ACTIVE, 3 file ❓ UNKNOWN
> Last verified against source: 2026-08-12

# `safety/`

## Purpose

Kiểm chứng báo cáo sinh ra: tách câu thành claim, verify từng claim, rồi hoặc trả
báo cáo cuối hoặc **từ chối trả lời** (abstain).

Toàn bộ **stdlib-only** — chạy được ở bất cứ đâu test chạy được.

## ⚠ Trạng thái phân đôi

| File | Status | Caller |
|---|---|---|
| `claims.py` (286) | ✅ **ACTIVE** | `training/evaluation/error_analysis.py:30` |
| `pipeline.py` | ❓ UNKNOWN | chỉ `tests/test_safety_pipeline.py` |
| `verifiers.py` (252) | ❓ UNKNOWN | chỉ `pipeline.py` + test |
| `reconciler.py` (328) | ❓ UNKNOWN | chỉ `pipeline.py` + test |

Quyết định: [D-001](../_meta/DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline).
Không gắn LEGACY vì đây là **interface chờ implementation**, không phải code bị bỏ.

## Ba thiết kế đáng chú ý

### 1. `pipeline.py` không tự chứa logic verify
Có chủ đích — để cắm một model phrase-grounding **thật** vào qua cùng protocol,
mà không phải viết lại orchestration.

### 2. `parse_coverage` được phơi ra
Một pipeline chỉ parse được 2 trên 12 câu thì **chưa kiểm tra được báo cáo**, dù
các con số của nó trông sạch đến đâu. Giấu tỉ lệ này đi là tự lừa mình.

### 3. Record đầu ra an toàn để lưu
**Không mang** `subject_id`, `study_id`, `dicom_id`, đường dẫn, hay reference text.

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| File | Doc | Nội dung |
|---|---|---|
| `claims.py` | [📄](claims.py.doc.md) | Tách/biểu diễn claim — ✅ **dùng thật** |
| `pipeline.py` | [📄](pipeline.py.doc.md) | `SafetyPipeline`, `ABSTENTION_TEXT` |
| `verifiers.py` | [📄](verifiers.py.doc.md) | Protocol verify |
| `reconciler.py` | [📄](reconciler.py.doc.md) | `RuleBasedClaimReconciler`, `ReconciliationOutcome` |
| `__init__.py` | — | |

## Execution flow (khi được dùng)

```text
draft report
   ↓  claims.*
parsed claims  (+ parse_coverage)
   ↓  verifiers.*
verification outcomes
   ↓  reconciler.RuleBasedClaimReconciler
ReconciliationOutcome
   ↓  pipeline.SafetyPipeline
final report  HOẶC  ABSTENTION_TEXT
```

## Entry points

Không có. Thư viện.

## Dependencies

Chỉ stdlib.

## Used by

`training/evaluation/error_analysis.py:30` (**chỉ `claims`**) ·
`tests/test_safety_pipeline.py` (297 dòng)

## Status

```text
✅ ACTIVE — claims.py
❓ UNKNOWN — pipeline.py, verifiers.py, reconciler.py
```

## Notes

- `RuleBasedClaimReconciler(require_grounding=True)` là chế độ nghiêm ngặt —
  claim không có bằng chứng grounding thì không được giữ.
- Nếu bạn nối `SafetyPipeline` vào pipeline sinh báo cáo, hãy cập nhật
  [D-001](../_meta/DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline)
  và chuyển sang [ACTIVE_COMPONENTS.md](../_meta/ACTIVE_COMPONENTS.md).

## Related documentation

[ARCHITECTURE.md §5](../_meta/ARCHITECTURE.md#5-hai-khối-stdlib-only) ·
[`training/evaluation/_index.md`](../training/evaluation/_index.md) ·
[LEGACY_AND_OPTIONAL.md §U2](../_meta/LEGACY_AND_OPTIONAL.md#u2--safety--phần-orchestration)

← [Về HOME](../../HOME.md)
