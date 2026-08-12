> Source: `safety/pipeline.py` (180 dòng)
> Status: ❓ UNKNOWN — chỉ test
> Last verified against source: 2026-08-12

# `safety/pipeline.py`

## Purpose
Điều phối: draft report → claim → verify → báo cáo cuối **hoặc từ chối trả lời**.

## ⚠ Chưa có caller production
Chỉ `tests/test_safety_pipeline.py`. [D-001](../_meta/DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline).

## Status
```text
❓ UNKNOWN
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `SafetyPipeline` | 87 | ★ `.run(...)` |
| `SafetyReport` | 44 | Kết quả; ⚠ **không mang** subject_id/study_id/path/reference |
| `ABSTENTION_TEXT` | — | Văn bản khi từ chối |
| `hallucination_rate(claims)` | 169 | |
| `supported_rate(claims)` | 177 | |

## Thiết kế: orchestration không chứa logic verify
Có chủ đích — để cắm một model phrase-grounding **thật** vào qua cùng protocol
(`verifiers.PhraseGroundingVerifier`) mà không phải viết lại điều phối.

## ★ Từ chối trả lời là một kết quả hợp lệ
Với báo cáo y khoa, "tôi không chắc" tốt hơn một câu sai nghe thuyết phục.
`ABSTENTION_TEXT` là đường thoát đó.

## Calls / Called by
Gọi: `safety.claims` (`:20`), `safety.reconciler` (`:29`), `safety.verifiers` (`:30`).
Được gọi: **chỉ** `tests/test_safety_pipeline.py:17`.

## Side effects
Không. Record đầu ra **an toàn để lưu** (không mang định danh).

## Related tests
`tests/test_safety_pipeline.py` (297 dòng)

## Developer notes
Nếu nối vào đường sinh báo cáo, nhớ giữ `parse_coverage` trong output — nó là thứ
cho biết pipeline đã thật sự kiểm được bao nhiêu.

## Source relationships

- **Parent:** [`_index.md`](_index.md)

← [HOME](../../HOME.md)
