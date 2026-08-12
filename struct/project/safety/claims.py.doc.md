> Source: `safety/claims.py` (286 dòng)
> Status: ✅ ACTIVE — caller thật duy nhất của safety/
> Last verified against source: 2026-08-12

# `safety/claims.py`

## Purpose
Tách báo cáo thành **claim** có cấu trúc: bệnh lý nào, cực tính gì (khẳng định /
phủ định / mơ hồ), bằng chứng ở đâu.

## Status
```text
✅ ACTIVE — file DUY NHẤT trong safety/ có caller production
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `Claim` | 122 | ★ Một khẳng định |
| `Evidence` | 107 | Bằng chứng kèm theo |
| `LexiconClaimParser` | 223 | ★ Parser dựa từ điển |
| `ClaimParser` (Protocol) | 170 | Interface — cắm parser học máy vào sau |
| `split_sentences(text)` | 177 | |
| `detect_polarity(sentence, mention_start)` | 192 | ★ Khẳng định hay phủ định |
| `unparsed_sentences(report, claims)` | 275 | ★ Câu **không** parse được |
| `_first_term(haystack, terms)` | 182 | |

## ★ `unparsed_sentences` — vì sao được phơi ra
Một pipeline chỉ parse được 2 trên 12 câu **chưa kiểm tra được báo cáo**, dù các
con số của nó trông sạch đến đâu. Hàm này khiến tỉ lệ đó không giấu được.

## `detect_polarity` — chỗ khó
"No evidence of pneumonia" và "Evidence of pneumonia" chỉ khác một từ nhưng ngược
nghĩa hoàn toàn. Parser phải bắt được phủ định, hedging, và phạm vi của chúng.

## Calls / Called by
Gọi: `re`, stdlib. **Torch-free.**
Được gọi: **`training/evaluation/error_analysis.py:30`** (production);
`safety/pipeline.py`, `verifiers.py`, `reconciler.py`; `tests/test_safety_pipeline.py:9`.

## Side effects
Không.

## Related tests
`tests/test_safety_pipeline.py`

## Developer notes
Đây là phần `safety/` đã được dùng thật. Ba file còn lại (`pipeline`, `verifiers`,
`reconciler`) chưa có caller — xem [D-001](../_meta/DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline).

## Source relationships

- **Parent:** [`_index.md`](_index.md)

← [HOME](../../HOME.md)
