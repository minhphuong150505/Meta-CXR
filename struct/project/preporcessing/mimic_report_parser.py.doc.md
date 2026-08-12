> Source: `preporcessing/mimic_report_parser.py` (159 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `preporcessing/mimic_report_parser.py`

## Purpose
Trích FINDINGS và IMPRESSION từ report `.txt` thô.

## ★ Vì sao không thể chỉ regex một tag
Một tỉ lệ đáng kể report **không có tag FINDINGS**. Hành vi cũ là rơi về nguyên văn
cả báo cáo — target khi đó lẫn cả INDICATION, TECHNIQUE, COMPARISON, IMPRESSION.
Model học từ target đó sẽ học sinh ra thứ không phải findings.

Parser hiện tại **khôi phục phần thân tường thuật** thay vì fallback mù.

## Main functions
| Hàm | Dòng | Vai trò |
|---|---|---|
| `get_target_text(report_text)` | 114 | ★ → `(findings, impression, ?)` |
| `extract_sections(report_text)` | 100 | ★ |
| `_report_sections(report_text)` | 50 | Chia theo tên section |
| `_narrative_after_comparison(comparison)` | 85 | ★ Cứu phần tường thuật nằm sau COMPARISON |
| `_normalise_section_name(name)` | 46 | Chuẩn hóa tên section |
| `clean_report_text(text)` | 144 | Dọn whitespace, ký tự lạ |
| `count_lexical_tokens(text)` | 155 | Đếm token cho giới hạn độ dài |

`_narrative_after_comparison` là phần tinh tế: nhiều report viết mô tả ngay sau
mục COMPARISON mà không có tag FINDINGS.

## Calls / Called by
Gọi: `re`, stdlib.
Được gọi: `preprocess_mimic_cxr.build_study_text`.

## Side effects
Không (hàm thuần trên chuỗi).

## Related tests
⚠ Không có test riêng trong `tests/` của checkout này.

## Developer notes
⚠ **Không đưa ví dụ report thật vào bất kỳ tài liệu hay test fixture nào** — đó là
dữ liệu bệnh nhân.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
