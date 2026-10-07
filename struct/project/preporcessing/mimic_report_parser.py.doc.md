> Source: `preporcessing/mimic_report_parser.py` (159 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-10-07

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
| `is_exam_header(text)` / `_strip_exam_header(line)` | — | Loại dòng tiêu đề exam (2026-10-07) |
| `clean_report_text(text)` | 144 | Dọn whitespace, ký tự lạ |
| `count_lexical_tokens(text)` | 155 | Đếm token cho giới hạn độ dài |

`_narrative_after_comparison` là phần tinh tế: nhiều report viết mô tả ngay sau
mục COMPARISON mà không có tag FINDINGS.

## ★ Dòng tiêu đề exam bị loại (2026-10-07)
`is_exam_header(text)` / `_strip_exam_header(line)`: một dòng không có chữ thường,
có chữ CHEST, và mọi từ còn lại thuộc từ vựng projection/view (AP, PA, LATERAL,
PORTABLE, VIEW(S), RADIOGRAPH(S), FILM, ...; giờ `10:11`, `A.M.`, `___`, số và dấu
bị bỏ qua) bị loại khỏi narrative; tiền tố tiêu đề kết thúc bằng `:`, `.` hoặc `,` rồi dấu cách (tiền tố DÀI NHẤT thắng, để giữ `A.M.` trong tiêu đề) bị cắt và giữ phần nội dung cùng dòng, vd. `CHEST, TWO VIEWS. The lungs...`. Từ vựng mở rộng lần 2 cùng ngày (ONE, FROM, AM, PM, LAT, VW(S), STUDY, REPORT) sau khi đo phần còn sót trên manifest v3.
`reason for exam` thêm vào alias → `indication`. Lý do: trước đó 9.8% target train
của `full_allviews_v2` bắt đầu bằng tiêu đề ("AP CHEST, 10:11 A.M., ___") và 3.4%
CHỈ là tiêu đề, khiến Stage 2 paper-mode sinh tiêu đề rồi dừng ở 57% study test.
Đo trên toàn bộ 227,835 report (parser cũ vs mới): 20,711 target đổi, 6,177 thành
rỗng (→ `IMPRESSION_ONLY`), 9 còn bắt đầu bằng tiêu đề; mọi target mất >15 từ đều
là phần REASON FOR EXAM (lý do chụp), 171/173 khối chữ ngay dưới dòng đó là lý do
bị xuống dòng. Từ vựng hẹp nên "CHEST TUBE IN PLACE." không bao giờ bị loại.

## Calls / Called by
Gọi: `re`, stdlib.
Được gọi: `preprocess_mimic_cxr.build_study_text`.

## Side effects
Không (hàm thuần trên chuỗi).

## Related tests
`tests/test_mimic_data_pipeline.py` (`FindingsAndImpressionTargetTest`, gồm 4 test tiêu đề exam thêm 2026-10-07).

## Developer notes
⚠ **Không đưa ví dụ report thật vào bất kỳ tài liệu hay test fixture nào** — đó là
dữ liệu bệnh nhân.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
