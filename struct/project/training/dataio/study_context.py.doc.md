# `training/dataio/study_context.py`

> Thư mục: [`training/dataio/`](_index.md) · Thêm 2026-10-11 · ✅ ACTIVE (chỉ dùng với `--prompt-style paper_context`)

## Purpose
Bối cảnh lâm sàng của một study cho prompt Stage 2: mục INDICATION/HISTORY của
chính báo cáo và FINDINGS của **lần chụp trước gần nhất** của cùng bệnh nhân. Bằng
chứng: chép FINDINGS lần trước (không học gì) đã ngang bài báo ở BLEU-1/BLEU-4 và hơn
mô hình v3 0,12 CE lexicon; 90% FINDINGS test có ngôn ngữ so sánh thời gian
(`docs/TONG_HOP_2026-10-11.md`).

## Quy tắc chống rò rỉ (`tests/test_study_context.py`)
- lần trước = study **sớm hơn hẳn** (StudyDate + StudyTime, `merge_asof` với
  `allow_exact_matches=False`) của **cùng** `subject_id`, chỉ trong các study có target
  FINDINGS hợp lệ; không bao giờ là chính nó hay study sau;
- indication lấy từ mục INDICATION/HISTORY (`mimic_report_parser.clinical_context`),
  không bao giờ từ FINDINGS/IMPRESSION.

## Main items
| Tên | Vai trò |
|---|---|
| `study_times(metadata_csv)` | study_id → thời điểm chụp sớm nhất |
| `prior_findings_table(split_csv, metadata_csv)` | mỗi study: FINDINGS lần trước + số ngày |
| `read_indication(reports_root, subject, study)` | INDICATION/HISTORY của báo cáo gốc |
| `attach_study_context(records, ...)` | ★ gắn `record["context"]`; lỗi nếu > 1% record không khớp manifest |
| `format_context_lines(context)` | hai dòng prompt; giá trị thiếu được ghi rõ ("not provided", "no earlier study available") |

Cắt: indication ≤ 60 từ, FINDINGS lần trước ≤ 150 từ. Dữ liệu trả về là văn bản báo
cáo: chỉ giữ trong bộ nhớ, không log, không commit. Chỉ cần pandas + parser stdlib.

## Callers
`training/run_medgemma_qlora.attach_prompt_inputs`, `scripts/generate_stage2_reports.stage1_records`
(cả hai chỉ khi `--prompt-style paper_context`), `fig9.build_instruction` (qua `format_context_lines`).
