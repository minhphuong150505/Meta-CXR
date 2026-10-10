# PLAN 2026-10-11 — Pilot trước khi train lại toàn bộ (CHỜ USER DUYỆT)

Trạng thái: **đề xuất, chưa sửa code, chưa train.** Căn cứ:
`docs/RESEARCH-2026-10-11-stage1-stage2-diagnosis.md`, `docs/CHI_MUC_TINH_MOI.md`.

## Mục tiêu

Chọn cấu hình Stage 2 (và hệ quả cho Stage 1) bằng **một vòng pilot nhỏ có quy
tắc chọn chốt trước**, rồi train toàn bộ dữ liệu **đúng một lần**.

## Vì sao pilot này, không phải pilot khác

- Đòn bẩy NLG lớn nhất trong tài liệu là bối cảnh đầu vào: indication (RaDialog
  BLEU-4 9,5 → 14,8) và lần chụp trước (MAIRA-2 ROUGE-L −28,9% khi bỏ). Dữ liệu
  dự án có sẵn: 91,7% study test có INDICATION/HISTORY; 89,6% test và 64,1% train
  có lần chụp trước với FINDINGS (khoảng cách trung vị 7–9 ngày).
- Kênh thị giác hiện tại là một vector (soft token sụp, TM-11). Kênh ảnh gốc của
  MedGemma đã được train trên MIMIC-CXR; tài liệu cho thấy MedGemma 4B đạt RadGraph
  F1 29,5 không cần fine-tune.
- Sửa Q-Former đòi train lại 1a cần cache đặc trưng ~148 GB trong khi `/home` còn
  59 GB; và RaDialog-align (Q-Former 32 query hoạt động bình thường) chỉ đạt BLEU-4
  9,5 / ROUGE-L 27,1 — ngang mức hiện tại. Vì vậy sửa Q-Former **không** nằm trong
  pilot này; chỉ làm nếu user muốn giữ đúng kiến trúc bài báo.

## Các arm (cùng seed, cùng 10.000 study train, 1 epoch, cùng LoRA r8/α16)

| arm | kênh ảnh | cue MHCAC | bối cảnh | ước tính |
|---|---|---|---|---|
| P0 | 32 soft token (như v3) | Eq. 22 | không | ~1 h train + ~35 phút chấm |
| P1 | 32 soft token | Eq. 22 | indication + FINDINGS lần trước | ~1,1 h + ~35 phút |
| P2 | ảnh gốc MedGemma (256 token) | ngưỡng hai mức | indication + FINDINGS lần trước | ~4,5 h + ~1 h |

Chấm trên **500 study val cố định** (cùng `sample_key`, cùng thứ tự), beam 4 +
length penalty 2 và greedy, giao thức bài báo (BLEU/METEOR/ROUGE-L/CIDEr/
BERTScore) + CE lexicon; bootstrap ghép cặp 2.000 lần, seed 16.

## Quy tắc chọn (chốt trước khi có kết quả)

1. **Bối cảnh:** chấp nhận nếu P1 − P0 có ROUGE-L **và** BERTScore với cận dưới
   khoảng tin cậy > 0.
2. **Kênh ảnh gốc:** vì đắt gấp ~4 lần, chấp nhận nếu P2 − P1 đạt BERTScore ≥ +0,01
   hoặc CE lexicon ≥ +0,02, cận dưới > 0, và không chỉ số nào giảm có ý nghĩa.
3. Không arm nào thỏa: giữ v3, chỉ thêm beam vào bộ sinh.
4. Sau đó train toàn bộ **một lần** với cấu hình thắng (paper-mode ~19 h/epoch;
   ảnh gốc ~80 h/epoch toàn bộ, hoặc tập con ~80k study ~37 h — user chọn).

## Code cần viết (sau khi duyệt)

1. Tiền xử lý (CPU): thêm `indication_clean`, `prior_findings_clean`,
   `prior_interval_days` vào manifest → `full_allviews_v4` (không đổi target).
2. Prompt: kiểu `paper_context` = prompt bài báo + dòng Indication + dòng FINDINGS
   lần trước (có số ngày); thiếu thì ghi "None".
3. Chế độ ảnh gốc + cue không kèm soft token (hiện chỉ có kèm soft token).
4. Beam search trong `generate_stage2_reports.py` (port từ probe).
5. Test CPU cho cả bốn; smoke 200 study trên GPU trước pilot.

## Điều kiện dừng

OOM, NaN, smoke không sinh được báo cáo, hoặc cohort 500 study val không khớp
`sample_key` giữa các arm.
