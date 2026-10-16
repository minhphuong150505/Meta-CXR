# Eval bundle — sinh 2026-09-13 từ máy train

Gom **mọi** kết quả đánh giá đang có trên host thành một chỗ, ở dạng **chỉ số
tổng hợp**.

## ⚠ Đọc phần này trước, nếu không bảng sẽ bị hiểu sai

### 1. Mỗi dòng Stage-2 thuộc về MỘT Stage-1 checkpoint cụ thể

Cột `stage1_source` trong `stage2_generation.csv` ghi rõ. Không được gộp chúng
lại: `img_proj` của một adapter được train trên soft token của đúng Stage-1 đó,
nên số của hai checkpoint khác nhau là hai thí nghiệm khác nhau.

Đây cũng là lý do không thể "chấm lại" Stage-2 cho một checkpoint mới mà không
train lại — đẩy soft token mới qua projector cũ là lệch phân phối, và nó **im
lặng** sinh báo cáo trôi chảy mô tả nhiễu, không lỗi ở đâu cả.
`run_20260912_mc_shallow` có số Stage-2 vì đã train lại thật (xem mục cuối).

### 2. Một dòng Stage-1 KHÔNG so sánh được với các dòng còn lại

`eval_kho_noi/stage1_pnu` dùng **ngưỡng mặc định 0,5** và framing
`masked_polarity`. F1 0,876 của nó **không phải** kết quả tốt hơn — dưới framing
đó một bộ dự đoán hằng số cũng đạt 0,8397. Cột `comparable` đánh dấu `NO`.
Framing không được ghi trong các file cũ nên giá trị đó là **suy ra** từ nguồn
ngưỡng, không phải đọc trực tiếp.

### 3. Số Stage-2 sinh trước 2026-09-09 mang LỖI DỪNG SINH

78% token sinh ra nằm **sau** `<end_of_turn>`. Các dòng thuộc
`eval_matched_20260908/`, `eval_final/`, `cueprobe/`, `signal_probe_val_20260908/`
và `*_old_eval` là loại đó. Không trộn chúng với các dòng sau bản sửa
(`cue4_fixedstop_20260909/`, `cue_mid_gen_20260909/`, `ftok_pilot_20260910/`,
`*_fixed_eval`).

### 4. Không có metric lâm sàng nào ở đây

CheXbert / RadGraph / RadCliQ **không được cài** trong repo này và
`training/evaluation/clinical.py` **raise** chứ không trả về 0. Cột
`clinical_metrics` ghi `unavailable`. BLEU/ROUGE/METEOR/CIDEr/BERTScore là
metric **từ vựng** và không được trình bày như độ chính xác lâm sàng.

### 5. n khác nhau giữa các dòng

Từ 25 đến 3.269 study. So hai dòng khác cohort hoặc khác n **không phải** một
phép so sánh — dự án này đã mắc lỗi đó và ghi lại ít nhất hai lần. Bootstrap
ghép cặp trong `stage1_paired_bootstrap.csv` mới là phép so hợp lệ.

## Các file

| file | nội dung |
|---|---|
| `stage1_classification.csv` | 6 lần chấm Stage-1, kèm nguồn ngưỡng, framing và cờ `comparable` |
| `stage1_paired_bootstrap.csv` | các phép so ghép cặp, delta + CI95 + cờ CI có loại trừ 0 |
| `stage2_generation.csv` | 35 lần chấm Stage-2, kèm `stage1_source`, n, độ dài sinh/tham chiếu |
| `stage2_paired_bootstrap.csv` | 16 phép so Stage-2 ghép cặp × 4 metric = 64 hàng, kèm cột `at_zero_boundary` |

### 6. Cột `at_zero_boundary` — đọc nó trước cột `ci_excludes_zero`

Một cận CI cách 0 dưới 0,0005 **đổi dấu theo nhiễu Monte-Carlo** của bootstrap.
Đã quan sát thật: `D − B` ROUGE-L là [−0,0147, +0,0002] ở lần chạy pilot và
[−0,0147, −0,0002] trong file này, **delta giống hệt**. Hàng nào có
`at_zero_boundary = True` là **chưa thiết lập được theo chiều nào**, bất kể
`ci_excludes_zero` ghi gì. Có 4 hàng như vậy, cả bốn đều thuộc pilot
(`D − B` ×3, `D − C` ×1). Kết quả finding-token duy nhất xa biên là
`D − A` BERTScore [−0,0133, −0,0021].

## Bảo mật dữ liệu

Bundle này **chỉ chứa số tổng hợp**. Không đọc `per_sample_results.jsonl`
(chứa văn bản báo cáo sinh ra và tham chiếu) và không đọc `sample_keys` trong
`.npz` (nối ngược về study). Đã quét chuỗi dạng định danh
(`p1X/pXXXXXXXX`, `sXXXXXXXX`, dicom id) — không có kết quả nào.

Ở máy dev, thư mục `Eval/` nằm trong `.gitignore`. **Đừng chuyển bundle này ra
khỏi đó**, và đừng thêm `per_sample_results.jsonl` hay `.npz` vào nó.

## Bổ sung 2026-09-13 — Stage 2 đã được train lại trên checkpoint mới

Cảnh báo số 1 ở trên **đã được xử lý**: `run_20260912_mc_shallow` giờ có số
Stage-2 thật, từ hai arm train lại chứ không phải suy ra.

| arm | Stage-1 | cue | ROUGE-L | METEOR | CIDEr | BERT-F1 |
|---|---|---|---:|---:|---:|---:|
| A | `run_20260820_ft` | none | 0,2552 | 0,2354 | 0,2000 | 0,7847 |
| B | `run_20260820_ft` | marginal | 0,2561 | 0,2366 | 0,2101 | 0,7823 |
| A' | `mc_shallow` | none | 0,2527 | 0,2352 | 0,2428 | 0,7814 |
| B' | `mc_shallow` | marginal | 0,2555 | 0,2335 | 0,2333 | 0,7817 |

Cùng một cohort 300 study val, cùng thứ tự (đã kiểm `sample_key`), cùng recipe,
cùng ngân sách 10.000 study. Bootstrap ghép cặp, 2.000 lần, seed 16:
**cả 16 khoảng tin cậy đều chứa 0.**

- `A' − A` (kênh soft token): CIDEr +0,0428 [−0,0073, +0,1112], ba metric còn
  lại phẳng hoặc âm nhẹ
- `B' − B` (kênh cue): CIDEr +0,0231 [−0,0125, +0,0631], ba metric còn lại phẳng
- `B − A` và `B' − A'` (cue có giúp không, trên từng Stage-1): đều chứa 0

⚠ Hai điểm ước lượng CIDEr đều dương, nhưng CI rất rộng và ba metric còn lại đi
**ngược chiều** ở cả hai cặp. Một cải thiện thật thì thường kéo theo ít nhất một
metric khác. Nên: **chưa thiết lập được, và không nhất quán.**
