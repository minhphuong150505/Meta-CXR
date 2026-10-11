# PLAN 2026-10-11 — Pilot trước khi train lại toàn bộ (CHỜ USER DUYỆT)

Trạng thái: **đề xuất, chưa sửa code, chưa train.** Căn cứ:
`docs/TONG_HOP_2026-10-11.md`, `docs/CHI_MUC_TINH_MOI.md`.

## Mục tiêu

Chọn cấu hình Stage 2 (và hệ quả cho Stage 1) bằng **một vòng pilot nhỏ có quy
tắc chọn chốt trước**, rồi train toàn bộ dữ liệu **đúng một lần**.

## Vì sao pilot này, không phải pilot khác

- Đòn bẩy NLG lớn nhất trong tài liệu là bối cảnh đầu vào: indication (RaDialog
  BLEU-4 9,5 → 14,8) và lần chụp trước (MAIRA-2 ROUGE-L −28,9% khi bỏ). Dữ liệu
  dự án có sẵn: 91,7% study test có INDICATION/HISTORY; 89,6% test và 64,1% train
  có lần chụp trước với FINDINGS (khoảng cách trung vị 7–9 ngày).
- Chép FINDINGS lần chụp trước (không học gì) đã ngang bài báo ở BLEU-1/BLEU-4 và hơn
  mô hình 0,12 CE lexicon — bằng chứng tại chỗ mạnh nhất cho bối cảnh.
- Cue oracle (nhãn THẬT làm cue) không đổi báo cáo: mô hình hiện tại bỏ qua cue. Muốn
  phân loại tốt hơn có ích cho báo cáo thì phải train để mô hình dựa vào cue.
- Kênh ảnh gốc MedGemma: probe (AUROC 0,690 so với soft token 0,725) và truy hồi tại
  chỗ không ủng hộ → chỉ là arm tùy chọn.
- Sửa Q-Former đòi train lại 1a cần cache đặc trưng ~148 GB trong khi `/home` còn
  59 GB; và RaDialog-align (Q-Former 32 query hoạt động bình thường) chỉ đạt BLEU-4
  9,5 / ROUGE-L 27,1 — ngang mức hiện tại. Vì vậy sửa Q-Former **không** nằm trong
  pilot này; chỉ làm nếu user muốn giữ đúng kiến trúc bài báo.

## Các arm (cùng seed, cùng 10.000 study train, 1 epoch, cùng LoRA r8/α16)

| arm | kênh ảnh | cue MHCAC | bối cảnh | ước tính |
|---|---|---|---|---|
| P0 | 32 soft token (như v3) | Eq. 22 | không | ~1 h train + ~35 phút chấm |
| P1 | 32 soft token | Eq. 22 | indication + FINDINGS lần trước | ~1,1 h + ~35 phút |
| P2 | 32 soft token | lúc train: nhãn thật có nhiễu; lúc suy luận: MHCAC ngưỡng hai mức | indication + FINDINGS lần trước | ~1,1 h + ~35 phút |
| P3 (tùy chọn) | ảnh gốc MedGemma (256 token) | ngưỡng hai mức | indication + FINDINGS lần trước | ~4,5 h + ~1 h |

Chấm trên **500 study val cố định** (cùng `sample_key`, cùng thứ tự), beam 4 +
length penalty 2 và greedy, giao thức bài báo (BLEU/METEOR/ROUGE-L/CIDEr/
BERTScore) + CE lexicon; bootstrap ghép cặp 2.000 lần, seed 16.

## Quy tắc chọn (chốt trước khi có kết quả)

1. **Bối cảnh:** chấp nhận nếu P1 − P0 có ROUGE-L **và** BERTScore với cận dưới
   khoảng tin cậy > 0.
2. **Train để dùng cue:** chấp nhận nếu P2 − P1 có CE lexicon ≥ +0,02 với cận dưới > 0
   và không chỉ số NLG nào giảm có ý nghĩa.
2b. **Kênh ảnh gốc (nếu chạy P3):** vì đắt gấp ~4 lần, chấp nhận nếu P3 − (arm thắng
   giữa P1/P2) đạt BERTScore ≥ +0,01 hoặc CE lexicon ≥ +0,02, cận dưới > 0.
3. Không arm nào thỏa: giữ v3, chỉ thêm beam vào bộ sinh.
4. Sau đó train toàn bộ **một lần** với cấu hình thắng (paper-mode ~19 h/epoch;
   ảnh gốc ~80 h/epoch toàn bộ, hoặc tập con ~80k study ~37 h — user chọn).

## Code cần viết (sau khi duyệt)

1. Tiền xử lý (CPU): thêm `indication_clean`, `prior_findings_clean`,
   `prior_interval_days` vào manifest → `full_allviews_v4` (không đổi target).
2. Prompt: kiểu `paper_context` = prompt bài báo + dòng Indication + dòng FINDINGS
   lần trước (có số ngày); thiếu thì ghi "None".
3. Cue lúc train lấy từ nhãn thật có nhiễu (tỉ lệ lật cố định, seed cố định); lúc
   suy luận vẫn là cue MHCAC. (P3: chế độ ảnh gốc + cue không kèm soft token.)
4. Beam search trong `generate_stage2_reports.py` (port từ probe).
5. Test CPU cho cả bốn; smoke 200 study trên GPU trước pilot.

## Điều kiện dừng

OOM, NaN, smoke không sinh được báo cáo, hoặc cohort 500 study val không khớp
`sample_key` giữa các arm.

## Execution report (2026-10-11, máy train)

`~/pilot_run.sh`, 01:17 → 07:15, `PILOT_ALL_DONE`. Mỗi arm 10.000 study train (7.604
record có FINDINGS hợp lệ), 1 epoch; chấm 500 study val cố định; giao thức bài báo, CE
lexicon và CE CheXbert (`f1chexbert`); bootstrap ghép cặp 2.000 lần. Artifact:
`~/pilot_20261011/` (`scores_*`, `ce_*`).

| beam 4, lp 2 | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore | CE lexicon | CheXbert micro F1-14 | micro F1-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 (như v3) | 0,370 | 0,121 | 0,142 | 0,301 | 0,284 | 0,410 | 0,212 | 0,438 | 0,438 |
| P1 (+ bối cảnh) | 0,408 | 0,165 | 0,168 | 0,336 | 0,487 | 0,437 | 0,289 | 0,441 | 0,441 |
| P2 (+ cue nhãn thật lúc train) | **0,423** | **0,178** | **0,172** | **0,345** | **0,518** | **0,448** | **0,308** | **0,463** | **0,494** |

| greedy | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore | CE lexicon | CheXbert micro F1-14 | micro F1-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0,296 | 0,094 | 0,127 | 0,288 | 0,193 | 0,410 | 0,213 | 0,452 | 0,469 |
| P1 | 0,349 | 0,146 | 0,156 | 0,338 | 0,407 | 0,455 | 0,287 | 0,457 | 0,453 |
| P2 | 0,345 | 0,144 | 0,156 | 0,337 | 0,387 | 0,460 | 0,290 | 0,470 | 0,480 |

Ghép cặp (n = 500):

| | BERTScore | ROUGE-L | CIDEr | BLEU-4 | CE lexicon | CheXbert micro F1-5 |
|---|---|---|---|---|---|---|
| P1 − P0, beam | +0,0275 [+0,0143; +0,0406] | +0,0359 [+0,0229; +0,0490] | +0,2027 [+0,1118; +0,3002] | +0,0386 [+0,0249; +0,0535] | +0,0763 [+0,0370; +0,1131] | +0,0022 [−0,0525; +0,0576] |
| P1 − P0, greedy | +0,0450 [+0,0319; +0,0574] | +0,0496 [+0,0372; +0,0617] | +0,2132 [+0,1338; +0,2965] | +0,0537 [+0,0410; +0,0660] | +0,0744 [+0,0318; +0,1122] | −0,0157 [−0,0653; +0,0379] |
| P2 − P1, beam | +0,0110 [+0,0010; +0,0209] | +0,0081 [−0,0000; +0,0164] | +0,0312 [−0,0145; +0,0740] | +0,0103 [+0,0026; +0,0180] | +0,0190 [−0,0075; +0,0453] | +0,0532 [+0,0158; +0,0889] |
| P2 − P1, greedy | +0,0050 [−0,0025; +0,0124] | −0,0004 [−0,0071; +0,0064] | −0,0195 [−0,0624; +0,0219] | −0,0040 [−0,0108; +0,0028] | +0,0029 [−0,0403; +0,0499] | +0,0266 [−0,0095; +0,0655] |

**Theo quy tắc đã chốt trước:**

1. **Bối cảnh — NHẬN.** P1 − P0 có ROUGE-L và BERTScore với cận dưới > 0 ở cả beam và
   greedy; BLEU-4 +0,04–0,05, CIDEr +0,20, CE lexicon +0,075. CheXbert micro không đổi,
   macro F1-14 tăng (+0,047 beam, +0,068 [+0,022; +0,110] greedy).
2. **Train cue bằng nhãn thật có nhiễu — KHÔNG đạt ngưỡng đã chốt.** CE lexicon beam
   +0,019 [−0,008; +0,045] (ngưỡng: ≥ +0,02 và cận dưới > 0), greedy +0,003. Ngoài quy
   tắc: CheXbert micro F1-5 beam +0,053 [+0,016; +0,089] có ý nghĩa, BERTScore và BLEU-4
   beam tăng có ý nghĩa, không chỉ số NLG nào giảm có ý nghĩa. Tức là có tín hiệu tốt
   nhưng chưa xác lập ở n = 500, một seed — quyết định dùng hay không là của user.

Lưu ý: P0 của pilot (0,121 BLEU-4 trên 500 study val) không so trực tiếp được với v3
trên 2.800 study test; chỉ các hiệu ghép cặp trong pilot là có giá trị. Bài báo: BLEU-1
0,390, BLEU-4 0,102, METEOR 0,173, ROUGE-L 0,280, CIDEr 0,291, BERTScore 0,426 (test,
không dùng bối cảnh).

## Train toàn bộ (user duyệt 2026-10-11: cấu hình P2, 2 epoch)

Stage 1 giữ nguyên (`run_20261005_paper`, pilot sửa BioViL không có lợi — xem
`PLAN-2026-10-11-stage1-improvement.md`). Khởi chạy 2026-10-11 15:50, `~/stage2_p2_full.sh`
→ `~/stage2_p2_20261011/`, mã `ce18c45` (cùng mã với pilot), log `~/stage2_p2_20261011.log`:

```
run_medgemma_qlora.py --pipeline-mode meta_cxr_qformer_with_mhcac_prompt --section-mode findings_only
  --prompt-style paper_context --cue-rule cutpoints --threshold-path configs/stage1_cutpoints/run_20261005_paper.json
  --train-cues oracle_mix --train-cue-model-share 0.2 --train-epochs 2 --batch-size 2 --grad-accum 8
  --save-every-updates 32 --skip-test
```

Sau khi train: sinh trên test với beam 4 + length penalty 2 và greedy
(`gen_test_beam`, `gen_test_greedy`); kết thúc bằng `P2FULL_ALL_DONE`. Chấm bằng
`score_stage2_paper_protocol.py` và `score_ce_chexbert.py`, ghép cặp với v3 trên cùng study.
Ước tính: dựng cache Stage 1 ~1,8 giờ, train ~40 giờ, sinh ~3 giờ. Để giải phóng đĩa đã xóa
các file `.pth` của pilot Stage 1 (`~/s1pilot_20261011`, 21 GB); dự đoán `.npz` và log còn giữ.

### Rà soát trước khi để run chạy 2 ngày (2026-10-11 16:15)

- **Độ dài chuỗi:** đo trên 7.604 record train và 1.360 record val của pilot P2 (cùng prompt):
  prompt p50 / p99 / max = 375 / 480 / 547 token, cả chuỗi (prompt + target) max 676 — dưới
  `--max-length 768`; 0 dòng bị cắt, 0 token target bị mất.
- **Chọn checkpoint:** val loss chấm cuối mỗi epoch, adapter tốt nhất được lưu riêng,
  `--patience 1`; LR cosine trên toàn bộ số update, warm-up 3%.
- **Giữ nguyên như pilot (chưa có phép đo ủng hộ thay đổi):** LoRA r8/α16, LR 1e-4, batch
  2 × 8; record train tính từ ảnh có augmentation; soft token đã sụp.
- Kết luận: không có tối ưu nào đã được kiểm chứng mà run này còn thiếu; không dừng run.

## Pilot biến thể của P2 (user yêu cầu 2026-10-11 17:00) — run toàn bộ tạm dừng

User muốn thử ba điểm "chưa tối ưu" trước khi train 2 ngày. Run toàn bộ (lúc đó mới dựng
cache, chưa train) được dừng ngay sau khi ghi xong cache train; cache giữ lại ở
`~/stage2_p2_20261011/full/.sensitive_stage1_cache/`. Script `~/p2var.sh` →
`~/p2var_20261011/`, mã `3371efd`, cùng 10.000 study train và 500 study val như pilot sáng.

| arm | khác P2 ở |
|---|---|
| R16 | `--lora-rank 16 --lora-alpha 32` |
| LR3 | `--lora-lr 3e-4` |
| CLEAN | `--clean-train-records` (record train tính từ ảnh không augmentation) |

Mốc so sánh: P2 của pilot sáng (`~/pilot_20261011/gen_P2_{beam,greedy}`, mã `ce18c45`; hai
commit chỉ khác ở các cờ mặc định tắt). **Quy tắc chốt trước:** nhận một biến thể nếu so với
P2 (beam, ghép cặp, n = 500) ROUGE-L **và** BERTScore có cận dưới > 0, hoặc CheXbert micro
F1-14 có cận dưới > 0 mà không chỉ số NLG nào giảm có ý nghĩa. Không biến thể nào đạt: train
toàn bộ đúng cấu hình P2 đã định. Các biến thể thắng được gộp lại cho lần train toàn bộ, không
pilot thêm tổ hợp.

Không nằm trong pilot này: sửa soft token sụp (cần sửa Q-Former + train lại 1a, ~1 ngày GPU,
chưa có thiết kế được kiểm chứng) và arm ảnh gốc MedGemma (user không chọn).
