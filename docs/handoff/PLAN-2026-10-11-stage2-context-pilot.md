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

### Execution report — pilot biến thể (2026-10-11 17:59 → 2026-10-12 00:01, `P2VAR_ALL_DONE`)

Run toàn bộ dừng 17:59 sau khi ghi cache train (giữ nguyên). Ba arm `rc=0`. Val 500 study,
ghép cặp với P2, bootstrap 2.000 lần.

| beam 4, lp 2 | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore | CheXbert micro F1-14 | micro F1-5 | val loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P2 (mốc) | 0,423 | 0,178 | 0,172 | 0,345 | 0,518 | 0,448 | 0,463 | 0,494 | 1,229 |
| R16 | 0,416 | 0,176 | 0,171 | 0,348 | 0,555 | 0,451 | 0,454 | 0,490 | 1,212 |
| **LR3** | **0,431** | **0,189** | **0,180** | **0,365** | **0,606** | **0,467** | **0,478** | **0,517** | **1,189** |
| CLEAN | 0,421 | 0,172 | 0,173 | 0,343 | 0,516 | 0,446 | 0,474 | 0,512 | 1,230 |

| hiệu so với P2 | BERTScore | ROUGE-L | CIDEr | BLEU-4 | CheXbert micro F1-14 |
|---|---|---|---|---|---|
| LR3, beam | **+0,0185 [+0,0059; +0,0318]** | **+0,0206 [+0,0100; +0,0322]** | +0,0879 [+0,0285; +0,1555] | +0,0168 [+0,0074; +0,0266] | +0,0155 [−0,0107; +0,0406] |
| LR3, greedy | **+0,0177 [+0,0078; +0,0280]** | **+0,0209 [+0,0108; +0,0312]** | +0,1495 [+0,0460; +0,2638] | +0,0241 [+0,0136; +0,0353] | +0,0111 [−0,0131; +0,0349] |
| R16, beam | +0,0025 [−0,0077; +0,0127] | +0,0032 [−0,0060; +0,0126] | +0,0376 [−0,0186; +0,0992] | +0,0010 [−0,0082; +0,0103] | −0,0089 [−0,0338; +0,0135] |
| R16, greedy | −0,0008 [−0,0113; +0,0100] | +0,0093 [−0,0003; +0,0196] | +0,1052 [+0,0195; +0,2033] | +0,0121 [+0,0022; +0,0225] | −0,0117 [−0,0353; +0,0129] |
| CLEAN, beam | −0,0018 [−0,0113; +0,0074] | −0,0020 [−0,0088; +0,0047] | −0,0013 [−0,0368; +0,0391] | −0,0052 [−0,0112; +0,0007] | +0,0116 [−0,0122; +0,0354] |
| CLEAN, greedy | −0,0080 [−0,0164; −0,0001] | −0,0020 [−0,0083; +0,0044] | +0,0112 [−0,0309; +0,0544] | +0,0036 [−0,0026; +0,0099] | −0,0113 [−0,0355; +0,0135] |

**Theo quy tắc đã chốt:** LR 3e-4 **đạt** (ROUGE-L và BERTScore cận dưới > 0, cả beam và
greedy). LoRA r16 **không đạt** (beam không khác P2). Cache ảnh sạch **không đạt** (không khác
P2; greedy BERTScore hơi thấp hơn) — giữ cache có augmentation, nên cache train đã dựng dùng
lại được.

Giới hạn phải ghi rõ: pilot chỉ ~475 update. LR cao hơn có thể thắng vì mô hình học được
nhiều hơn trong ngân sách ngắn; ở lần train toàn bộ (~10.600 update mỗi epoch) lợi ích có thể
nhỏ hơn. Chưa có phép đo nào cho chế độ dài.

## Train toàn bộ — khởi chạy lại 2026-10-12 00:07 (user duyệt: P2 + LR 3e-4, 2 epoch)

`~/stage2_p2lr3_full.sh`, log `~/stage2_p2lr3.log`, log train `~/stage2_p2_20261011/full_lr3.log`,
output `~/stage2_p2_20261011/full`, mã `ce18c45` (checkout `~/Meta-CXR-v3`, cùng nơi dựng cache).
Lệnh như lần trước cộng `--lora-lr 3e-4`. Xác nhận lúc khởi động: cache train được dùng lại
(`reusing ..._train_qformer_all_09aa2fb4a3955654.pt`); 170.370 record train, 109.140 có lần
chụp trước, 161.362 có indication, 0 record không khớp; cue train 1.747.488 ô từ nhãn thật /
467.322 từ MHCAC; val 1.360, test 2.800 record. Tốc độ ~1,0 it/s, 85.185 vòng mỗi epoch →
~23,5 giờ/epoch, ~47 giờ train, rồi sinh trên test (beam, greedy). Kết thúc bằng
`P2LR3_ALL_DONE`, dự kiến rạng sáng 2026-10-14.

Việc còn lại khi xong: đọc val loss epoch 1 và 2; chấm `gen_test_beam` / `gen_test_greedy`
bằng `score_stage2_paper_protocol.py` và `score_ce_chexbert.py`; so ghép cặp với v3
(`~/stage2_v3_test_beam_20261009/gen/beam4_lp2.jsonl`, khớp theo study — `sample_key` của hai
bên có thể khác cách tính, kiểm tra trước); cập nhật TONG_HOP, CHI_MUC, README, CLAUDE.md.

### Sửa ghi nhận 2026-10-12 00:18 — run 00:07 đã dừng, run được duyệt bắt đầu 00:18:47

Run khởi chạy lúc 00:07 bị dừng ở vòng 348/85.185 theo yêu cầu user (user muốn duyệt danh
sách rà soát và các điểm chưa kiểm chứng TRƯỚC khi train); adapter dở và log nằm ở
`~/stage2_p2_20261011/stopped_0012/`. User duyệt lại cùng cấu hình (P2 + LR 3e-4, 2 epoch,
batch 2 × 8) sau khi được trình: epoch thứ hai và LR 3e-4 ở chế độ dài chưa có phép đo; soft
token sụp chưa sửa; VRAM 15,8/16 GB (nếu hết bộ nhớ: chạy lại với batch 1 × 16); kết quả không
cùng điều kiện với bài báo. Run hiện hành: bắt đầu 2026-10-12 00:18:47, cả ba cache
train/val/test được dùng lại, ~1,0 it/s. Dự kiến `P2LR3_ALL_DONE` rạng sáng 2026-10-14.

## Execution report — train toàn bộ xong 2026-10-14 (P2 + LR 3e-4, 2 epoch)

Lệnh: `~/stage2_p2lr3_full.sh`, exit 0; `P2LR3_ALL_DONE` lúc 2026-10-14 00:34:23. Train
2026-10-12 00:18:47 → 2026-10-14 00:04:03 (47h45m), 85.185 vòng mỗi epoch, ~1,0 it/s, 0 OOM,
0 traceback, máy không reboot, VRAM 12,4–15,8 GB. Mã `ce18c45`.

| epoch | train loss | val loss |
|---|---:|---:|
| 1 | 1,02386 | 0,99053 |
| 2 | 0,85472 | **0,95619** |

Epoch 2 giảm val loss 0,034 nên adapter được giữ là của epoch 2. Train loss giảm 0,169, tức
khoảng cách train–val mở ra (−0,03 → +0,10). Val loss là teacher-forced, không đo chất lượng
sinh. Tham chiếu: nhánh LR3 của pilot (10.000 study) val loss 1,189.

**Lệch so với kế hoạch:** script không truyền `--limit` cho `generate_stage2_reports.py` nên
bước sinh chỉ lấy mặc định **300 study** test (mẫu cố định, seed 16), không phải 2.800. Sinh
300 study: beam 1.102 giây, 0 lỗi, đỉnh VRAM 8,9 GiB; greedy 0 lỗi.

**Khớp study với v3:** `sample_key` khác cách tính (v3 dùng key của record cache, file mới dùng
key tính từ tên ảnh) → 0 key chung. Khớp theo thứ tự dòng của cache test dùng chung: báo cáo
tham chiếu trùng 2.800/2.800 dòng với v3, 300/300 với mỗi file mới (không khớp theo nội dung
tham chiếu vì chỉ có 2.798 tham chiếu khác nhau). Bản v3 mang key mới nằm ở thư mục riêng
`~/stage2_p2_20261011/score/` trên máy train.

**Test, 300 study, giao thức bài báo** (v3 tính lại trên đúng 300 study này; CE = CheXbert):

| | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore | số từ (trung vị) | CE micro F1-14 | CE macro F1-14 | CE micro F1-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 beam | 0,374 | 0,105 | 0,141 | 0,273 | 0,166 | 0,381 | 47 | 0,451 | 0,311 | 0,498 |
| P2+LR3 beam 4, lp 2 | 0,434 | 0,190 | 0,188 | 0,348 | 0,393 | 0,461 | 46 | 0,547 | 0,404 | 0,602 |
| P2+LR3 greedy | 0,367 | 0,164 | 0,174 | 0,350 | 0,383 | 0,473 | 38 | 0,535 | 0,407 | 0,589 |
| bài báo (cả test, không bối cảnh) | 0,390 | 0,102 | 0,173 | 0,280 | 0,291 | 0,426 | — | — | — | — |

Hiệu ghép cặp so với v3 beam (bootstrap 2.000 lần, seed 16, n = 300), mọi khoảng đều không
chứa 0:

| | BERTScore | ROUGE-L | CIDEr | BLEU-4 | CE micro F1-14 | CE micro F1-5 |
|---|---|---|---|---|---|---|
| beam | +0,080 [+0,064; +0,095] | +0,075 [+0,062; +0,090] | +0,227 [+0,138; +0,319] | +0,085 [+0,070; +0,100] | +0,095 [+0,058; +0,133] | +0,104 [+0,056; +0,153] |
| greedy | +0,092 [+0,075; +0,108] | +0,077 [+0,061; +0,094] | +0,217 [+0,085; +0,346] | +0,077 [+0,060; +0,094] | +0,084 [+0,047; +0,120] | +0,092 [+0,048; +0,132] |

**Không cùng điều kiện với bài báo:** mô hình mới nhận thêm indication và FINDINGS của lần
chụp trước trong prompt (2.566 và 2.508 trên 2.800 study test có hai thông tin này), bài báo
thì không; MedGemma thay Vicuna; bài báo chấm trên cả test. Chỉ riêng việc chép FINDINGS lần
trước đã ngang bài báo ở BLEU-1/BLEU-4 (TONG_HOP mục 4.4), nên phần hơn bài báo không chứng
minh kênh ảnh tốt hơn. Một run, một seed, n = 300.

**Đang chạy:** sinh trên cả 2.800 study test, bắt đầu 2026-10-14 00:36, cùng adapter và tham
số, chỉ thêm `--limit 0` (`~/stage2_genfull.sh`, log `~/stage2_genfull.log`, output
`gen_test_beam_full` / `gen_test_greedy_full`, kết thúc bằng `GENFULL_ALL_DONE`, dự kiến
~05:15). Số trên 2.800 study sẽ thay bảng trên. Không train thêm, không đổi pipeline.
