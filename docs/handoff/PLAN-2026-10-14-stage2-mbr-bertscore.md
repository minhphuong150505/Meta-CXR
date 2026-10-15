# PLAN 2026-10-14 — Tăng BERTScore bằng giải mã MBR (không train lại)

Trạng thái: **xong (2026-10-15).** Quy tắc chốt trước chọn M trên val; test (n = 2.800, chạy
một lần) lặp lại phần tăng BERTScore: +0,0191 [+0,0155; +0,0228]; CheXbert micro F1-14 +0,0087
[+0,0012; +0,0161]. BLEU-1/BLEU-4/METEOR/CIDEr toàn tập giảm vì output ngắn hơn — xem "Báo cáo
thực thi" cuối file. Adapter: P2 + LR 3e-4 (2 epoch,
`~/stage2_p2_20261011/full/`), Stage 1 `run_20261005_paper`. Chỉ suy luận; không đổi
checkpoint nào, không đổi bộ chấm.

## Vì sao

Đo trên output test đã có (2.800 study, chỉ số tổng hợp, BERTScore `distilroberta-base`
rescale như giao thức bài báo):

| | Precision | Recall | F1 |
|---|---:|---:|---:|
| beam 4, lp 2 (cấu hình đang báo cáo) | 0,493 | 0,439 | 0,466 |
| greedy | 0,537 | 0,414 | 0,474 |
| bản tốt hơn trong hai bản, chọn theo từng study (oracle, dùng tham chiếu, không đạt được) | | | 0,503 |

- Recall thấp hơn precision ở cả hai: mô hình bỏ sót nội dung.
- Beam dài hơn greedy: recall +0,026 nhưng precision −0,044, F1 −0,009. Kéo dài báo cáo
  bằng giải mã làm BERTScore giảm (ngược chiều BLEU/CIDEr, vốn đang bị phạt độ ngắn:
  BLEU brevity penalty 0,838 beam / 0,660 greedy).
- Hai ứng viên của cùng một study chênh nhau nhiều (oracle 0,503): khâu chọn ứng viên
  còn dư địa. MBR chọn ứng viên có kỳ vọng BERTScore cao nhất so với các mẫu do chính mô
  hình sinh ra, không dùng tham chiếu.

## Code (commit cùng file này)

- `VariantLLM.generate_candidates` (sampling N chuỗi, soft token lặp theo số chuỗi);
  `generate` giữ nguyên kwargs (thân hàm chuyển sang `_generate_texts`).
- `generate_stage2_reports.py --num-samples N --temperature --top-p --top-k` →
  `candidates_<split>.jsonl` (không có khoá `generated`); seed theo study.
- `scripts/mbr_select_stage2.py`: giả thuyết = beam + greedy + 8 mẫu; giả tham chiếu =
  8 mẫu; utility BERTScore F1 `distilroberta-base`, idf tắt.
- Test: `tests/test_mbr_select.py`, `tests/test_beam_generation.py`.

## Các arm (val, cả 1.360 study có FINDINGS hợp lệ, cùng cache record của run train)

| arm | giải mã |
|---|---|
| B | beam 4, length penalty 2,0 (mốc — cấu hình đang báo cáo) |
| G | greedy |
| M | MBR, utility = BERTScore, trên {B, G, 8 mẫu temperature 0,7, top-p 0,95, top-k tắt} |

Tối đa 256 token mới cho mọi arm, như lần sinh test đã báo cáo.

## Quy tắc chọn (chốt trước khi có kết quả)

1. Một arm X ∈ {G, M} **đủ điều kiện** nếu, ghép cặp với B trên val (bootstrap 2.000 lần,
   seed 16): BERTScore có cận dưới khoảng tin cậy > 0, **và** CheXbert micro F1-14 không
   giảm có ý nghĩa (cận trên khoảng tin cậy > 0).
2. Trong các arm đủ điều kiện, chọn arm có BERTScore val cao hơn. Không arm nào đủ điều
   kiện: giữ B và ghi kết quả âm.
3. Arm được chọn chạy **một lần** trên test 2.800 study, ghép cặp với B trên test; báo cáo
   mọi chỉ số của giao thức bài báo và CheXbert, tăng hay giảm đều ghi.
4. Mọi số từ M ghi kèm "giải mã MBR (utility = BERTScore)" — utility là chính mô hình của
   metric nên M được kỳ vọng tăng BERTScore nhiều hơn các chỉ số khác.

Mẫu của test được sinh trong cùng pipeline (để không phí giờ GPU) nhưng **không chấm**
trước khi quyết định trên val được ghi vào file này.

## Lệnh (máy train)

```bash
cd ~/Meta-CXR-v3 && git pull        # checkout của run train, cùng cache record
EXPECT_REV=<commit> setsid nohup bash ~/mbr_pipeline.sh > ~/stage2_mbr_20261014.log 2>&1 < /dev/null &
```

`~/mbr_pipeline.sh` → `~/stage2_mbr_20261014/`:
0. Kiểm tra tương đương: sinh lại 20 study test đầu bằng beam với code mới, phải trùng
   từng chữ với `~/stage2_p2_20261011/gen_test_beam_full` (nếu không thì dừng).
1–3. Val: `gen_val_beam`, `gen_val_greedy`, `gen_val_samples` (`--num-samples 8`).
4. `mbr_val` (`mbr_select_stage2.py`).
5. Chấm val: `score/val_paper.json` (giao thức bài báo, `--baseline B`),
   `score/val_ce.json` (CheXbert, label cache chép từ run trước) → in `VAL_SCORED`.
6. `gen_test_samples` (`--split test --limit 0 --num-samples 8`) → `MBR_PIPELINE_DONE`.

Ước tính: beam val ~85 phút, greedy ~55 phút, mẫu val ~2 giờ, chấm ~20 phút, mẫu test ~4 giờ.

## Điều kiện dừng

GPU không rảnh lúc khởi động, ổ dữ liệu không phải `ntfs3`, thư mục output đã tồn tại,
checkout không đúng commit, kiểm tra tương đương không trùng 20/20, bất kỳ bước nào trả mã
khác 0, hoặc `mbr_select_stage2.py` báo cohort/tham chiếu không khớp.

## Báo cáo thực thi

### Val — 2026-10-15, chấm và quyết định TRƯỚC khi chấm bất kỳ thứ gì trên test

`~/mbr_pipeline.sh` chạy một lần từ 2026-10-14 21:35:50, checkout `~/Meta-CXR-v3` ở `6dd5688`
(`EXPECT_REV=6dd5688`), log `~/stage2_mbr_20261014.log`, output `~/stage2_mbr_20261014/`.

- Kiểm tra tương đương: **20/20** beam test sinh lại trùng từng chữ với bản đã báo cáo.
- `gen_val_beam` 1.360 study, 0 lỗi, 4.460 s; `gen_val_greedy` 1.360, 0 lỗi, 2.782 s;
  `gen_val_samples` 1.360 × 8 mẫu, 0 lỗi, 7.135 s (~0,19 study/s, ~11,9 GB VRAM).
- `mbr_val`: 59.840 cặp, 18,5 s trên GPU (BERTScore batch 64). Ứng viên được chọn: beam 393,
  greedy 576, mẫu 391. Kỳ vọng utility trung bình (F1 thô): beam 0,942 / greedy 0,947 /
  mẫu 0,939 / ứng viên được chọn 0,951.
- Chi phí suy luận của M ≈ beam + greedy + 8 mẫu ≈ 10,6 s/study, khoảng 3,2 lần beam (3,3 s/study).

Chỉ số trên toàn tập val (n = 1.360, giao thức bài báo; CheXbert = `score_ce_chexbert.py`):

| arm | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore | từ (median) | CheXbert micro F1-14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B (beam 4, lp 2) | 0,4567 | 0,2199 | 0,2014 | 0,3928 | 0,7066 | 0,5070 | 36 | 0,5447 |
| G (greedy) | 0,4026 | 0,1965 | 0,1899 | 0,3937 | 0,7016 | 0,5121 | 32 | 0,5377 |
| M (MBR, utility = BERTScore) | 0,4123 | 0,1986 | 0,1941 | 0,3960 | 0,7149 | **0,5269** | 32 | **0,5609** |

Ghép cặp với B (bootstrap 2.000 lần, seed 16), Δ [khoảng tin cậy 95%]:

| chỉ số | G − B | M − B |
|---|---|---|
| **BERTScore** | +0,0051 [−0,0017; +0,0120] | **+0,0199 [+0,0132; +0,0262]** |
| **CheXbert micro F1-14** | −0,0070 [−0,0198; +0,0067] | **+0,0162 [+0,0045; +0,0274]** |
| CheXbert macro F1-14 | +0,0004 [−0,0192; +0,0214] | +0,0184 [−0,0036; +0,0389] |
| CheXbert micro F1-5 | −0,0086 [−0,0275; +0,0129] | **+0,0214 [+0,0042; +0,0380]** |
| CheXbert macro F1-5 | −0,0063 [−0,0294; +0,0195] | +0,0173 [−0,0076; +0,0425] |
| ROUGE-L | +0,0009 [−0,0063; +0,0085] | +0,0032 [−0,0043; +0,0103] |
| CIDEr | −0,0050 [−0,0899; +0,0820] | +0,0084 [−0,0764; +0,0910] |
| BLEU-4 (theo study) | −0,0051 [−0,0139; +0,0039] | −0,0046 [−0,0133; +0,0041] |
| lexicon CE macro F1 | +0,0182 [−0,0034; +0,0408] | **+0,0247 [+0,0010; +0,0473]** |
| output trùng B | 162 / 1.360 | 397 / 1.360 |

**Áp quy tắc:** G không đủ điều kiện (cận dưới BERTScore −0,0017 ≤ 0). M đủ điều kiện (cận
dưới BERTScore +0,0132 > 0 và cận trên CheXbert micro F1-14 +0,0274 > 0). **Chọn M.** Bước test:
M chạy một lần trên 2.800 study test, giả thuyết = beam và greedy test đã báo cáo
(`~/stage2_p2_20261011/gen_test_{beam,greedy}_full`, sinh ở `ce18c45`; kiểm tra tương đương ở
trên cho thấy code `6dd5688` tái tạo đúng beam đó) cộng 8 mẫu test, ghép cặp với B = beam test
đã báo cáo. Đã kiểm tra trước: hai file test có cùng 2.800 khoá, tham chiếu trùng 2.800/2.800;
31 dòng mẫu test đầu tiên khớp khoá và tham chiếu, mỗi dòng 8 ứng viên.

Lưu ý khi đọc:

- Utility là chính mô hình của BERTScore, nên phần tăng BERTScore một phần là do cách chọn.
  CheXbert (micro F1-14, micro F1-5) và lexicon CE không dùng mô hình đó và cũng tăng có ý
  nghĩa trên val — dấu hiệu MBR chọn được báo cáo nhất quán lâm sàng hơn, chưa kiểm trên test.
- BLEU-1 toàn tập 0,4567 → 0,4123 và METEOR 0,2014 → 0,1941: output ngắn hơn (median 36 → 32
  từ) nên brevity penalty của BLEU nặng hơn. Bộ chấm không cho khoảng tin cậy của BLEU-1/METEOR
  toàn tập; BLEU-4 theo study không đổi có ý nghĩa.
- Số val cao hơn số test của cùng cách giải mã (B: BERTScore 0,507 val so với 0,466 test;
  CIDEr lấy IDF từ tham chiếu của chính tập được chấm nên không so được giữa hai tập). Không so
  val với test.
- Không phải giao thức của bài báo: prompt có indication và FINDINGS lần trước, cộng giải mã MBR.

Quyết định này được ghi và commit (`abb7c15`, push 01:47:53) trước khi chấm test. Lúc đó mẫu
test (`gen_test_samples`, bắt đầu 2026-10-15 01:41:53) đang sinh và chưa được chấm.

### Test — 2026-10-15, chạy một lần theo quy tắc đã chốt

`gen_test_samples` xong 06:18:49: 2.800 study × 8 mẫu, 0 lỗi, 16.594 s (~0,17 study/s).
`~/mbr_test_wait.sh` thấy `MBR_PIPELINE_DONE` và chạy `ARM=M bash ~/mbr_test_phase.sh` một lần
(checkout `6dd5688`, GPU rảnh); `TEST_SCORED arm=M` lúc 06:22:29. Log
`~/stage2_mbr_20261014_test.log`, output `~/stage2_mbr_20261014/{mbr_test,score}/`.

- `mbr_test`: 123.200 cặp, 39,8 s trên GPU. Ứng viên được chọn: beam 625, greedy 1.327, mẫu 848.
  Kỳ vọng utility trung bình (F1 thô): beam 0,937 / greedy 0,942 / mẫu 0,933 / ứng viên được
  chọn 0,945.
- B là beam test đã báo cáo, chấm lại ra đúng số đã báo cáo (BLEU-1 0,434 / BLEU-4 0,190 /
  METEOR 0,190 / ROUGE-L 0,355 / CIDEr 0,416 / BERTScore 0,466; CheXbert micro F1-14 0,564).

Chỉ số trên toàn tập test (n = 2.800, giao thức bài báo; CheXbert = `score_ce_chexbert.py`):

| arm | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore | từ (median) | CheXbert micro F1-14 | CheXbert macro F1-14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B (beam 4, lp 2) | 0,4337 | 0,3082 | 0,2366 | 0,1900 | 0,1897 | 0,3549 | 0,4158 | 0,4656 | 47 | 0,5644 | 0,4253 |
| M (MBR, utility = BERTScore) | 0,3824 | 0,2763 | 0,2138 | 0,1722 | 0,1815 | 0,3599 | 0,3885 | **0,4847** | 40 | **0,5730** | 0,4354 |
| bài báo | 0,390 | | | 0,102 | 0,173 | 0,280 | 0,291 | 0,426 | | | |

Ô trống: bài báo không báo (CE của bài báo dùng bộ gán nhãn CheXpert, không so trực tiếp được).
CheXbert macro-14 precision / recall: B 0,517 / 0,398, M 0,535 / 0,406.

Ghép cặp với B (bootstrap 2.000 lần, seed 16), Δ [khoảng tin cậy 95%]; cột cuối nhắc lại M − B
trên val (bảng ở trên):

| chỉ số | M − B, test | M − B, val |
|---|---|---|
| **BERTScore** | **+0,0191 [+0,0155; +0,0228]** | +0,0199 [+0,0132; +0,0262] |
| **CheXbert micro F1-14** | **+0,0087 [+0,0012; +0,0161]** | +0,0162 [+0,0045; +0,0274] |
| CheXbert macro F1-14 | +0,0101 [−0,0014; +0,0218] | +0,0184 [−0,0036; +0,0389] |
| CheXbert micro F1-5 | +0,0085 [−0,0004; +0,0175] | +0,0214 [+0,0042; +0,0380] |
| CheXbert macro F1-5 | +0,0096 [−0,0026; +0,0216] | +0,0173 [−0,0076; +0,0425] |
| **ROUGE-L** | **+0,0050 [+0,0016; +0,0084]** | +0,0032 [−0,0043; +0,0103] |
| CIDEr | −0,0274 [−0,0593; +0,0022] | +0,0084 [−0,0764; +0,0910] |
| BLEU-4 (theo study) | −0,0038 [−0,0076; +0,0001] | −0,0046 [−0,0133; +0,0041] |
| lexicon CE macro F1 | +0,0027 [−0,0135; +0,0195] | +0,0247 [+0,0010; +0,0473] |
| output trùng B | 628 / 2.800 | 397 / 1.360 |

Kết luận:

- **BERTScore tăng trên test**: 0,4656 → 0,4847, +0,0191 [+0,0155; +0,0228], gần đúng mức đo
  trên val. Mục tiêu của plan đạt và lặp lại trên tập giữ riêng.
- Phần tăng đến từ precision, không phải recall (đo lại bằng `bert_score`, cùng mô hình và
  rescale, F1 khớp bộ chấm): precision 0,493 → 0,545, recall 0,439 → 0,426. MBR chọn bản ngắn
  và "đồng thuận" hơn; nó không sửa lỗi bỏ sót nội dung đã nêu ở mục "Vì sao" — recall còn giảm
  nhẹ. Chọn theo từng study giữa B và M bằng tham chiếu (oracle, không đạt được) cho F1 0,505;
  M hơn B ở 49,5% study.
- CheXbert micro F1-14 tăng có ý nghĩa nhưng chỉ bằng khoảng một nửa mức val (+0,0087 so với
  +0,0162); macro F1-14, micro F1-5 và macro F1-5 dương nhưng khoảng tin cậy chạm 0. Lexicon CE,
  tăng có ý nghĩa trên val, không đổi trên test. Ý "MBR chọn báo cáo nhất quán lâm sàng hơn" chỉ
  còn đứng ở micro F1-14, với mức tăng nhỏ.
- ROUGE-L tăng nhỏ, có ý nghĩa (+0,0050).
- Các chỉ số thưởng độ dài giảm: output ngắn hơn (median 47 → 40 từ), BLEU-1 toàn tập
  0,434 → 0,382, BLEU-4 0,190 → 0,172, METEOR 0,190 → 0,181, CIDEr 0,416 → 0,388 (ghép cặp
  −0,0274 [−0,0593; +0,0022], không có ý nghĩa). ⚠ BLEU-1 của M (0,382) thấp hơn bài báo
  (0,390); của B (0,434) cao hơn.
- Chi phí suy luận của M ≈ 3,2 lần beam.
- Không phải giao thức của bài báo: prompt có indication và FINDINGS lần trước, cộng giải mã MBR
  mà utility chính là mô hình của BERTScore.

Số chính trong `docs/SO_SANH_VOI_BAI_BAO.md` vẫn là beam (B); M được thêm thành một bảng riêng
ở mục 2.2. Đổi số chính sang M hay không là quyết định của người dùng.
