# PLAN 2026-10-14 — Tăng BERTScore bằng giải mã MBR (không train lại)

Trạng thái: **val đã chạy và chấm (2026-10-15), quy tắc chọn M; test chưa chấm** — xem
"Báo cáo thực thi" cuối file. Adapter: P2 + LR 3e-4 (2 epoch,
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

Quyết định này được ghi và commit trước khi chấm test. Mẫu test (`gen_test_samples`, bắt đầu
2026-10-15 01:41:53) đang sinh và chưa được chấm.

### Test

Chưa chạy. `ARM=M bash ~/mbr_test_phase.sh` sau `MBR_PIPELINE_DONE`.
