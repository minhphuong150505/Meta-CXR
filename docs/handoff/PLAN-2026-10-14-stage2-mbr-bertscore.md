# PLAN 2026-10-14 — Tăng BERTScore bằng giải mã MBR (không train lại)

Trạng thái: **code xong, chưa chạy trên GPU.** Adapter: P2 + LR 3e-4 (2 epoch,
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
