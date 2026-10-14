# `scripts/mbr_select_stage2.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-14 · ✅ ACTIVE (chạy trên máy train)

## Purpose
Giải mã **MBR (minimum Bayes risk)** cho Stage 2: với mỗi study, giữ ứng viên có
**kỳ vọng utility cao nhất so với các mẫu do chính mô hình sinh ra**, không bao giờ so
với báo cáo tham chiếu (tham chiếu không tham gia việc chọn). Utility là BERTScore F1
`distilroberta-base`, idf tắt — đúng mô hình và cấu hình của
[`score_stage2_paper_protocol.py`](score_stage2_paper_protocol.py.doc.md). Thêm để tăng
BERTScore mà không train lại.

```text
E[u(h)] = trung bình trên các mẫu y (y khác chính h) của u(h, y)
```

- **Giả thuyết (hypothesis):** `generated` của mỗi file `--hyp` (ví dụ beam, greedy)
  cộng mọi mẫu của `--samples`.
- **Giả tham chiếu (pseudo-reference):** CHỈ các mẫu. Bản giải mã tất định không phải
  một lần rút từ phân phối của mô hình, cho nó "bỏ phiếu" sẽ kéo kỳ vọng về phía nó.
- Một mẫu không bao giờ được chấm với chính nó; hai văn bản giống hệt nhau nhận
  utility 1,0 mà không gọi mô hình.
- Rescale theo baseline là một phép biến đổi affine dương của F1 nên không đổi argmax;
  không áp dụng. Hòa thì lấy ứng viên đứng trước (thứ tự `--hyp`, rồi các mẫu).

## ⚠ Utility chính là mô hình của metric
MBR dự kiến tăng BERTScore nhiều hơn các metric khác. Mọi số sinh ra từ đây phải ghi
"giải mã MBR (utility = BERTScore)" và phải chấm cả BLEU/METEOR/ROUGE-L/CIDEr và
CheXbert.

## Đầu vào / đầu ra
- `--samples candidates_<split>.jsonl` — từ
  [`generate_stage2_reports.py --num-samples N`](generate_stage2_reports.py.doc.md)
  (mỗi dòng có `candidates`, không có `generated`).
- `--hyp NAME=generated_<split>.jsonl` (lặp được, tùy chọn). Tên `sample` được giữ riêng.
- Mọi file phải cùng tập `sample_key` và cùng văn bản tham chiếu cho từng study, nếu
  không thì dừng — đó là dấu hiệu trộn hai cohort. Mỗi study cần ≥ 2 mẫu.
- Ghi `generated_<split>.jsonl` (`sample_key`, `generated`, `reference`, `mbr_source`)
  cho các script chấm, và `mbr_summary.json` — chỉ số tổng hợp: số study, số ứng viên,
  số lần mỗi nguồn được chọn, kỳ vọng utility trung bình theo nguồn, độ dài trung vị.
  Không có văn bản báo cáo. Output đi qua `_assert_private_output_location`.

## Hàm
| Hàm | Vai trò |
|---|---|
| `pool_candidates(hyp_files, samples_file)` | gom ứng viên theo study, kiểm tra cohort và tham chiếu |
| `needed_pairs(texts, is_ref)` | các cặp (i < j) mà kỳ vọng cần đọc: ít nhất một bên là mẫu |
| `expected_utility(utility, is_ref)` | kỳ vọng trên các giả tham chiếu, bỏ đường chéo |
| `select(studies, utility_fn, chunk=250)` | chấm theo khối 250 study (bert_score giữ embedding mỗi câu trong RAM), argmax |
| `bertscore_utility(device, batch_size)` | `BERTScorer` nạp một lần, lần gọi đầu tiên |
| `summarize(studies, hyp_names)` | thống kê không chứa văn bản |

## Calls / Called by
- Gọi: `bert_score.BERTScorer` (import lười, chỉ khi chấm thật),
  `scripts.evaluate_explanation._assert_private_output_location`.
- Được gọi bởi: người chạy trên máy train; xem `docs/handoff/PLAN-2026-10-14-stage2-mbr-bertscore.md`.

## Tests
`tests/test_mbr_select.py` (CPU, utility đồ chơi): ứng viên ngoại lai không được chọn,
mẫu không tự chấm, bản tất định không bỏ phiếu, hòa lấy ứng viên đầu, chỉ chấm cặp cần,
văn bản trùng không gọi mô hình, chia khối không đổi kết quả, từ chối cohort/tham chiếu
khác nhau, `main` ghi summary không chứa văn bản.
