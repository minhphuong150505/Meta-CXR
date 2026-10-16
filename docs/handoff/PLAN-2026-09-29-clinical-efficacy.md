# PLAN 2026-09-29 — Cải thiện Clinical Efficacy (CE) của báo cáo sinh ra

Planner: Claude (dev box, không GPU). Trạng thái: **đề xuất, chưa chạy bước nào.**
Liên quan: D-023 (ba lớp, không nhị phân), `training/evaluation/paper_protocol.py`,
`scripts/evaluate_clinical_efficacy.py`.

## Mục tiêu

Tăng CE precision / recall / F1 của Stage 2 (báo cáo FINDINGS), so được với
Bảng 3 của bài báo:

| | Precision | Recall | F1 (macro) |
|---|---:|---:|---:|
| Bài báo META-CXR | 0,411 | 0,465 | 0,428 |
| Repo này | **chưa đo** | **chưa đo** | **chưa đo** |

CE là phép tính nhị phân (chỉ đếm nhãn labeler = 1) vì bài báo tính như vậy.
Đây là một trong hai ngoại lệ D-023 cho phép. Mọi dự đoán Stage 1 dùng làm cue
hoặc để rerank vẫn là **ba lớp**.

## Dữ kiện đã biết (định hướng kế hoạch)

1. **Chưa có labeler trong repo**, nên CE chưa từng được đo. Mọi kết luận Stage 2
   trước đây chỉ dựa trên metric từ vựng (BLEU/ROUGE/METEOR/CIDEr/BERTScore).
2. **Báo cáo sinh ra quá ngắn.** Sau khi sửa stop token (2026-09-09), trung vị là
   24 từ so với 52 từ của báo cáo thật, trên 100 ca val. Báo cáo ngắn thì bỏ sót
   bệnh, nên recall CE gần như chắc bị kéo xuống.
3. **Chưa thí nghiệm cue nào đo CE.** Bảy lần kết luận "cue không giúp" đều dựa
   trên metric từ vựng. Cue chẩn đoán có thể không đổi n-gram nhưng vẫn làm báo
   cáo đúng bệnh hơn. PromptMRG báo cáo CE F1 0,476 trên MIMIC-CXR nhờ prompt
   chẩn đoán.
4. **Có sẵn checkpoint Stage 2** (arm A `medgemma_direct`, arm C
   `native_qformer_guided`, 4 arm pilot finding-token) và một cohort 300 ca val đã
   cố định theo `sample_key`. Các bước 1–3 dưới đây vì vậy chỉ cần inference.
5. Các cue rule cũ (`marginal_positive`, ...) đã bị gỡ. Rule hiện có: `argmax`,
   `paper_thresholds`, `none`.

## Nguyên tắc đo

- Mọi so sánh dùng **cùng cohort, cùng thứ tự** (`--restrict-to <jsonl>`), cùng
  cách decode trừ biến đang thử, và bootstrap theo cặp (2.000 lần, seed 16).
- Báo cáo P, R, F1 cả macro lẫn micro, kèm CI, và luôn ghi độ dài trung vị.
- **Labeler dùng để chấm CE không được là labeler dùng để chọn/sửa báo cáo.**
  Ví dụ: nếu rerank bằng `safety/claims.py` thì phải chấm bằng CheXbert.
- Ghi rõ labeler nào, phiên bản nào, và nhãn tham chiếu lấy từ đâu: labeler chạy
  lại trên báo cáo gốc, hay file CheXpert CSV của MIMIC. Hai nguồn cho số khác
  nhau, và bài báo không nói rõ nó dùng nguồn nào.
- Val để chọn cấu hình, test chỉ chạy **một lần** ở cuối.

## Các hướng, theo thứ tự thực hiện

### Bước 0 — Đo được CE (bắt buộc)

- Cài labeler, ưu tiên **CheXbert** (chuẩn phổ biến cho CE hiện nay). Phương án
  dự phòng là CheXpert labeler dạng rule-based. Cài vào venv riêng trên host,
  không cài vào venv Stage 1.
- Viết script gán nhãn JSONL (báo cáo sinh ra + báo cáo gốc) thành 2 CSV
  `sample_key` + 14 cột. Output để ở thư mục private, không commit.
- Đo baseline trên 300 ca val cho: zero-shot MedGemma, arm A, arm C.
- Kiểm tra độ tin cậy: chạy labeler trên chính báo cáo gốc so với CheXpert CSV
  của MIMIC.
- **Xong khi:** có bảng P/R/F1 kèm CI cho 3 hệ thống.

### Bước 1 — Chữa báo cáo quá ngắn (nhắm recall)

- Thử `min_new_tokens` (ví dụ 40/60), length penalty và beam search (4) trên
  checkpoint tốt nhất của bước 0.
- Theo dõi repetition: 5-gram lặp ≥3 lần phải giữ gần 0%, vì lỗi này đã được
  sửa và không được quay lại. `--no-repeat-ngram-size 5` là phương án dự phòng.
- **Kỳ vọng:** recall tăng, precision có thể giảm nhẹ. Chọn theo F1.

### Bước 2 — Đo lại cue bằng CE (giả thuyết kiểu PromptMRG)

- Arm C (hoặc arm được train với cue), cùng 300 ca: `--cue-rule none` so với
  `argmax` so với `paper_thresholds` (ngưỡng Eq. 22 fit trên val bằng
  `scripts/calibrate_thresholds.py`).
- ⚠ Các arm cũ được train với cue rule đã gỡ, nên đổi cue lúc inference là can
  thiệp lệch phân phối. Nếu có tín hiệu thì mới train một arm với `argmax` cue để
  xác nhận.
- **Kỳ vọng:** nếu cue giúp CE, hiệu ứng thể hiện ở precision và recall của các
  bệnh Stage 1 phân loại tốt (Edema, Pleural Effusion, ...).

### Bước 3 — Rerank ứng viên theo Stage 1 (nhắm precision)

- Sinh N ứng viên (N = 4–8, sampling hoặc beam). Gán nhãn từng ứng viên bằng
  `safety/claims.LexiconClaimParser` (14 nhãn, chạy CPU). Chọn ứng viên khớp nhất
  với dự đoán **ba lớp** của MHCAC, theo điểm đồng thuận có trọng số xác suất.
- Chấm CE bằng CheXbert (labeler độc lập).
- Tham khảo: CCS (clinical consensus selection), FactReranker.
- **Chi phí:** thời gian sinh báo cáo nhân N. Trần hiệu quả bị giới hạn bởi độ
  chính xác Stage 1 và coverage của parser (0,65 theo `lexicon_v2`).

### Bước 4 — Cải thiện Stage 1 (nâng trần cho bước 2–3)

- Chỉ làm nếu bước 2–3 cho thấy CE bị giới hạn bởi chất lượng phân loại.
- Ứng viên: logit-adjusted / disease-balanced loss (như PromptMRG), vẫn **ba
  lớp**. Đánh giá bằng giao thức bài báo (`scripts/evaluate_stage1.py`).
- **Chi phí:** một run Stage 1, khoảng 13 giờ cộng đánh giá.

### Bước 5 — RL với reward lâm sàng (tác động lớn nhất, đắt nhất)

- SCST hoặc GRPO trên adapter Stage 2, reward = F1 CheXbert (có thể cộng
  BERTScore để giữ độ trôi chảy). Tài liệu báo cáo mức tăng lớn: Miura et al.
  2021 tăng +22,1 điểm F1 trích xuất thông tin lâm sàng.
- ⚠ Rủi ro: card 16 GB với MedGemma 4B QLoRA và rollout nhiều mẫu. Cần đo VRAM
  và throughput trước. Reward hacking cũng có thể xảy ra (báo cáo liệt kê bệnh
  tràn lan), nên theo dõi cả precision lẫn độ dài.
- Chỉ bắt đầu khi bước 0–3 đã có baseline và pipeline chấm CE ổn định.

## Tóm tắt ưu tiên

| # | Hướng | Nhắm vào | Cần train? | Chi phí |
|---|---|---|---|---|
| 0 | Labeler + baseline CE | đo lường | không | vài giờ |
| 1 | Chữa độ dài | recall | không | thấp |
| 2 | Cue ba lớp, đo bằng CE | P + R | không (xác nhận: có) | thấp |
| 3 | Rerank theo Stage 1 | precision | không | trung bình |
| 4 | Loss cân bằng bệnh ở Stage 1 | trần của 2–3 | có | ~13 giờ |
| 5 | RL reward lâm sàng | P + R | có | cao |

## Điều kiện dừng / cảnh báo

- Labeler trên báo cáo gốc lệch nhiều so với CheXpert CSV (> 10% ô): dừng lại và
  quyết định nguồn nhãn tham chiếu trước khi so sánh gì.
- Một can thiệp làm tăng recall nhưng repetition hoặc độ dài vượt xa báo cáo thật:
  không chấp nhận, vì đó là inflate chứ không phải cải thiện.
- CI chạm 0 thì ghi "chưa xác lập", không ghi "cải thiện".

## Không được làm

- Không đưa lại mention gate hay bất kỳ khung nhị phân nào (D-023). Nếu một hướng
  có vẻ cần, hỏi người dùng trước.
- Không commit báo cáo sinh ra, nhãn labeler, JSONL/CSV có `sample_key` gắn với
  dữ liệu, hay checkpoint.
- Không gọi metric từ vựng là độ chính xác lâm sàng. RadGraph/RadCliQ không có
  trong repo, nên ghi "không có", không ghi 0.

## Tài liệu tham khảo

- PromptMRG (AAAI 2024): https://arxiv.org/abs/2308.12604
- Miura et al. 2021, NAACL: https://aclanthology.org/2021.naacl-main.416/
- CCS, Clinical Consensus Selection: https://arxiv.org/html/2605.30131
- FactReranker: https://arxiv.org/pdf/2303.08335

## Execution report

_(executor ghi vào đây)_
