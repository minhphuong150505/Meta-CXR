# Chẩn đoán Stage 1 / Stage 2 và đề xuất cải tiến (2026-10-11)

Câu hỏi của user: vì sao soft token của META-Former gần như ngẫu nhiên với một số
bệnh; độ dài và config các phase Stage 1 có hợp lý không; vì sao báo cáo sinh ra
yếu; cần sửa gì ở cả pipeline để không phải train lại nhiều lần. Mọi số dưới đây đo
trên máy train từ các checkpoint hiện có (`run_20261005_paper`, adapter Stage 2 v3),
**không train thêm gì**. Chỉ có số tổng hợp — không văn bản báo cáo, không ID.

Script (đều chỉ đọc, chạy trên máy train): `scripts/diagnose_qformer_collapse.py`,
`probe_soft_tokens_cached.py`, `retrieval_report_baselines.py`,
`analyze_stage2_generations.py`, `prior_report_baseline.py`,
`embed_medgemma_images.py`, `probe_image_features.py`. Kết quả thô:
`~/diag_qformer_20261011b/`, `~/diag_qformer_20261011c/`, `~/probe_softtok_20261011/`,
`~/stage2_analysis_20261011/`, `~/eval_1b_test_20261011/`.

## 1. Soft token META-Former: sụp thành MỘT vector, và sụp ở phase 1a

`diagnose_qformer_collapse.py`, 200 study val, mỗi checkpoint:

| checkpoint | cosine giữa 32 token / study | số hướng hiệu dụng (PR) | entropy cross-attn tầng 3 (1 = đều) | độ giống bản đồ attention giữa các query (tầng 3) |
|---|---:|---:|---:|---:|
| khởi tạo BLIP-2 | 0,445 | 2,94 | 0,61 | 0,27 |
| sau **1a** | **0,9989** | **1,002** | **0,11** | **1,000** |
| sau 1b | 0,9989 | 1,002 | 0,15 | 0,978 |
| sau 1c (soft token v3 dùng) | 0,9997 | 1,001 | 0,09 | 0,906 |
| 1c của run cũ 0925b | 0,9999 | 1,000 | 0,05 | 0,999 |

- Ngay sau 1a, từ tầng cross-attention thứ hai trở đi **cả 32 query nhìn đúng một
  token ảnh** (trọng số lớn nhất ~0,79). Hidden state của 32 query trùng nhau từ tầng 3
  (cosine 1,000). Lỗi có tính hệ thống: run cũ cũng vậy.
- Mục tiêu của 1a không đòi hỏi các query khác nhau: ITC lấy max theo query, LM dùng
  query làm prefix, ITM lấy trung bình — và **ITM ở mức đoán bừa suốt 4 epoch**
  (val 0,644 → 0,670, chance 0,6365). Val ITC loss tăng sau epoch 1 (2,95 → 3,38)
  trong khi R@5 đứng ở 0,61–0,67.
- **Phase 1b đổi đầu vào của Q-Former trong khi Q-Former đã đóng băng** (fade-out ở 10%
  đầu): stream adapter (train bởi MHCAC + MPC, không có LayerNorm sau nó ở PubMedCLIP và
  Swin) đẩy norm token PubMedCLIP 19,6 → 1.713 và Swin 12 → 413, đồng thời thêm một
  thành phần chung cho cả ảnh: cosine giữa các token trong một ảnh 0,01 → 0,45 (BioViL),
  0,71 (PubMedCLIP), 0,65 (Swin) — token ảnh mất phần lớn tính không gian.
- **Phase 1c (1 epoch, batch 8) không kéo lại được**: ITC R@5 0,65 (sau 1a) → **0,17**
  (sau 1c); thay mọi token ảnh bằng token trung bình của ảnh vẫn cho đầu ra cosine 0,81
  với đầu ra thật — Q-Former gần như chỉ đọc "ảnh trung bình".
- Hệ quả: linear probe trên soft token (fit 50k train, chọn L2 trên val, chấm test
  2.780 study) giữ được bệnh phổ biến (Edema 0,848 vs MHCAC 0,843, Pleural Effusion
  0,896 vs 0,899) nhưng mất bệnh khu trú/hiếm (Fracture 0,488 = đoán bừa, Pleural Other
  0,635 vs 0,833, Lung Lesion 0,715 vs 0,748). Một vector toàn cục không chứa được
  bằng chứng khu trú, và các mục tiêu ở mức cả báo cáo (ITC/LM) bị bệnh phổ biến chi phối.

_(thí nghiệm đang chạy trên máy train — mục này được cập nhật khi có kết quả)_

## 2. Stage 1: độ dài và config các phase

| phase | quan sát | kết luận |
|---|---|---|
| 1a (4 epoch, batch 128) | gate R@5 0,61 / 0,63 / 0,66 / 0,67; val ITC loss tăng sau epoch 1; ITM = chance | 2 epoch là đủ; ITM không đóng góp; vấn đề là **sụp token**, không phải thiếu epoch |
| 1b (5 epoch) | val AUROC 0,757 / 0,764 / **0,778** / 0,775 / 0,773; macro recall đỉnh epoch 2 | 3 epoch là đủ (tiết kiệm ~40% của 7 giờ) |
| 1c (1 epoch, batch 8) | val AUROC 0,778 (1b tốt nhất) → 0,771; macro recall 0,492 → 0,486; R@5 0,65 → 0,17; gradient căn chỉnh lớn gấp 7–35 lần gradient phân loại trên projector chung, cosine ≈ 0 | 1c **không giúp phân loại** trên val và **phá** căn chỉnh |
| stream adapter | norm token PubMedCLIP/Swin tăng 90× / 34× trong 1b | thiếu chuẩn hoá sau adapter |

_(thí nghiệm đang chạy trên máy train — mục này được cập nhật khi có kết quả)_

## 3. Vì sao báo cáo sinh ra yếu

Test v3, 2.800 study, giao thức bài báo. Mốc truy hồi = chép nguyên FINDINGS của study
train (60.000 study) gần nhất theo một khoá. CE lexicon = `LexiconClaimParser` gán nhãn
cả hai phía (xấp xỉ tương đối, không phải CheXbert).

| | BLEU-4 | ROUGE-L | CIDEr | BERTScore | CE lexicon macro F1 | F1-5 |
|---|---:|---:|---:|---:|---:|---:|
| **v3 beam (sinh)** | **0,098** | **0,265** | **0,128** | **0,369** | 0,239 | 0,379 |
| v3 greedy | 0,073 | 0,250 | 0,084 | 0,361 | 0,230 | 0,346 |
| một báo cáo cố định cho mọi study | 0,061 | 0,238 | 0,062 | 0,340 | 0,060 | 0,128 |
| báo cáo train ngẫu nhiên | 0,046 | 0,202 | 0,035 | 0,290 | 0,164 | 0,235 |
| chép theo soft token | 0,062 | 0,220 | 0,066 | 0,315 | 0,259 | 0,362 |
| chép theo nhãn dự đoán MHCAC | 0,057 | 0,213 | 0,052 | 0,304 | **0,294** | **0,455** |
| chép theo nhãn THẬT (oracle) | 0,072 | 0,230 | 0,085 | 0,334 | **0,416** | **0,559** |

_(thí nghiệm đang chạy trên máy train — mục này được cập nhật khi có kết quả)_

Nguyên nhân, theo bằng chứng:

1. **Kênh thị giác chỉ còn một vector** (mục 1). Hoán đổi soft token giữa các study làm
   BERTScore −0,072: LLM có dùng, nhưng chỉ có nội dung toàn cục.
2. **LLM không chuyển thông tin phân loại vào văn bản.** Chép một báo cáo thật có nhãn
   khớp dự đoán MHCAC đạt CE lexicon 0,294, cao hơn báo cáo LLM sinh (0,239). Hoán đổi
   cue chỉ −0,014 BERTScore; cue theo ngưỡng mới không đổi NLG. Theo từng bệnh, mô hình
   **không bao giờ** nhắc Fracture, Lung Lesion, Pleural Other, Enlarged
   Cardiomediastinum (mention recall 0,00), rất ít Pneumonia (0,17) và Lung Opacity
   (0,24), nhưng luôn viết câu âm tính quen thuộc (pneumothorax 0,95, effusion 0,87).
3. **Metric NLG thưởng văn phong, không thưởng nội dung lâm sàng.** Một báo cáo cố định
   có ROUGE-L 0,238, cao hơn chép báo cáo có nhãn đúng hoàn toàn (0,230). Vì vậy phân
   loại tốt hơn gần như không làm BLEU/ROUGE nhích. Tài liệu xác nhận: PromptMRG thêm
   prompt chẩn đoán thì CE F1 0,370 → 0,444 nhưng BLEU-4 0,116 → 0,106; RaDialog thêm
   danh sách bệnh thì CE 26,1 → 39,4, BLEU-4 9,0 → 9,5.
4. **Thiếu bối cảnh lần chụp trước.** 90,0% FINDINGS test có ngôn ngữ so sánh thời gian;
   89,6% study test có lần chụp trước với FINDINGS; mô hình không có. Nó tự viết câu so
   sánh trong 51% báo cáo mà bản gốc không so sánh. MAIRA-2: bỏ ảnh/báo cáo trước lúc suy
   luận làm ROUGE-L −28,9%, train không có nó −11,7%.
5. **Thiếu chỉ định (indication).** 91,7% study test có INDICATION/HISTORY (trung vị 8
   từ). RaDialog thêm indication: BLEU-4 9,5 → 14,8, ROUGE-L 27,1 → 31,6.
6. **Viết thiếu, viết ngắn.** BERTScore P 0,424 / R 0,299 (greedy); 37 từ (greedy) / 47
   từ (beam) so với 56; chỉ nhắc 32–34% các bệnh mà bác sĩ có bàn tới.
7. **Stage 2 train chưa tới.** 1 epoch, LoRA r 8, LR 1e-4; loss train vẫn giảm ở cuối
   (1,028 → 1,016 hai thập phân vị cuối); val 1,061 ≈ train 1,080. RaDialog: 4 epoch,
   LR 3e-4. Kênh nhìn ảnh gốc của MedGemma (MedSigLIP, đã train trên 231k ảnh +
   báo cáo MIMIC-CXR) bị bỏ hoàn toàn ở chế độ paper.

_(thí nghiệm đang chạy trên máy train — mục này được cập nhật khi có kết quả)_

_(thí nghiệm đang chạy trên máy train — mục này được cập nhật khi có kết quả)_

## 4. Tài liệu: đòn bẩy nào thật sự tăng metric trên MIMIC-CXR

| nguồn | thay đổi | hiệu quả đo được |
|---|---|---|
| MAIRA-2 (arXiv 2406.04449) | ảnh + báo cáo lần trước, mục comparison | bỏ lúc suy luận: ROUGE-L −28,9%; train không có: −11,7% |
| MAIRA-2 | ảnh lateral + technique | bỏ lúc suy luận: ROUGE-L −22% (tập có lateral) |
| MAIRA-2 7B | đầy đủ bối cảnh, MLP 4 lớp trên 1.369 token RAD-DINO | ROUGE-L 38,4, BLEU-4 23,4, RadGraph 34,6 (findings) |
| RaDialog (arXiv 2311.18681) | thêm indication | BLEU-4 9,5 → 14,8, ROUGE-L 27,1 → 31,6 |
| RaDialog | danh sách bệnh có cấu trúc | CE 26,1 → 39,4; BLEU-4 9,0 → 9,5 |
| RaDialog | LLM 7B → 13B → 33B | không đổi (B-4 9,5 / 9,5 / 9,5) |
| PromptMRG (AAAI 2024) | prompt chẩn đoán | CE F1 0,370 → 0,444; BLEU-4 0,116 → 0,106 |
| MedGemma (arXiv 2507.05201) | MedGemma 4B PT, không fine-tune | RadGraph F1 29,5 (F+I); fine-tune ảnh + indication: 30,3 |
| DeCo (arXiv 2405.20985) | Q-Former vs pooling | nén bằng query gây "thiếu hụt ngữ nghĩa thị giác"; pooling 2D đơn giản tốt hơn |
| ViT Registers (arXiv 2309.16588) | token norm cao | hút attention ("attention sink"), cần token đệm |

Mức của dự án hiện tại (beam): BLEU-4 0,098, ROUGE-L 0,265 — ngang RaDialog-align
(9,5 / 27,1), một hệ Q-Former 32 query gần giống META-CXR. Tức là **chỉ sửa Q-Former
khó kéo NLG vượt mức này**; các đòn bẩy lớn trong tài liệu là bối cảnh đầu vào (lần
chụp trước, indication, lateral) và số token thị giác.

## 5. Đề xuất (chờ user duyệt — chưa sửa code, chưa train)

_(thí nghiệm đang chạy trên máy train — mục này được cập nhật khi có kết quả)_

## 6. Giới hạn

- CE lexicon là xấp xỉ bằng luật, chỉ dùng để so tương đối; CE thật cần CheXbert
  (`f1chexbert`, thư viện mà MLRG/nhiều bài dùng) — chưa cài.
- Mốc truy hồi dùng 60.000 study train, không phải toàn bộ.
- Một checkpoint mỗi phase, một seed.
