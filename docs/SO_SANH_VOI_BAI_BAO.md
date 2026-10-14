# So sánh pipeline và kết quả với bài báo META-CXR

Cập nhật 2026-10-14. Bài báo gốc: D. Edirisinghe et al., "Chest X-Ray Report Generation
Using Abnormality Guided Vision Language Model", IEEE Access, vol. 13, 2025.

File này trả lời bốn câu hỏi:

1. Pipeline của dự án khác bài báo ở đâu, và vì sao khác (mục 2).
2. Phần nào là tính mới về pipeline so với bài báo (mục 3).
3. Chấm theo đúng các phép đo của bài báo thì kết quả ra sao (mục 4).
4. Vì sao kết quả chênh với bài báo (mục 5).

Số của bài báo lấy từ văn bản PDF (mục III–V). Các bảng trong PDF là ảnh, nên chỉ những
số được nhắc trong phần chữ mới trích được; chỗ nào không trích được thì ghi "không trích
được". Số của dự án lấy từ artifact trên máy train, nguồn ở mục 8. File chỉ chứa số tổng
hợp, không có văn bản báo cáo hay ID bệnh nhân.

Chỉ mục chi tiết từng hạng mục (TM-01 … TM-29): [`CHI_MUC_TINH_MOI.md`](CHI_MUC_TINH_MOI.md).
Chẩn đoán và thí nghiệm: [`TONG_HOP_2026-10-11.md`](TONG_HOP_2026-10-11.md).

## 0. Tóm tắt

- **Mô hình được báo cáo:** Stage 1 là `run_20261005_paper` (ba pha, tham số MHCAC theo
  bài báo). Stage 2 là MedGemma 1.5 4B QLoRA, prompt có thêm indication và FINDINGS của
  lần chụp trước, train 2 epoch (xong 2026-10-14).
- **Phân loại (Stage 1), giao thức argmax của bài báo:** weighted P/R/F1 0,839 / 0,781 /
  0,780 so với bài báo 0,87 / 0,78 / 0,73; F1 5 bệnh 0,669 so với 0,701. Với quy tắc hai
  ngưỡng của dự án (không phải giao thức bài báo): 0,827 / 0,834 / 0,820 và 0,749.
- **Sinh báo cáo (Stage 2), 2.800 study test, cách tính của bài báo:**

  | | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore | CE macro F1 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Bài báo | 0,390 | 0,102 | 0,173 | 0,280 | 0,291 | 0,426 | 0,428 |
  | Dự án, cùng đầu vào như bài báo (v3, beam) | 0,357 | 0,098 | 0,137 | 0,265 | 0,128 | 0,369 | 0,290 |
  | Dự án, thêm bối cảnh (beam) | 0,434 | 0,190 | 0,190 | 0,355 | 0,416 | 0,466 | 0,425 |

- **Cách đọc bảng trên:** khi đầu vào giống bài báo (chỉ soft token và danh sách P/N/U),
  dự án thấp hơn bài báo ở mọi chỉ số. Khi thêm indication và báo cáo lần chụp trước, dự
  án cao hơn bài báo ở mọi chỉ số NLG, nhưng đó **không còn là cùng điều kiện**. Phần hơn
  đến từ bối cảnh văn bản, không chứng minh kênh ảnh tốt hơn.
- **Chưa đo được:** RadGraph F1, RadCliQ, Bảng 4 (CheXpert val), IU-Xray, và các bảng
  ablation 5, 6, 7 của bài báo.

## 1. Điều kiện so sánh

| | Bài báo | Dự án |
|---|---|---|
| Tập test | MIMIC-CXR-JPG, 3.269 (bài báo gọi là "radiographs") | cùng tập chia chính thức, 3.269 study |
| Số study chấm Stage 1 | 3.269 | 3.269 |
| Số study chấm Stage 2 | 3.269 (theo mô tả) | 2.800 (study có FINDINGS hợp lệ sau khi làm sạch target) |
| Số lần chạy | không nêu | một run, một seed cho mỗi cấu hình |
| Khoảng tin cậy | không có | bootstrap ghép cặp theo study cho mọi so sánh nội bộ |
| Phần cứng | RTX 4070 16 GB | RTX 5060 Ti 16 GB |

Số của bài báo không có khoảng tin cậy, nên mọi chênh lệch "dự án − bài báo" trong file
này là chênh lệch giữa hai điểm ước lượng, không kiểm định được.

## 2. Khác biệt pipeline và lý do

Mỗi dòng ghi: bài báo làm gì, dự án làm gì, vì sao khác, và ảnh hưởng đã đo (nếu có).

### 2.1 Dữ liệu và target

| Hạng mục | Bài báo | Dự án | Vì sao khác | Ảnh hưởng đo được |
|---|---|---|---|---|
| Góc chụp | chỉ ảnh frontal | một ảnh neo (PA > AP > lateral) và tối đa 1 ảnh phụ | muốn dùng thông tin lateral, 55% study train có hơn một ảnh | chưa có ablation đa góc so với chỉ frontal; tăng lên 3 ảnh phụ làm AUROC −0,0085 |
| Đơn vị mẫu | không nêu | một dòng mỗi study | lấy mẫu theo ảnh lặp báo cáo 145.077 lần và đếm nặng study nhiều ảnh | không tách riêng |
| Ô nhãn trống | coi là âm tính | coi là âm tính; study không có thông tin CheXpert nào bị loại khỏi loss phân loại | tránh biến study không nhãn thành 14 ô âm tính | 2.379 study train bị loại |
| Target FINDINGS | không mô tả cách lấy | parser riêng, bản v3 bỏ dòng tiêu đề phiếu chụp | 9,8% target train bắt đầu bằng dòng tiêu đề, mô hình học lại và 57% báo cáo test chỉ là dòng đó | BLEU-1 0,056 → 0,285 (greedy; hai run khác cohort) |
| Bối cảnh lâm sàng | không dùng | indication/history và FINDINGS của lần chụp trước đưa vào prompt | 90,0% FINDINGS test có ngôn ngữ so sánh thời gian; 89,6% study test có lần chụp trước | xem mục 4.2 và 5.2 |

### 2.2 Encoder và đặc trưng

| Hạng mục | Bài báo | Dự án | Vì sao khác | Ảnh hưởng đo được |
|---|---|---|---|---|
| Ba encoder | ResNet50 (BioViL-T), ViT (PubMedCLIP), Swin | cùng ba họ; Swin là MedCLIP Swin-Tiny | giống bài báo | — |
| Token PubMedCLIP | không nêu chi tiết | đọc qua `post_layernorm`, trừ trung bình theo ảnh, giữ CLS, tiền xử lý CLIP riêng | token thô gần như trùng nhau (cosine giữa các patch 0,674) nên attention phẳng | cosine giảm còn −0,014; đóng góp tới phân loại chưa tách riêng |
| Số token | 296 (theo mô tả) | 196 + 50 + 50 = 296 | giống bài báo | — |
| StreamAdapter, ViewFusion, loss MPC, view consistency | không có | có | phục vụ đa góc chụp; ý tưởng MPC từ EVOKE | chưa tách riêng; bật hai cờ kiểu EVOKE làm F1 5 bệnh −0,0160 |
| Che ngẫu nhiên 10% đặc trưng | có | có | giống bài báo | — |

Hai điểm **giống code gốc của tác giả** nhưng là lỗi, dự án giữ nguyên vì sửa không có lợi
trong pilot: token BioViL bị `reshape` không hoán vị trục (196 "token" không phải 196 ô
ảnh), và projector 1408 chiều của BioViL khởi tạo ngẫu nhiên, chỉ được train ở pha 1c.

### 2.3 META-Former (Q-Former)

| Hạng mục | Bài báo | Dự án | Vì sao khác | Ảnh hưởng đo được |
|---|---|---|---|---|
| Mục tiêu căn chỉnh | ITC + ITM + ITG của BLIP-2 | cùng ba mục tiêu; ITC dùng loss sigmoid kiểu SigLIP | trên GPU 16 GB, ITC softmax ở batch 8 cho kết quả ngẫu nhiên trong cả bốn lần đo | lần đầu ITC vượt ngẫu nhiên: R@5 0,664 / 0,641 |
| Batch căn chỉnh | không nêu | 128 bằng GradCache trên đặc trưng encoder đã cache | cần batch lớn cho loss đối chiếu mà không vượt 16 GB | như dòng trên |
| Tiết kiệm bộ nhớ | không nêu | gradient checkpointing Q-Former, bf16 | giới hạn 16 GB | không đổi kết quả (có test so khớp gradient) |

Code công khai của tác giả comment toàn bộ khối ITC/ITM/ITG. Dự án train khối này theo
mô tả trong bài báo, không theo code.

### 2.4 MHCAC và loss phân loại

| Hạng mục | Bài báo | Dự án | Vì sao khác |
|---|---|---|---|
| Số tầng, expert token | 6 tầng, 8 common + 14 chuyên biệt | giống | — |
| Dropout, label smoothing | 0,2; không smoothing | giống | — |
| Loss | CE có trọng số + contrastive + orthogonality + sparsity | giống; trọng số lớp tính lại theo tần suất train của dự án | bài báo không công bố bảng trọng số |
| Ba lớp P/N/U | có | có | giống bài báo |
| Hướng dẫn bằng văn bản lúc train | expert token cross-attend embedding văn bản có dropout, chủ yếu ở các tầng đầu; lúc suy luận không có văn bản | giống: một đường, self-attention trước, mask theo phần tử (bản teacher/student cũ đã gỡ) | làm theo mô tả trong bài báo |

### 2.5 Lịch huấn luyện Stage 1

| Hạng mục | Bài báo | Dự án | Vì sao khác | Ảnh hưởng đo được |
|---|---|---|---|---|
| Ba pha | căn chỉnh → chuyển sang MHCAC → train chung | giống (1a, 1b, 1c) | — | — |
| Độ dài | 20.000 bước biểu diễn | theo epoch: 1a 4 epoch, 1b 5 epoch, 1c 1 epoch | bài báo không nêu batch nên "20.000 bước" không quy đổi được | 1b đạt đỉnh ở epoch thứ ba; 1c không tăng AUROC test (+0,0037 [−0,0012; +0,0095]) |
| LR pha 1b | 5e-5 → 2e-4 → 1e-5 | giống | — | — |
| Pha 1c mở khóa | projection head của encoder | thêm ResNet50 layer4 và hai khối cuối của CLIP | thử cho encoder thích nghi | bằng chứng cũ đo trong khung nhị phân đã gỡ, không dùng được |
| Chọn checkpoint | không nêu | macro recall ba lớp trên val | loss bị lớp đa số chi phối | không tách riêng |

### 2.6 Quy tắc quyết định của Stage 1

| Hạng mục | Bài báo | Dự án | Vì sao khác |
|---|---|---|---|
| Số báo cáo phân loại | argmax ba lớp | **số chính:** hai ngưỡng theo từng bệnh trên `p_pos/(p_pos+p_neg)`, fit trên val; argmax luôn in kèm | argmax hầu như không bao giờ chọn Uncertain và bỏ sót dương tính; hai ngưỡng vẫn là ba lớp |
| Ngưỡng cho prompt | Eq. 22 (điểm gần góc trái trên của ROC), không nêu fit trên tập nào | v3: Eq. 22 fit trên val; mô hình mới: hai ngưỡng của Stage 1 | bài báo không công bố file ngưỡng; Eq. 22 fit trên val làm weighted F1 còn 0,702 |

### 2.7 Stage 2

| Hạng mục | Bài báo | Dự án | Vì sao khác | Ảnh hưởng đo được |
|---|---|---|---|---|
| LLM | Vicuna-7B, INT8, LoRA khởi tạo từ RaDialog (đã fine-tune trên MIMIC-CXR) | MedGemma 1.5 4B IT, 4-bit NF4, QLoRA train từ đầu | phần cứng 16 GB và giấy phép; quyết định của user | chưa có ablation đổi LLM |
| LoRA | r 8, α 16 | giống | — | r 16 không tốt hơn trong pilot |
| Ảnh vào LLM | 32 token META-Former qua projection | giống (không dùng ảnh gốc của MedGemma) | giữ thiết kế bài báo | — |
| Prompt | query + danh sách P/N/U | v3: nguyên văn prompt bài báo. Mô hình mới: thêm dòng `Indication:` và `Previous study findings (N days earlier):` | thiếu bối cảnh là nguyên nhân có bằng chứng mạnh nhất (mục 5.2) | pilot: ROUGE-L +0,036 [+0,023; +0,049], BERTScore +0,028 [+0,014; +0,041] |
| Cue lúc train | dự đoán của MHCAC | 80% lấy từ nhãn CheXpert thật, 20% giữ dự đoán MHCAC; val/test luôn dùng MHCAC | v3 bỏ qua cue: đổi sang cue từ nhãn thật không đổi chỉ số nào | pilot: CheXbert micro F1-5 +0,053 [+0,016; +0,089]; CE lexicon chưa đạt ngưỡng |
| Độ dài train | 5.000 bước | v3: 1 epoch, LR 1e-4. Mô hình mới: 2 epoch, LR 3e-4 | v3 còn giảm loss ở cuối epoch; LR 3e-4 đạt quy tắc trong pilot | val loss 0,99053 → 0,95619 |
| Giải mã | không nêu | beam 4, length penalty 2 (greedy báo kèm) | greedy viết ngắn hơn tham chiếu | v3: BERTScore +0,0083, ROUGE-L +0,0150, CIDEr +0,0439 so với greedy |
| Stop token | không nêu | dừng ở `<end_of_turn>` | sửa lỗi của chính dự án (78% token sinh sau điểm kết thúc) | không phải khác biệt với bài báo |

### 2.8 Đánh giá

| Hạng mục | Bài báo | Dự án | Vì sao khác |
|---|---|---|---|
| NLG | BLEU, METEOR, ROUGE-L, CIDEr | cùng bộ chỉ số, `pycocoevalcap`, chữ thường, tách từ nltk | bài báo không nêu công cụ; đây là cách phổ biến |
| BERTScore | 0,426, không nêu cấu hình | `distilroberta-base`, rescale theo baseline | cấu hình của CXR-Report-Metric mà bài báo trích; BERTScore thô (~0,8) là thang khác |
| CE | CheXpert labeler, macro P/R/F1 trên 14 bệnh | CheXbert (`f1chexbert`), dương tính hoặc uncertain = 1 | CheXpert labeler cần Java và không có trong repo; CheXbert là chuẩn của các bài gần đây |
| RadGraph, RadCliQ | có | chưa cài | mã nghiên cứu có giấy phép riêng; báo "chưa có", không báo 0 |
| Khoảng tin cậy | không | bootstrap ghép cặp | để biết chênh lệch có vượt nhiễu không |

## 3. Tính mới về pipeline so với bài báo

Chỉ liệt kê thay đổi **đang nằm trong pipeline được báo cáo**. Mức bằng chứng theo
[`CHI_MUC_TINH_MOI.md`](CHI_MUC_TINH_MOI.md): A = có số đo cho thấy tốt hơn; B = khác/bổ
sung, chưa tách riêng đóng góp.

| # | Tính mới | Mức | Bằng chứng |
|---|---|---|---|
| 1 | Quy tắc hai ngưỡng theo bệnh cho quyết định ba lớp (fit trên val) | A | F1 5 bệnh +0,0805 [+0,0743; +0,0866] so với argmax; macro recall −0,0149 |
| 2 | GradCache + loss sigmoid cho pha căn chỉnh 1a trên GPU 16 GB | A | ITC từ ngẫu nhiên (bốn lần đo) lên R@5 0,664 / 0,641 |
| 3 | Làm sạch target FINDINGS (bỏ dòng tiêu đề phiếu chụp) | A | báo cáo chỉ-tiêu-đề 57% → 0% |
| 4 | Giải mã beam 4 + length penalty 2, lặp soft token theo từng beam | A | CIDEr +0,0439 [+0,0230; +0,0653] so với greedy (v3) |
| 5 | Bối cảnh lâm sàng trong prompt: indication và FINDINGS lần chụp trước | B | pilot đạt quy tắc; run đầy đủ hơn v3 ở mọi chỉ số (mục 4.2) |
| 6 | Train Stage 2 với cue từ nhãn thật pha 20% cue mô hình | B | pilot: CheXbert micro F1-5 +0,053; không tách riêng ở run đầy đủ |
| 7 | Đa góc chụp: ViewFusion, StreamAdapter, MPC, view consistency | B | chưa có ablation so với chỉ frontal |
| 8 | Luồng PubMedCLIP: `post_layernorm`, trừ trung bình token, tiền xử lý riêng | B | cosine giữa patch 0,674 → −0,014 |
| 9 | Đánh giá: bootstrap ghép cặp, mốc so sánh không học, CE bằng CheXbert | B | thước đo, không phải thay đổi mô hình |

Bài báo tự nêu ở phần hướng phát triển việc "kết hợp ảnh với dữ liệu lâm sàng và thông tin
văn bản của bệnh nhân" và "mở rộng sang đa góc chụp và ảnh theo thời gian". Mục 5 và 7 là
hai hướng đó được làm thật: mục 5 dùng **văn bản** của lần chụp trước (không dùng ảnh lần
trước), mục 7 dùng ảnh phụ của cùng lần chụp.

Mục 5 và 6 đi cùng LR 3e-4 và 2 epoch trong một run duy nhất, nên **chưa tách được đóng
góp của từng thay đổi** ở chế độ train đầy đủ. Pilot (10.000 study) là bằng chứng tách
riêng duy nhất.

Phát hiện từ phân tích (không phải thay đổi mô hình, nhưng bài báo không có):

- 32 soft token của META-Former sụp thành một vector ngay ở pha 1a (cosine giữa các
  token của một study 0,9989). Cả 32 query cùng đọc token CLS của PubMedCLIP.
- Soft token giữ thông tin bệnh phổ biến, mất bệnh khu trú/hiếm (Fracture ở mức ngẫu nhiên).
- Mô hình Stage 2 kiểu bài báo (v3) bỏ qua cue: thay cue bằng nhãn thật không đổi chỉ số nào.
- Chỉ chép FINDINGS của lần chụp trước, không học gì, đã ngang bài báo ở BLEU-1 / BLEU-4.

Đã thử và không cải thiện (không nằm trong pipeline): cờ kiểu EVOKE, tối đa 3 ảnh phụ,
cue Stage 2 theo hai ngưỡng trên adapter cũ, 13 finding token học được, logit-adjusted
loss, sửa bố cục token BioViL, LoRA r 16, record train từ ảnh sạch.

## 4. Kết quả theo các phép đo của bài báo

### 4.1 Phân loại bất thường (bài báo: Hình 10, Bảng 5/7, Hình 5)

Test 3.269 study, 14 bệnh, ba lớp. Precision/recall/F1 có trọng số theo từng bệnh rồi
lấy trung bình 14 bệnh, `zero_division=1`.

| | Bài báo | Dự án, argmax (giao thức bài báo) | Dự án, hai ngưỡng (số chính, không phải giao thức bài báo) |
|---|---:|---:|---:|
| Weighted precision | 0,87 | 0,839 | 0,827 [0,823; 0,832] |
| Weighted recall | 0,78 | 0,781 | 0,834 [0,830; 0,838] |
| Weighted F1 | 0,73 | 0,780 | 0,820 [0,815; 0,824] |
| F1 trung bình 5 bệnh | 0,701 | 0,669 | 0,749 [0,741; 0,757] |
| Macro recall ba lớp | không báo | 0,461 | 0,446 |

AUROC một-lớp-với-phần-còn-lại, trung bình 14 bệnh: Positive 0,784, Negative 0,778,
Uncertain 0,663. Bài báo chỉ nêu vài giá trị trong phần chữ (Hình 5):

| Bệnh, lớp | Bài báo | Dự án |
|---|---:|---:|
| Support Devices, Positive | 0,92 | 0,904 |
| Support Devices, Negative | 0,84 | 0,903 |
| Pleural Effusion, Positive | 0,74 | 0,893 |
| Pleural Effusion, Negative | 0,85 | 0,893 |
| Edema, Negative | 0,78 | 0,838 |

Năm bệnh dùng cho F1-5, quy tắc hai ngưỡng (weighted F1 / AUROC Positive): Atelectasis
0,692 / 0,758; Cardiomegaly 0,691 / 0,770; Consolidation 0,857 / 0,760; Edema 0,724 /
0,841; Pleural Effusion 0,783 / 0,893. Bài báo không cho số theo từng bệnh trên MIMIC-CXR.

### 4.2 Chỉ số NLG (bài báo: Bảng 2)

Test 2.800 study, cùng cách tính cho mọi dòng của dự án.

| | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | số từ (trung vị) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bài báo | 0,390 | không trích được | không trích được | 0,102 | 0,173 | 0,280 | 0,291 | — |
| v3 beam (đầu vào như bài báo) | 0,357 | 0,213 | 0,140 | 0,098 | 0,137 | 0,265 | 0,128 | 47 |
| v3 greedy | 0,285 | — | — | 0,073 | 0,120 | 0,250 | 0,084 | 37 |
| Mới, beam 4 (thêm bối cảnh) | 0,434 | 0,308 | 0,237 | 0,190 | 0,190 | 0,355 | 0,416 | 47 |
| Mới, greedy (thêm bối cảnh) | 0,376 | 0,271 | 0,209 | 0,168 | 0,177 | 0,354 | 0,375 | 39 |

Chênh lệch so với bài báo (điểm ước lượng):

| | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|---:|
| v3 beam − bài báo | −0,033 | −0,004 | −0,036 | −0,015 | −0,163 |
| Mới beam − bài báo | +0,044 | +0,088 | +0,017 | +0,075 | +0,125 |

Chênh lệch nội bộ, ghép cặp trên cùng 2.800 study (mới beam − v3 beam): BLEU-4 +0,095
[+0,089; +0,101], ROUGE-L +0,090 [+0,085; +0,096], CIDEr +0,288 [+0,256; +0,320].

### 4.3 Chỉ số lâm sàng (bài báo: Bảng 3)

| | BERTScore | CE precision | CE recall | CE macro F1 | RadGraph F1 | RadCliQ ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Bài báo (CheXpert labeler) | 0,426 | 0,411 | 0,465 | 0,428 | 0,224 | 1,211 |
| v3 beam (CheXbert) | 0,369 | 0,397 | 0,284 | 0,290 | chưa có | chưa có |
| Mới, beam 4 (CheXbert) | 0,466 | 0,517 | 0,398 | 0,425 | chưa có | chưa có |
| Mới, greedy (CheXbert) | 0,474 | 0,520 | 0,395 | 0,424 | chưa có | chưa có |

CE của dự án là macro trên 14 bệnh, 2.800 study. Micro F1-14: v3 0,460; mới beam 0,564;
mới greedy 0,563. Trên 5 bệnh phổ biến (mới beam): macro F1 0,551, micro F1 0,615. Ghép
cặp mới beam − v3 beam: macro F1-14 +0,136 [+0,121; +0,151], micro F1-14 +0,104
[+0,092; +0,116].

Số CE của bài báo và của dự án dùng **hai labeler khác nhau** (CheXpert labeler theo
luật so với CheXbert), và bài báo không nêu nhãn tham chiếu lấy từ labeler chạy lại hay
từ file CheXpert của MIMIC. Vì vậy 0,425 so với 0,428 chỉ là so sánh gần đúng.

### 4.4 Các bảng của bài báo chưa tái lập

| Bảng/hình của bài báo | Số bài báo | Trạng thái ở dự án |
|---|---|---|
| Bảng 2, IU-Xray (ngoài phân phối) | BLEU-4 0,121; ROUGE-L 0,287 | chưa chạy; chưa có loader IU-Xray cho pipeline hiện tại |
| Bảng 4, CheXpert val 200 ảnh, `p1/(p0+p1)` | F1 trung bình 0,699 | có phần chấm điểm, chưa có loader CheXpert val cho Stage 1 |
| Bảng 5, F1-5 theo từng encoder | bảng là ảnh, không trích được | chưa chạy lại trên checkpoint hiện tại |
| Bảng 6, BERTScore theo từng encoder | không trích được | chưa chạy |
| Bảng 7, ablation loss của MHCAC | F1-5 đầy đủ 0,701 | chưa chạy (mỗi dòng là một lần train Stage 1) |
| Hình 9, LLM gốc / fine-tune / instruction-tune | hình | không áp dụng trực tiếp (LLM khác) |
| Hình 10, số common expert token | 8 token: 0,87 / 0,78 / 0,73 | dùng 8 theo bài báo, không quét lại |
| Bảng 9, chi phí suy luận | ~3,5 giây, 14,9 GB | bước sinh: 3,7 giây/study (beam), 2,4 giây (greedy); đỉnh VRAM 8,9 GiB. Đo riêng bước sinh với record Stage 1 lấy từ cache, nên không cùng phạm vi đo |
| Hình 12, chuyên gia chấm | 86,75–88,75% | không có đánh giá của bác sĩ |

## 5. Vì sao kết quả chênh với bài báo

### 5.1 Phân loại

**Weighted F1 cao hơn bài báo (0,780 so với 0,73) nhưng F1 5 bệnh thấp hơn (0,669 so với
0,701), cùng là argmax.** Hai số này đo hai thứ khác nhau:

- Weighted F1 lấy trung bình 14 bệnh và có trọng số theo tần suất lớp. Ở các bệnh hiếm,
  lớp Negative chiếm gần hết (Fracture, Pleural Other, Lung Lesion có weighted F1 0,93–
  0,96 trong khi F1 lớp Positive là 0,00–0,14). Số này vì vậy chủ yếu phản ánh việc đoán
  đúng lớp đa số. Bài báo có precision 0,87 và recall 0,78 nhưng F1 chỉ 0,73, tức mô hình
  của họ lệch khỏi lớp đa số nhiều hơn; dự án không biết nguyên nhân vì bài báo không
  công bố trọng số lớp hay dự đoán.
- F1 5 bệnh là nơi lớp Positive và Uncertain có nhiều mẫu, nên phản ánh khả năng phân
  biệt thật. Ở đây argmax của dự án thấp hơn 0,032.

Các nguyên nhân có bằng chứng tại chỗ cho phần thấp hơn ở 5 bệnh:

1. **Điểm làm việc của argmax.** AUROC không đổi khi chuyển sang hai ngưỡng, nhưng F1 5
   bệnh tăng 0,0805. Tức khả năng xếp hạng đã có, argmax chỉ chọn điểm cắt kém: nó gần
   như không bao giờ chọn Uncertain và bỏ sót dương tính.
2. **Encoder gần như đóng băng, BioViL đọc qua phép chiếu ngẫu nhiên ở 1a/1b.** MHCAC chỉ
   hơn một probe tuyến tính trên một token MedCLIP-Swin 0,012 AUROC (0,745 so với 0,734).
3. **Nhiễu nhãn.** So với nhãn bác sĩ (685 study test), precision dương tính của nhãn tự
   động chỉ 0,42–0,46 ở bốn bệnh; cùng mô hình chấm bằng nhãn bác sĩ có AUROC Positive cao
   hơn 0,022 [+0,008; +0,036]. Trần này áp cho cả bài báo, nên nó giải thích mức tuyệt đối
   thấp chứ không giải thích khoảng cách với bài báo.
4. **Lớp Uncertain yếu** (AUROC 0,663): rất ít mẫu, nhãn nhiễu nhất.

Những điều **không biết** và không nên suy diễn: batch và số bước thật của bài báo, bảng
trọng số lớp, ngưỡng Eq. 22 fit trên tập nào (đường ROC duy nhất của họ vẽ trên test),
và code công khai của họ không train khối căn chỉnh như bài báo mô tả.

**Với hai ngưỡng, dự án cao hơn bài báo ở F1 5 bệnh (0,749 so với 0,701).** Đây không phải
giao thức của bài báo: ngưỡng được fit trên val để tối đa weighted F1. Chỉ được viết là
"quy tắc quyết định khác cho kết quả tốt hơn trên cùng mô hình", không phải "mô hình phân
loại tốt hơn bài báo". Cái giá: macro recall giảm 0,0149.

### 5.2 Sinh báo cáo, khi đầu vào giống bài báo (v3 thấp hơn bài báo)

Xếp theo độ mạnh của bằng chứng:

1. **Thiếu nội dung, không phải sai văn phong.** BERTScore precision 0,424 (ngang bài
   báo) nhưng recall 0,299; mô hình chỉ nhắc 32–34% số bệnh bác sĩ có bàn tới; không bao
   giờ nhắc Fracture, Lung Lesion, Pleural Other, Enlarged Cardiomediastinum. CIDEr chịu
   ảnh hưởng nặng nhất (−0,163) vì nó thưởng n-gram hiếm, tức đúng phần nội dung bị thiếu.
2. **Kênh ảnh chỉ còn một vector.** 32 soft token sụp (cosine 0,9997 ở checkpoint v3 dùng),
   giữ được bệnh phổ biến, mất bệnh khu trú. Hoán đổi soft token giữa các study làm
   BERTScore −0,072, nên kênh này vẫn được dùng, nhưng nghèo thông tin.
3. **LLM không dùng cue.** Thay cue Eq. 22 bằng cue từ nhãn thật ở 977/1.000 study: không
   chỉ số nào đổi có ý nghĩa, 153 báo cáo giống hệt. Lợi ích "abnormality-guided" của bài
   báo vì thế không xuất hiện ở v3.
4. **LLM và khởi tạo khác.** Bài báo dùng Vicuna-7B với LoRA lấy từ RaDialog, đã fine-tune
   trên MIMIC-CXR, rồi instruction-tune tiếp. Dự án train QLoRA cho MedGemma 4B từ đầu,
   1 epoch, loss còn giảm ở cuối. Chưa có ablation tách riêng yếu tố này.
5. **Tập chấm khác.** Dự án chấm 2.800 study có FINDINGS hợp lệ; bài báo ghi 3.269. Không
   biết bài báo xử lý các study không có FINDINGS thế nào.
6. **Thước đo thưởng văn phong.** Một báo cáo cố định cho mọi study đạt ROUGE-L 0,238 và
   BERTScore 0,340. Khoảng 0,015 ROUGE-L giữa v3 và bài báo nằm trong vùng mà khác biệt
   văn phong giữa hai LLM đã đủ gây ra.

BLEU-4 gần như ngang (0,098 so với 0,102) trong khi METEOR và CIDEr thấp rõ: beam kéo
dài báo cáo về đúng độ dài (47 từ) nên khớp được cụm từ khuôn mẫu, nhưng không thêm nội
dung.

### 5.3 Sinh báo cáo, khi thêm bối cảnh (mô hình mới cao hơn bài báo)

Phần hơn bài báo đến từ **thông tin đầu vào mà bài báo không có**, không phải từ kiến trúc:

1. **Báo cáo lần chụp trước mang phần lớn nội dung.** Chép nguyên FINDINGS lần trước (không
   học gì), study nào không có thì dùng v3, đã đạt BLEU-1 0,391 / BLEU-4 0,106, ngang bài
   báo. Trên 2.508 study có lần trước, bản chép hơn v3 về CE lexicon 0,356 so với 0,236.
   Tập test MIMIC nghiêng về bệnh nhân chụp nhiều lần: 89,6% study test có lần trước, train
   chỉ 64,1%.
2. **Báo cáo gốc viết theo kiểu so sánh.** 90,0% FINDINGS test có ngôn ngữ so sánh thời
   gian. Không có lần trước, mô hình phải bịa câu so sánh (v3 làm vậy ở 51% báo cáo mà bản
   gốc không so sánh).
3. **Indication cho biết bác sĩ đang tìm gì.** Tài liệu: RaDialog thêm indication BLEU-4
   9,5 → 14,8; MAIRA-2 bỏ lần chụp trước lúc suy luận ROUGE-L −28,9%.
4. **Bằng chứng tách riêng trong pilot** (10.000 study, 500 study val): chỉ thêm bối cảnh,
   ROUGE-L +0,036 [+0,023; +0,049], BERTScore +0,028 [+0,014; +0,041], CIDEr +0,20.
5. **Train dài hơn và LR cao hơn** (2 epoch, 3e-4 so với 1 epoch, 1e-4): val loss epoch 2
   thấp hơn epoch 1 0,034. Đóng góp tới chất lượng sinh chưa tách riêng ở run đầy đủ.

Kênh ảnh **không thay đổi** giữa v3 và mô hình mới: cùng checkpoint Stage 1, soft token
vẫn sụp. Vì vậy không được viết "META-Former/MHCAC của dự án tốt hơn bài báo" dựa trên
bảng NLG.

### 5.4 Clinical Efficacy

- **Macro F1 ngang bài báo (0,425 so với 0,428) nhưng cấu trúc khác:** precision cao hơn
  (0,517 so với 0,411), recall thấp hơn (0,398 so với 0,465). Mô hình của dự án nói ít
  bệnh hơn nhưng đúng hơn khi nói; đây là cùng hiện tượng viết thiếu ở mục 5.2.
- **v3 thấp hơn nhiều (0,290):** recall 0,284, do bỏ qua cue và không nhắc các bệnh hiếm.
- **Mô hình mới tăng 0,136 macro F1 so với v3:** bối cảnh lần trước mang sẵn danh sách
  bệnh của bệnh nhân, và train với cue từ nhãn thật dạy LLM đưa cue vào văn bản (pilot:
  CheXbert micro F1-5 +0,053 [+0,016; +0,089]).
- Hai labeler khác nhau nên chênh 0,003 với bài báo không có ý nghĩa theo hướng nào.

### 5.5 Tóm tắt nguyên nhân theo từng chênh lệch

| Chênh lệch | Hướng | Nguyên nhân chính | Mức bằng chứng |
|---|---|---|---|
| Weighted F1 argmax 0,780 so với 0,73 | cao hơn | chỉ số bị lớp đa số chi phối; không biết cấu hình của bài báo | không giải thích được chắc chắn |
| F1 5 bệnh argmax 0,669 so với 0,701 | thấp hơn | điểm cắt của argmax; encoder đóng băng | đo trực tiếp (hai ngưỡng +0,0805 trên cùng mô hình) |
| F1 5 bệnh hai ngưỡng 0,749 so với 0,701 | cao hơn | quy tắc quyết định khác, fit trên val | đo trực tiếp; không cùng giao thức |
| NLG v3 thấp hơn bài báo | thấp hơn | viết thiếu nội dung; soft token sụp; LLM bỏ qua cue; LLM và khởi tạo khác | ba ý đầu đo trực tiếp; ý cuối chưa tách |
| NLG mô hình mới cao hơn bài báo | cao hơn | thêm indication và báo cáo lần chụp trước | pilot tách riêng; mốc chép lần trước |
| CE macro F1 0,425 so với 0,428 | ngang | precision cao hơn, recall thấp hơn; labeler khác | gần đúng |

## 6. Cách viết trong luận văn

- Được viết là **cải thiện có bằng chứng**: quy tắc hai ngưỡng (kèm câu "không phải giao
  thức argmax của bài báo"), GradCache + SigLIP cho 1a, làm sạch target, beam.
- Được viết là **mở rộng đầu vào**: bối cảnh lâm sàng trong prompt. Bảng so với bài báo
  phải có hai dòng của dự án (cùng đầu vào, và thêm bối cảnh) và ghi rõ dòng nào cùng điều
  kiện.
- Không được viết: "vượt bài báo ở sinh báo cáo" mà không kèm điều kiện; "mô hình phân
  loại tốt hơn bài báo"; bất kỳ số RadGraph, RadCliQ, IU-Xray hay CheXpert val nào.

## 7. Giới hạn

- Một run, một seed cho mỗi cấu hình; bài báo cũng không nêu số lần chạy.
- Ba thay đổi của mô hình Stage 2 mới (bối cảnh, cue lúc train, LR/epoch) chưa tách riêng
  ở chế độ train đầy đủ.
- Stage 2 chấm trên 2.800 study, bài báo ghi 3.269.
- CE dùng CheXbert, bài báo dùng CheXpert labeler.
- BERTScore của bài báo không nêu cấu hình; dự án giả định bản rescale `distilroberta-base`.
- Các bảng trong PDF là ảnh; BLEU-2, BLEU-3 và số theo từng bệnh của bài báo không trích được.
- Nhãn CheXpert tự động có nhiễu; nhãn này vừa dùng để train vừa dùng để chấm.

## 8. Nguồn số liệu

| Nội dung | Artifact trên máy train | Tài liệu trong repo |
|---|---|---|
| Stage 1, test | `~/eval_20261010_cutpoints/test/` | `handoff/PLAN-2026-10-10-cutpoint-headline.md` |
| File hai ngưỡng | — | `configs/stage1_cutpoints/run_20261005_paper.json` |
| Stage 2 v3 | `~/stage2_v3_test_beam_20261009/` | `handoff/PLAN-2026-10-07-stage2-header-fix.md` |
| Stage 2 mới, 2.800 study | `~/stage2_p2_20261011/score/n2800_*.json` | `handoff/PLAN-2026-10-11-stage2-context-pilot.md` |
| Pilot Stage 2 | `~/stage2_p2_20261011/` | cùng file kế hoạch |
| Chẩn đoán soft token, cue, mốc chép | `~/stage2_analysis_20261011/`, `~/diag_qformer_20261011b/` | `TONG_HOP_2026-10-11.md` |
| Nhiễu nhãn, probe encoder | `~/stage1_meas_20261011/` | `handoff/PLAN-2026-10-11-stage1-improvement.md` |

Script chấm: `scripts/evaluate_stage1.py` (`--cutpoints`),
`scripts/score_stage2_paper_protocol.py`, `scripts/score_ce_chexbert.py`.
