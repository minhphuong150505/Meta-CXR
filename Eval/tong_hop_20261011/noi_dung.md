---
title: TỔNG HỢP ĐỢT ĐIỀU TRA META-CXR NGÀY 10 VÀ 11 THÁNG 10 NĂM 2026
subtitle: Chẩn đoán Stage 1 và Stage 2, thí nghiệm, đối chiếu với bài báo gốc và kế hoạch huấn luyện
subtitle: Trạng thái tại 16 giờ 31 ngày 11 tháng 10 năm 2026, khi lần huấn luyện Stage 2 toàn bộ đang chạy
chapter_new_page: true
data: so_lieu.json
---
<!-- Nguồn: docs/TONG_HOP_2026-10-11.md, docs/CHI_MUC_TINH_MOI.md và ba file docs/handoff/PLAN-2026-10-11-*.md, mã 9b2963d. -->

::heading Tóm tắt
Tài liệu này gộp năm tài liệu của đợt điều tra ngày 10 và 11 tháng 10 năm 2026 thành một bản đọc liền mạch. Mọi con số được lấy từ năm tài liệu đó, vốn được ghi lại từ artifact trên máy huấn luyện; tài liệu không tính lại số nào và chỉ chứa số tổng hợp. Khoảng tin cậy là khoảng 95% từ bootstrap ghép cặp theo study.

Về chẩn đoán, có sáu kết luận. Thứ nhất, 32 soft token của META-Former sụp thành một vector ngay ở pha 1a, vì cả 32 query cùng đọc token CLS toàn cục của PubMedCLIP. Thứ hai, soft token giữ thông tin của các bệnh phổ biến và mất thông tin của các bệnh khu trú hoặc hiếm. Thứ ba, mô hình Stage 2 cũ, gọi là v3, bỏ qua kênh cue: đưa cue lấy từ nhãn thật vào cũng không làm báo cáo thay đổi. Thứ tư, báo cáo yếu chủ yếu vì thiếu bối cảnh lâm sàng: chỉ chép phần FINDINGS của lần chụp trước, không học gì, đã ngang bài báo ở BLEU-1 và BLEU-4. Thứ năm, luồng BioViL có hai lỗi kế thừa từ mã nguồn gốc của tác giả. Thứ sáu, nhãn CheXpert tự động khá nhiễu so với nhãn bác sĩ, với hệ số kappa từ 0,49 đến 0,93 tùy bệnh.

Về Stage 1, số chính dùng quy tắc hai ngưỡng theo từng bệnh, fit trên tập val: weighted F1 0,820 và F1 trung bình năm bệnh 0,749 trên 3.269 study test, so với 0,73 và 0,701 của bài báo. Đây không phải giao thức argmax của bài báo. Checkpoint run_20261005_paper được giữ nguyên; pha 1b rút còn 3 epoch và pha 1c bỏ khỏi mặc định.

Về Stage 2, prompt có bối cảnh, cue lấy từ nhãn thật lúc huấn luyện, giải mã beam và độ đo CE bằng CheXbert đã được thêm vào. Trong pilot trên 500 study val, bối cảnh tăng BLEU-4 từ 0,121 lên 0,165 và CIDEr từ 0,284 lên 0,487; cấu hình P2 đạt 0,178 và 0,518. Lần huấn luyện toàn bộ với cấu hình P2, 2 epoch, bắt đầu lúc 15 giờ 50 ngày 11 tháng 10 và chưa kết thúc, nên tài liệu này chưa có số test của mô hình mới. Mô hình mới nhận thêm indication và báo cáo lần chụp trước, vì vậy không cùng điều kiện đầu vào với bài báo.

# Phạm vi, nguồn và cách đo

## Năm tài liệu nguồn
::table Năm tài liệu được gộp và vị trí của chúng trong tài liệu này
::widths 6.5, 6.5, 3.5
::font 10.5
| Tài liệu nguồn | Nội dung | Chương |
|---|---|---|
| docs/TONG_HOP_2026-10-11.md | Chẩn đoán Stage 1 và Stage 2, tài liệu liên quan, đề xuất | 2, 3, 4, 7, 8 |
| docs/CHI_MUC_TINH_MOI.md | Chỉ mục khác biệt so với bài báo, TM-01 đến TM-29 | 6 |
| docs/handoff/PLAN-2026-10-11-soft-token-probe.md | Linear probe trên soft token | 2 |
| docs/handoff/PLAN-2026-10-11-stage1-improvement.md | Nghiên cứu cải thiện Stage 1, pilot sửa BioViL | 3 |
| docs/handoff/PLAN-2026-10-11-stage2-context-pilot.md | Pilot Stage 2 và lần huấn luyện toàn bộ | 5 |
::note Năm tài liệu nguồn vẫn là nơi ghi chi tiết. Khi có chỗ lệch, tài liệu nguồn và artifact trên máy huấn luyện là chuẩn.

Bài báo gốc là công trình META-CXR của Edirisinghe và cộng sự, đăng trên IEEE Access năm 2025 [1]. Các chi tiết của bài báo được đối chiếu với văn bản PDF, gồm các mục III đến V, phương trình 22, các bảng 2 đến 7 và các hình 10, 11.

## Trạng thái tại thời điểm viết
Stage 1 giữ checkpoint run_20261005_paper và không huấn luyện lại. Stage 2 đang huấn luyện toàn bộ với cấu hình P2 trong 2 epoch; lúc 16 giờ 31 tiến trình đang dựng record Stage 1 cho tập train, đạt 70.132 trên 222.758 study, chưa có lỗi và chưa có val loss. Lần chạy dự kiến kết thúc trong khoảng trưa đến chiều ngày 13 tháng 10. Một script chờ đã được đặt trên máy huấn luyện để tự chấm tập test sau khi lần chạy kết thúc. Mã nguồn mới nhất là commit 9b2963d trên nhánh feat/stage2-finding-tokens.

## Các phép đo
::table Các thí nghiệm của đợt điều tra, cỡ mẫu và công cụ
::widths 5.2, 6.3, 5
::font 9.5
| Thí nghiệm | Script | Cỡ mẫu |
|---|---|---|
| Sụp token theo pha | scripts/diagnose_qformer_collapse.py | 200 study val mỗi checkpoint |
| Probe soft token trên cache của v3 | scripts/probe_soft_tokens_cached.py | 50.000 train, 2.780 test |
| Probe cùng giao thức | scripts/probe_image_features.py | 53.300 fit, 5.922 giữ lại, 2.780 test |
| Đặc trưng từng encoder | scripts/embed_cxr_encoders.py | 53.300 fit, 2.780 test |
| Nhãn bác sĩ | scripts/analyze_radiologist_labels.py | 685 study test |
| Mốc truy hồi | scripts/retrieval_report_baselines.py | 60.000 train, 2.800 test |
| Embedding ảnh MedGemma | scripts/embed_medgemma_images.py | 62.800 ảnh |
| Mốc chép lần chụp trước | scripts/prior_report_baseline.py | 2.508 trên 2.800 study test |
| Phân tích lỗi và CE lexicon | scripts/analyze_stage2_generations.py | 2.800 study test |
| Cue theo ngưỡng và cue oracle | probe trên máy huấn luyện | 2.800 và 1.000 study test |
| Pha 1b tốt nhất trên test | pretraining.train, chế độ chỉ đánh giá | 3.269 study test |
| Pilot 1b sửa BioViL | script trên máy huấn luyện | 60.000 train, 1.808 val |
| Pilot Stage 2 | script trên máy huấn luyện | 10.000 train, 500 val |
| Chấm theo giao thức bài báo | scripts/score_stage2_paper_protocol.py | theo từng lần chấm |
| CE bằng CheXbert | scripts/score_ce_chexbert.py | theo từng lần chấm |

CE lexicon là độ đo dùng bộ phân tích theo luật của dự án để gán nhãn 14 bệnh, có cực tính, cho cả báo cáo sinh ra lẫn báo cáo tham chiếu. Đây là thước đo tương đối, không phải CE tính bằng CheXbert, và số tuyệt đối của hai độ đo không so được với nhau.

# Chẩn đoán META-Former

## Soft token sụp ở pha nào
::table Mức độ sụp của 32 soft token theo checkpoint, đo trên 200 study val
::widths 4.3, 3, 3, 3.1, 3.1
::font 10
| Checkpoint | Cosine giữa 32 token | Số hướng hiệu dụng | Entropy cross-attention tầng 3 | Độ giống bản đồ attention |
|---|---|---|---|---|
| Khởi tạo BLIP-2 | 0,445 | 2,94 | 0,61 | 0,27 |
| Sau pha 1a | **0,9989** | **1,002** | **0,11** | **1,000** |
| Sau pha 1b | 0,9989 | 1,002 | 0,15 | 0,978 |
| Sau pha 1c, v3 dùng | 0,9997 | 1,001 | 0,09 | 0,906 |
| Pha 1c của run cũ 0925b | 0,9999 | 1,000 | 0,05 | 0,999 |
::check D["collapse"]["cos_1a"] > 0.99 and D["collapse"]["cos_init"] < 0.5 | Số liệu không ủng hộ câu soft token sụp ở pha 1a

Cosine trung bình giữa 32 token của một study tăng từ 0,445 ở khởi tạo lên 0,9989 ngay sau pha 1a và giữ nguyên ở các pha sau. Một phép đo độc lập trên cache Stage 2 của v3, cùng checkpoint 1c, cho cosine trong một study là 0,9995 và giữa các study là 0,4112. Bộ đọc BLIP-2 chưa huấn luyện, đo ngày 3 tháng 9 năm 2026 trên một run khác, cho 0,80; như vậy pha 1a làm hiện tượng nặng thêm. Run cũ cũng sụp, nên đây là lỗi có tính hệ thống của thiết kế hoặc cách huấn luyện chứ không phải của riêng một lần chạy.

## Cơ chế
Sau pha 1a, ở hai tầng cross-attention đầu, token nhận nhiều attention nhất là token ở chỉ số 196 trong 100% study, tức token CLS toàn cục của PubMedCLIP, chiếm 0,23 và 0,33 tổng attention. Vì cả 32 query cùng đọc một token toàn cục, trạng thái của chúng trùng nhau: cosine 0,915 sau tầng 1 và 1,000 từ tầng 3. Các tầng sau dùng chung trọng số cho mọi query, nên không còn gì phá được đối xứng và hiện tượng sụp không đảo ngược được. Các mục tiêu của pha 1a cũng không đòi các query khác nhau: ITC lấy giá trị lớn nhất theo query, LM dùng các query làm tiền tố, ITM lấy trung bình. ITM ở mức ngẫu nhiên suốt pha 1a, với loss val từ 0,644 đến 0,670 so với mức ngẫu nhiên 0,6365.

## Ảnh hưởng của pha 1b và pha 1c
Ở pha 1b, Q-Former bị đóng băng sau 10% số bước đầu, trong khi StreamAdapter, vốn không có LayerNorm phía sau ở luồng PubMedCLIP và Swin, đẩy norm token từ 19,6 lên 1.713 ở PubMedCLIP và từ 12 lên 413 ở Swin. Cosine giữa các token trong một ảnh tăng từ 0,01 lên 0,45 ở BioViL và lên 0,71, 0,65 ở hai luồng còn lại, tức token ảnh mất phần lớn tính không gian.

Ở pha 1c, dài 1 epoch với batch 8, chỉ số truy hồi ITC R@5 giảm từ 0,65 xuống 0,17. Khi thay mọi token ảnh bằng token trung bình của ảnh, đầu ra của Q-Former vẫn có cosine 0,81 với đầu ra thật, nghĩa là Q-Former gần như chỉ đọc ảnh trung bình. Gradient của mục tiêu căn chỉnh lớn gấp 7 đến 35 lần gradient phân loại trên lớp chiếu dùng chung, và cosine giữa hai gradient xấp xỉ 0.

## Thông tin bệnh còn lại trong soft token
Có hai phép đo với giao thức khác nhau, nên số tuyệt đối lệch nhau và không so trực tiếp được; kết luận theo từng bệnh thì trùng nhau.

::table Linear probe trên cache của v3, fit trên 50.000 study train, chấm trên 2.780 study test
::widths 7.5, 2.4, 2.2, 2.2, 2.2
::font 10.5
| Nguồn | AUROC trung bình | Negative | Positive | Uncertain |
|---|---|---|---|---|
| MHCAC trên cùng 2.780 study | 0,7454 | 0,7800 | 0,7847 | 0,6162 |
| Probe trên trung bình 32 token | 0,7127 | 0,7262 | 0,7464 | 0,6301 |
| Probe trên cả 32 token, 24.576 chiều | 0,6530 | 0,6779 | 0,6748 | 0,5715 |
| Probe trên trung bình, nhãn train xáo trộn | 0,5607 | không có | không có | không có |
::note Mốc xáo trộn nhãn là 0,56 chứ không phải 0,50 vì mô hình được chọn theo val; chênh lệch dưới khoảng 0,05 giữa probe và MHCAC chỉ nên coi là xấp xỉ.

::table Linear probe cùng giao thức, fit trên 53.300 study, chấm trên 2.780 study test
::widths 7.5, 2.4, 2.2, 2.2, 2.2
::font 10.5
| Nguồn | AUROC trung bình | Positive | Negative | Uncertain |
|---|---|---|---|---|
| MHCAC, mốc | **0,745** | 0,785 | 0,780 | 0,616 |
| Soft token, trung bình 32 | 0,725 | 0,752 | 0,743 | 0,644 |
| Ảnh MedGemma, trung bình 256 token | 0,690 | 0,731 | 0,709 | 0,584 |

::table AUROC lớp Positive theo bệnh trong phép đo cùng giao thức, 2.780 study test
::widths 5.5, 3.6, 3.6, 3.8
::font 10.5
| Bệnh | MHCAC | Soft token | Ảnh MedGemma |
|---|---|---|---|
| Edema | 0,843 | 0,846 | 0,822 |
| Pleural Effusion | 0,899 | 0,892 | 0,860 |
| Support Devices | 0,905 | 0,900 | 0,900 |
| Pneumothorax | 0,865 | 0,831 | 0,782 |
| Lung Lesion | 0,748 | 0,718 | 0,704 |
| Pleural Other | 0,833 | 0,747 | 0,602 |
| Fracture | 0,684 | **0,492** | 0,623 |

Trong phép đo thứ nhất, trung bình 32 token giữ được 87% phần AUROC vượt mức ngẫu nhiên của MHCAC. Soft token ngang MHCAC ở các bệnh phổ biến và mất ở các bệnh khu trú hoặc hiếm; Fracture rơi về mức ngẫu nhiên. Một vector duy nhất cũng không chứa được vị trí, mức độ, nhiều bệnh cùng lúc hay chi tiết thiết bị. Nút thắt vì vậy là hiện tượng sụp, chỉ còn một token hiệu dụng, chứ không phải thiếu tín hiệu của 14 nhãn. Ảnh MedGemma khi lấy trung bình thấp hơn soft token ở 12 trên 14 bệnh.

# Stage 1

## Độ dài và cấu hình các pha
::table Quan sát theo từng pha của run_20261005_paper
::widths 3.2, 8.3, 5
::font 10
| Pha | Quan sát | Kết luận |
|---|---|---|
| 1a, 4 epoch, batch 128 | R@5 theo epoch 0,61; 0,63; 0,66; 0,67. Loss ITC trên val thấp nhất ở epoch 1 rồi tăng từ 2,95 lên 3,38. ITM ở mức ngẫu nhiên | 2 epoch là đủ; vấn đề chính là sụp token |
| 1b, 5 epoch | AUROC val theo epoch 0,757; 0,764; 0,778; 0,775; 0,773. Macro recall đạt đỉnh ở epoch chỉ số 2 | 3 epoch là đủ |
| 1c, 1 epoch, batch 8 | AUROC val từ 0,778 xuống 0,771; macro recall từ 0,492 xuống 0,486; R@5 từ 0,65 xuống 0,17 | Không giúp phân loại, phá căn chỉnh |
| StreamAdapter | Norm token tăng 90 lần và 34 lần trong pha 1b | Thiếu chuẩn hóa sau adapter |

So sánh checkpoint 1b tốt nhất với checkpoint 1c trên 3.269 study test, dùng argmax và bootstrap ghép cặp 1.000 lần: AUROC trung bình là 0,7473 so với 0,7436, chênh +0,0037 với khoảng tin cậy [-0,0012; +0,0095], không có ý nghĩa thống kê. Macro recall nghiêng về 1b, +0,0050 [+0,0021; +0,0076]; weighted F1 và F1 năm bệnh nghiêng về 1c, lần lượt -0,0074 [-0,0091; -0,0056] và -0,0144 [-0,0179; -0,0107]. Khả năng phân biệt của hai checkpoint như nhau; chênh lệch F1 là do điểm làm việc của argmax dịch chuyển, và quy tắc hai ngưỡng fit trên val hiệu chỉnh lại điểm này.
::check D["s1"]["b_vs_c_auroc_lo"] < 0 < D["s1"]["b_vs_c_auroc_hi"] | Văn bản nói chênh AUROC giữa 1b và 1c không có ý nghĩa nhưng khoảng tin cậy không chứa 0

Tăng số epoch không giúp vì sau 10% số bước đầu của pha 1b chỉ còn MHCAC, StreamAdapter, view fusion và đầu MPC được học. Cả ba encoder, lớp chiếu dùng chung và lớp chuẩn hóa thị giác đều đóng băng. Đường phân loại là một bộ đọc trên đặc trưng cố định, gần với linear probe mở rộng, nên hội tụ trong 2 đến 3 epoch rồi quá khớp. Một run cũ theo cách chấm nhị phân đã gỡ, khi kéo từ 10 lên 15 epoch, cho AUROC test thay đổi -0,0005 [-0,0043; +0,0031]. Theo quyết định của người dùng ngày 11 tháng 10, pha 1b còn 3 epoch và có chấm test, pha 1c chỉ chạy khi được yêu cầu. Lịch cosine 3 epoch là một đường tốc độ học mới và chưa được chạy.

## Số chính hiện tại
::table Kết quả phân loại ba lớp trên 3.269 study test
::widths 5.5, 2.3, 2.3, 2.3, 2, 2.1
::font 10
| Quy tắc quyết định | Weighted precision | Weighted recall | Weighted F1 | F1 năm bệnh | Macro recall |
|---|---|---|---|---|---|
| Hai ngưỡng fit trên val, số chính | 0,827 | 0,834 | **0,820** | **0,749** | 0,446 |
| Argmax, giao thức bài báo | 0,839 | 0,781 | 0,780 | 0,669 | 0,461 |
| Bài báo | 0,87 | 0,78 | 0,73 | 0,701 | không có |
::note Khoảng tin cậy của F1 năm bệnh theo quy tắc hai ngưỡng là [0,741; 0,757]. AUROC không phụ thuộc quy tắc quyết định: 0,784; 0,778; 0,663 cho ba lớp.
::check D["s1"]["f1w"] > D["s1"]["f1w_paper"] and D["s1"]["f15_lo"] > D["s1"]["f15_paper"] | Số chính Stage 1 không cao hơn bài báo như văn bản viết

Mỗi bệnh có hai ngưỡng trên tỉ số giữa xác suất dương tính và tổng xác suất dương tính với âm tính, fit trên 1.808 study val để tối đa weighted F1 ba lớp. Quy tắc vẫn cho ba lớp. So với argmax, F1 năm bệnh tăng +0,0805 [+0,0743; +0,0866] và macro recall giảm 0,0149. Bài báo báo cáo số phân loại bằng argmax và không nêu ngưỡng của phương trình 22 được fit trên tập nào, vì vậy mọi số chính phải được ghi kèm rằng đây không phải giao thức argmax của bài báo.

## Nhiễu nhãn
::table Độ khớp giữa nhãn CheXpert tự động và nhãn bác sĩ trên 685 study test
::widths 6.5, 3, 4, 3
::font 10.5
| Bệnh | Kappa | Precision dương tính của nhãn tự động | Recall |
|---|---|---|---|
| Atelectasis | 0,933 | 0,949 | 0,944 |
| Pleural Effusion | 0,884 | 0,953 | 0,901 |
| Edema | 0,858 | 0,883 | 0,896 |
| Consolidation | 0,849 | 0,754 | 0,929 |
| Support Devices | 0,816 | 0,817 | 0,965 |
| Cardiomegaly | 0,806 | 0,771 | 0,914 |
| Fracture | 0,804 | 0,725 | 0,925 |
| Lung Lesion | 0,752 | 0,768 | 0,811 |
| Pneumothorax | 0,716 | 0,586 | 0,895 |
| Lung Opacity | 0,708 | 0,724 | 0,917 |
| Pneumonia | 0,683 | 0,446 | 0,673 |
| Enlarged Cardiomediastinum | 0,601 | 0,448 | 0,565 |
| No Finding | 0,563 | 0,455 | 0,833 |
| Pleural Other | 0,493 | 0,417 | 0,600 |
::check abs(D["labels"]["kappa_min"] - 0.493) < 1e-9 and abs(D["labels"]["kappa_max"] - 0.933) < 1e-9 | Khoảng kappa trong tóm tắt không khớp bảng

Mô hình vừa học vừa được chấm bằng nhãn tự động, nên các bệnh có nhãn kém có trần thấp bất kể kiến trúc. Khi chấm cùng một mô hình theo nhãn bác sĩ trên 676 study, AUROC lớp Positive cao hơn +0,022 [+0,008; +0,036] và macro recall cao hơn +0,049 [+0,022; +0,064] so với chấm bằng nhãn tự động.

## Đo đặc trưng từng encoder
::table Probe tuyến tính ba lớp trên đặc trưng của từng nguồn, 53.300 study fit và 2.780 study test, ảnh sạch
::widths 9.5, 3.5, 3.5
::font 10.5
| Nguồn đặc trưng | AUROC trung bình | AUROC Positive |
|---|---|---|
| MHCAC, dự đoán của mô hình, mốc | 0,745 | 0,785 |
| MedCLIP-Swin, token pooled, pretrained | **0,734** | 0,768 |
| RAD-DINO, CLS nối trung bình patch | 0,721 | 0,764 |
| PubMedCLIP, token CLS | 0,638 | 0,661 |
| BioViL sau run mở khóa encoder 10 epoch | 0,726 | 0,764 |
| BioViL sau pha 1c, 1 epoch mở khóa | 0,707 | 0,739 |
| BioViL pretrained, lớp chiếu 1408 chưa huấn luyện | 0,490 | không có |
| Soft token sau pha 1a | 0,721 | 0,749 |
| Soft token sau pha 1b | 0,689 | 0,718 |
| Soft token sau pha 1c | 0,725 | 0,753 |
::check D["probe"]["raddino"] < D["probe"]["swin"] and D["probe"]["soft_1b"] < D["probe"]["soft_1a"] | Bảng probe không ủng hộ kết luận về RAD-DINO hoặc về pha 1b

Từ bảng này rút ra bốn điều. Thứ nhất, RAD-DINO một mình thấp hơn MedCLIP-Swin, nên không được tích hợp. Thứ hai, pha 1b làm soft token kém đi 0,031 AUROC và pha 1c kéo lại. Thứ ba, MHCAC chỉ hơn probe tuyến tính trên một token của MedCLIP-Swin 0,012 AUROC, tức hai luồng còn lại gần như không đóng góp. Thứ tư, đặc trưng CLS của PubMedCLIP giống hệt nhau giữa bản pretrained, sau pha 1c và sau run mở khóa encoder, đều 0,6378, nên chưa xác minh được việc mở khóa hai khối cuối của CLIP có thực sự đổi trọng số hay không.

## Hai lỗi ở luồng BioViL
Lỗi thứ nhất là token bị trộn. BioViL trả về tensor có dạng số kênh 1408 trên lưới 14 nhân 14, và mã nguồn đổi hình dạng thành 196 token 1408 chiều mà không hoán vị trục. Vì vậy 196 token thực chất là các nhóm khoảng 7 bản đồ kênh được duỗi phẳng chứ không phải 196 ô ảnh, và một ô ảnh thật bị rải ra cả 196 token. Mã hóa vị trí, attention theo vùng và bản đồ giải thích trên luồng này đều mất ý nghĩa. Mã nguồn gốc của tác giả có đúng dòng lệnh này.

Lỗi thứ hai là lớp chiếu 1408 chiều được khởi tạo ngẫu nhiên và đóng băng trong pha 1a và 1b. Khi nạp trọng số BioViL-T với kích thước đầu ra 1408, lớp chiếu 128 chiều gốc bị bỏ, và lớp chiếu mới chỉ nằm trong danh sách huấn luyện của pha 1c. Hệ quả là phần lớn tín hiệu phân loại hiện đến từ MedCLIP-Swin, còn BioViL-T, encoder có độ phân giải không gian cao nhất, gần như chưa được dùng đúng. Một cờ tùy chọn để dùng bố cục patch đúng đã được thêm vào mã; mặc định vẫn là bố cục cũ và cấu hình đang dùng không bật cờ này.

## Pilot sửa luồng BioViL
Pilot chỉ chạy pha 1b trên 60.000 study train trong 2 epoch, từ cùng một checkpoint 1a, chấm trên 1.808 study val. Arm A là cấu hình đang dùng. Arm B dùng bố cục patch, huấn luyện lớp chiếu của BioViL và không đóng băng lớp chiếu dùng chung. Arm C là arm B cộng thêm mở khóa encoder nông.

::table Kết quả pilot 1b theo epoch trên 1.808 study val
::widths 2, 3, 2.2, 2.6, 2.2, 4.5
::font 10
| Arm | Epoch | AUROC trung bình | Macro recall | F1 năm bệnh | Loss phân loại trên train ở cuối |
|---|---|---|---|---|---|
| A | 0 | 0,702 | 0,476 | 0,689 | không có |
| A | 1 | **0,740** | 0,483 | 0,714 | 0,570 |
| B | 0 | 0,688 | 0,469 | 0,701 | không có |
| B | 1 | 0,721 | 0,477 | 0,706 | 0,566 |
| C | 0 | 0,704 | 0,485 | 0,716 | không có |
| C | 1 | 0,726 | 0,482 | 0,727 | 0,560 |

::table So sánh ghép cặp với arm A, bootstrap 1.000 lần, epoch tốt nhất theo macro recall
::widths 3, 4.5, 3, 6
::font 10.5
| So sánh | Chỉ số | Hiệu | Khoảng tin cậy 95% |
|---|---|---|---|
| B trừ A | AUROC trung bình | **-0,0197** | [-0,0288; -0,0103] |
| B trừ A | AUROC Positive | -0,0238 | [-0,0359; -0,0132] |
| B trừ A | Macro recall | -0,0052 | [-0,0093; -0,0014] |
| B trừ A | F1 năm bệnh | -0,0077 | [-0,0126; -0,0027] |
| C trừ A | AUROC trung bình | **-0,0358** | [-0,0470; -0,0247] |
| C trừ A | AUROC Positive | -0,0366 | [-0,0500; -0,0219] |
| C trừ A | Macro recall | +0,0025 | [-0,0032; +0,0081] |
| C trừ A | F1 năm bệnh | +0,0017 | [-0,0040; +0,0080] |
::check D["biovil"]["BA_hi"] < 0 and D["biovil"]["CA_hi"] < 0 | Văn bản nói B và C thấp hơn A có ý nghĩa nhưng cận trên không âm

Theo quy tắc đã chốt trước, không arm nào thắng, nên checkpoint run_20261005_paper được giữ và Stage 1 không được huấn luyện lại. Arm B và C có loss trên train thấp hơn nhưng AUROC trên val thấp hơn arm A có ý nghĩa thống kê. Pilot này không chứng minh bố cục patch kém hơn về nguyên tắc: B và C khởi từ một checkpoint 1a đã khớp với bố cục cũ, chỉ có 2 epoch trên 60.000 study, và cả ba arm còn đang tăng ở epoch cuối. Pilot chỉ cho thấy trong ngân sách kiểm chứng rẻ, sửa hai lỗi không đem lại lợi ích đo được, nên chưa đủ căn cứ để chi 15 đến 20 giờ huấn luyện lại pha 1a và 1b.

Khi đọc mã còn có hai phát hiện phụ chưa được sửa. Augmentation lúc huấn luyện chỉ đi vào luồng BioViL, còn PubMedCLIP và Swin có tiền xử lý riêng không augmentation. Record train của Stage 2 được tính từ ảnh đã augmentation ngẫu nhiên, trong khi record val và test được tính từ ảnh sạch.

# Chẩn đoán Stage 2 của mô hình v3

## Các mốc so sánh
::table Mô hình v3 và các mốc không học trên 2.800 study test, giao thức bài báo
::widths 5.3, 1.3, 1.3, 1.4, 1.3, 1.4, 1.4, 1.5, 1.6
::font 9
| Hệ thống | B-1 | B-4 | MTR | R-L | CIDEr | BS | CE lex | F1-5 lex |
|---|---|---|---|---|---|---|---|---|
| Bài báo META-CXR | 0,390 | 0,102 | 0,173 | 0,280 | 0,291 | 0,426 | không có | không có |
| v3, giải mã beam | 0,357 | 0,098 | 0,137 | 0,265 | 0,128 | 0,369 | 0,239 | 0,379 |
| v3, giải mã greedy | 0,285 | 0,073 | 0,120 | 0,250 | 0,084 | 0,361 | 0,230 | 0,346 |
| Một báo cáo cố định | 0,256 | 0,061 | 0,101 | 0,238 | 0,062 | 0,340 | 0,060 | 0,128 |
| Báo cáo train ngẫu nhiên | 0,281 | 0,046 | 0,103 | 0,202 | 0,035 | 0,290 | 0,164 | 0,235 |
| Chép theo soft token | 0,330 | 0,062 | 0,125 | 0,220 | 0,066 | 0,315 | 0,259 | 0,362 |
| Chép theo ảnh MedGemma | 0,338 | 0,068 | 0,130 | 0,226 | 0,077 | 0,328 | 0,240 | 0,336 |
| Chép theo nhãn MHCAC | 0,334 | 0,057 | 0,123 | 0,213 | 0,052 | 0,304 | 0,294 | 0,455 |
| Chép theo nhãn thật | 0,333 | 0,072 | 0,128 | 0,230 | 0,085 | 0,334 | 0,416 | 0,559 |
| Chép lần trước, không có thì v3 beam | **0,391** | **0,106** | 0,155 | 0,259 | 0,169 | 0,365 | không có | không có |
::note B-1 và B-4 là BLEU-1 và BLEU-4, MTR là METEOR, R-L là ROUGE-L, BS là BERTScore, CE lex và F1-5 lex là CE lexicon trên 14 bệnh và trên năm bệnh.
::check D["prior"]["bleu1"] >= D["prior"]["paper_bleu1"] and D["prior"]["bleu4"] >= D["prior"]["paper_bleu4"] | Mốc chép lần chụp trước không ngang bài báo ở BLEU như văn bản viết

Chỉ xét 2.508 study có lần chụp trước, việc chép FINDINGS của lần trước so với v3 beam cho CE lexicon 0,356 so với 0,236, CIDEr tăng +0,0459 [+0,0210; +0,0699], ROUGE-L giảm 0,0067 [-0,0116; -0,0017] và BERTScore thay đổi -0,0049 [-0,0107; +0,0009]. Ghép chép lần trước nếu có, nếu không thì dùng v3 beam, trên cả 2.800 study đạt BLEU-1 0,391 và BLEU-4 0,106, ngang bài báo. Đây là bằng chứng tại chỗ mạnh nhất cho vai trò của bối cảnh.
::check D["prior"]["cider_lo"] > 0 | Văn bản nói CIDEr tăng khi chép lần chụp trước nhưng cận dưới không dương

## Mô hình không dùng cue
::table Thay cue và giữ nguyên mọi thứ khác trên mô hình v3
::widths 5.5, 1.6, 3.2, 3.1, 3.1
::font 9.5
| Phép thay | n | BERTScore | CIDEr | CE lexicon |
|---|---|---|---|---|
| Phương trình 22 sang ngưỡng hai mức, beam | 2.800 | +0,0010 [-0,0015; +0,0035] | +0,0040 [-0,0073; +0,0157] | từ 0,239 xuống 0,215 |
| Phương trình 22 sang nhãn thật, greedy | 1.000 | -0,0024 [-0,0079; +0,0032] | +0,0050 [-0,0104; +0,0196] | từ 0,269 xuống 0,249 |
| Hoán đổi cue giữa các study, val | 300 | -0,014 | không có | không có |
| Hoán đổi soft token giữa các study, val | 300 | -0,072 | không có | không có |
::check D["oracle"]["bs_lo"] < 0 < D["oracle"]["bs_hi"] and D["oracle"]["cider_lo"] < 0 < D["oracle"]["cider_hi"] | Văn bản nói cue oracle không đổi chỉ số nào có ý nghĩa nhưng một khoảng tin cậy không chứa 0

Cue lấy từ nhãn thật khác cue cũ ở 977 trên 1.000 study, nhưng 153 báo cáo vẫn giống hệt từng chữ và không chỉ số nào thay đổi có ý nghĩa. Mô hình v3 vì vậy bỏ qua kênh cue, và cải thiện phân loại ở Stage 1 không thể cải thiện báo cáo của nó. Một nguyên nhân khả dĩ là lúc huấn luyện cue theo phương trình 22 quá nhiễu, trung bình 4,37 bệnh dương tính mỗi study, nên mô hình học cách không tin; trong khi đó soft token đã mang phần lớn thông tin của các bệnh phổ biến.

## Viết thiếu và viết theo khuôn
Báo cáo sinh ra dài 37 từ với greedy và 47 từ với beam, so với 56 từ của bác sĩ. BERTScore precision là 0,424 trong khi recall chỉ 0,299, và mô hình chỉ nhắc 32 đến 34% số bệnh mà bác sĩ có bàn tới. Mô hình không bao giờ nhắc Fracture, Lung Lesion, Pleural Other và Enlarged Cardiomediastinum, rất ít nhắc Pneumonia và Lung Opacity, và gần như luôn viết các câu âm tính quen thuộc về pneumothorax và effusion. Với greedy chỉ có 1.473 báo cáo khác nhau trên 2.800 study, và báo cáo phổ biến nhất chiếm 7,5 đến 11,4%.

## Thiếu bối cảnh lâm sàng
Có 90,0% phần FINDINGS của tập test chứa ngôn ngữ so sánh theo thời gian, và mô hình tự viết câu so sánh trong 51% báo cáo mà bản gốc không so sánh. Lần chụp trước có FINDINGS tồn tại ở 89,6% study test, 65,4% study val và 64,1% study train, với khoảng cách trung vị 7 đến 9 ngày. Phần INDICATION hoặc HISTORY có ở 91,7% study test, trung vị 8 từ. Bài báo không dùng các nguồn này.

## Độ đo và độ dài huấn luyện
Một báo cáo cố định cho mọi study đạt ROUGE-L 0,238 và BERTScore 0,340, cao hơn việc chép một báo cáo train có nhãn đúng hoàn toàn, vốn đạt 0,230 và 0,334. Các độ đo sinh văn bản vì vậy thưởng văn phong hơn nội dung; đây là giới hạn của thước đo chứ không phải lỗi của mô hình, và trùng với nhận xét trong PromptMRG [4], RaDialog [3] và ReRef [12].

Mô hình v3 được huấn luyện 1 epoch trên 170.370 mẫu. Loss trên train vẫn giảm ở cuối, và loss val 1,061 xấp xỉ loss train 1,080, tức mô hình thiếu huấn luyện chứ không quá khớp. Ảnh hưởng của điều này tới chất lượng sinh chưa đo được.

## Xếp hạng nguyên nhân
Xếp theo độ mạnh của bằng chứng, nguyên nhân thứ nhất là thiếu bối cảnh lần chụp trước. Thứ hai là mô hình bỏ qua cue, đã đo trực tiếp bằng cue oracle. Thứ ba là độ đo thưởng văn phong. Thứ tư là kênh ảnh chỉ còn một vector. Thứ năm là thiếu indication, có tài liệu ủng hộ mạnh và đã được gộp vào pilot cùng lần chụp trước nhưng chưa tách riêng. Thứ sáu là huấn luyện Stage 2 ngắn, với bằng chứng yếu.

# Pilot Stage 2 và lần huấn luyện toàn bộ

## Thiết kế pilot
Ba arm dùng cùng hạt giống, cùng 10.000 study train, trong đó 7.604 record có FINDINGS hợp lệ, cùng 1 epoch và cùng LoRA hạng 8, alpha 16. Arm P0 giống v3: 32 soft token và cue theo phương trình 22, không có bối cảnh. Arm P1 thêm indication và FINDINGS của lần chụp trước vào prompt. Arm P2 là P1, nhưng lúc huấn luyện cue lấy từ nhãn CheXpert thật, trừ 20% record cố định giữ cue của MHCAC, còn lúc suy luận cue lấy từ MHCAC theo ngưỡng hai mức. Mỗi arm được chấm trên 500 study val cố định, với beam 4 kèm length penalty 2 và với greedy, bootstrap ghép cặp 2.000 lần, hạt giống 16.

Quy tắc được chốt trước khi có kết quả. Bối cảnh được nhận nếu hiệu P1 trừ P0 có cận dưới lớn hơn 0 ở cả ROUGE-L và BERTScore. Cách huấn luyện cue được nhận nếu hiệu P2 trừ P1 ở CE lexicon đạt ít nhất +0,02 với cận dưới lớn hơn 0 và không độ đo sinh văn bản nào giảm có ý nghĩa.

Để chạy pilot, bốn phần mã đã được viết: kiểu prompt có bối cảnh, lấy lần chụp trước hẳn gần nhất của cùng bệnh nhân; chế độ cue huấn luyện từ nhãn thật; giải mã beam trong bộ sinh chính thức; và hai script chấm điểm theo giao thức bài báo và theo CheXbert.

## Kết quả pilot
::table Kết quả pilot trên 500 study val
::widths 2.1, 1.5, 1.4, 1.4, 1.4, 1.4, 1.5, 1.4, 1.5, 1.5, 1.4
::font 9
| Giải mã | Arm | B-1 | B-4 | MTR | R-L | CIDEr | BS | CE lex | F1-14 | F1-5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Beam | P0 | 0,370 | 0,121 | 0,142 | 0,301 | 0,284 | 0,410 | 0,212 | 0,438 | 0,438 |
| Beam | P1 | 0,408 | 0,165 | 0,168 | 0,336 | 0,487 | 0,437 | 0,289 | 0,441 | 0,441 |
| Beam | P2 | **0,423** | **0,178** | **0,172** | **0,345** | **0,518** | **0,448** | **0,308** | **0,463** | **0,494** |
| Greedy | P0 | 0,296 | 0,094 | 0,127 | 0,288 | 0,193 | 0,410 | 0,213 | 0,452 | 0,469 |
| Greedy | P1 | 0,349 | 0,146 | 0,156 | 0,338 | 0,407 | 0,455 | 0,287 | 0,457 | 0,453 |
| Greedy | P2 | 0,345 | 0,144 | 0,156 | 0,337 | 0,387 | 0,460 | 0,290 | 0,470 | 0,480 |
::note F1-14 và F1-5 là micro F1 tính bằng CheXbert trên 14 bệnh và trên năm bệnh. Các ký hiệu khác như ở bảng các mốc so sánh. Số của P0 trên 500 study val không so trực tiếp được với v3 trên 2.800 study test.

::table So sánh ghép cặp trong pilot, n bằng 500, bootstrap 2.000 lần, hạt giống 16
::widths 4, 4.5, 2.5, 5.5
::font 9.5
| So sánh | Chỉ số | Hiệu | Khoảng tin cậy 95% |
|---|---|---|---|
| P1 trừ P0, beam | BERTScore | +0,0275 | [+0,0143; +0,0406] |
| P1 trừ P0, beam | ROUGE-L | +0,0359 | [+0,0229; +0,0490] |
| P1 trừ P0, beam | CIDEr | +0,2027 | [+0,1118; +0,3002] |
| P1 trừ P0, beam | BLEU-4 | +0,0386 | [+0,0249; +0,0535] |
| P1 trừ P0, beam | CE lexicon | +0,0763 | [+0,0370; +0,1131] |
| P1 trừ P0, beam | CheXbert micro F1-5 | +0,0022 | [-0,0525; +0,0576] |
| P1 trừ P0, greedy | BERTScore | +0,0450 | [+0,0319; +0,0574] |
| P1 trừ P0, greedy | ROUGE-L | +0,0496 | [+0,0372; +0,0617] |
| P1 trừ P0, greedy | CIDEr | +0,2132 | [+0,1338; +0,2965] |
| P1 trừ P0, greedy | BLEU-4 | +0,0537 | [+0,0410; +0,0660] |
| P1 trừ P0, greedy | CE lexicon | +0,0744 | [+0,0318; +0,1122] |
| P1 trừ P0, greedy | CheXbert micro F1-5 | -0,0157 | [-0,0653; +0,0379] |
| P2 trừ P1, beam | BERTScore | +0,0110 | [+0,0010; +0,0209] |
| P2 trừ P1, beam | ROUGE-L | +0,0081 | [-0,0000; +0,0164] |
| P2 trừ P1, beam | CIDEr | +0,0312 | [-0,0145; +0,0740] |
| P2 trừ P1, beam | BLEU-4 | +0,0103 | [+0,0026; +0,0180] |
| P2 trừ P1, beam | CE lexicon | +0,0190 | [-0,0075; +0,0453] |
| P2 trừ P1, beam | CheXbert micro F1-5 | +0,0532 | [+0,0158; +0,0889] |
| P2 trừ P1, greedy | BERTScore | +0,0050 | [-0,0025; +0,0124] |
| P2 trừ P1, greedy | ROUGE-L | -0,0004 | [-0,0071; +0,0064] |
| P2 trừ P1, greedy | CIDEr | -0,0195 | [-0,0624; +0,0219] |
| P2 trừ P1, greedy | BLEU-4 | -0,0040 | [-0,0108; +0,0028] |
| P2 trừ P1, greedy | CE lexicon | +0,0029 | [-0,0403; +0,0499] |
| P2 trừ P1, greedy | CheXbert micro F1-5 | +0,0266 | [-0,0095; +0,0655] |

Bối cảnh được nhận theo quy tắc: hiệu P1 trừ P0 có cận dưới lớn hơn 0 ở cả ROUGE-L và BERTScore, với cả beam và greedy. Micro F1 của CheXbert không đổi, còn macro F1 trên 14 bệnh tăng.
::check min(D["pilot"]["p1p0_beam_rouge_lo"], D["pilot"]["p1p0_beam_bs_lo"], D["pilot"]["p1p0_greedy_rouge_lo"], D["pilot"]["p1p0_greedy_bs_lo"]) > 0 | Quy tắc nhận bối cảnh không thỏa

Cách huấn luyện cue bằng nhãn thật không đạt ngưỡng đã chốt: CE lexicon với beam tăng +0,0190 với khoảng tin cậy [-0,0075; +0,0453]. Ngoài quy tắc, micro F1 năm bệnh của CheXbert với beam tăng +0,0532 [+0,0158; +0,0889], BERTScore và BLEU-4 với beam tăng có ý nghĩa, và không độ đo sinh văn bản nào giảm có ý nghĩa. Riêng ROUGE-L với beam có cận dưới cách 0 chưa tới 0,0005 nên được coi là chưa xác lập. Tín hiệu là tốt nhưng chưa xác lập ở n bằng 500 và một hạt giống. Người dùng đã chọn cấu hình P2 cho lần huấn luyện toàn bộ, biết rằng điều kiện thứ hai chưa đạt.
::check D["pilot"]["p2p1_beam_ce"] < 0.02 and D["pilot"]["p2p1_beam_ce_lo"] < 0 | Văn bản nói cue nhãn thật không đạt ngưỡng nhưng số liệu cho thấy đạt
::check D["pilot"]["p2p1_beam_f15_lo"] > 0 | Văn bản nói CheXbert F1-5 tăng có ý nghĩa nhưng cận dưới không dương

## Lần huấn luyện toàn bộ
Lần huấn luyện dùng cấu hình P2 trong 2 epoch, với Stage 1 là checkpoint 1c tốt nhất của run_20261005_paper, batch 2 nhân với tích lũy gradient 8, lưu checkpoint mỗi 32 bước cập nhật. Chế độ là soft token kèm prompt MHCAC, chỉ phần FINDINGS, kiểu prompt có bối cảnh, cue suy luận theo ngưỡng hai mức, cue huấn luyện từ nhãn thật với 20% giữ cue mô hình. Sau khi huấn luyện, hệ thống sinh báo cáo trên tập test bằng beam 4 kèm length penalty 2 và bằng greedy, tối đa 256 token mới. Ước tính ban đầu là khoảng 1,8 giờ dựng record Stage 1, khoảng 40 giờ huấn luyện và khoảng 3 giờ sinh.

Trước khi để lần chạy kéo dài hai ngày, ba điểm đã được rà soát. Về độ dài chuỗi, trên 7.604 record train và 1.360 record val của pilot P2, prompt dài tối đa 547 token và cả chuỗi tối đa 676 token, dưới giới hạn 768, nên không dòng nào bị cắt. Về chọn checkpoint, val loss được chấm cuối mỗi epoch và adapter tốt nhất được lưu riêng. Về các thiết lập giữ nguyên như pilot, gồm hạng LoRA, tốc độ học 1e-4 và kích thước batch, chưa có phép đo nào ủng hộ thay đổi. Kết luận của lần rà soát là không có tối ưu đã kiểm chứng nào mà lần chạy còn thiếu.

Lúc 16 giờ 31 ngày 11 tháng 10, sau 41 phút, tiến trình còn sống, log không có lỗi, ổ dữ liệu dùng đúng driver và phân vùng làm việc còn 44 GB trống. Script chờ sẽ chấm hai bộ báo cáo sinh trên test bằng hai script chấm điểm và ghép cặp với v3 beam trên các study chung trong 2.800 study; nếu lần chạy dừng vì lỗi thì script không chấm gì.

Khi lần chạy kết thúc còn bốn việc. Bước 1, đọc val loss của epoch 1 và epoch 2 để biết epoch thứ hai có giúp không. Bước 2, đọc điểm test, so ghép cặp với v3 và so với bài báo; bản greedy mới được ghép với v3 beam, không cùng cách giải mã, và điều này phải được ghi rõ. Bước 3, ghi rõ mô hình mới có indication và báo cáo lần chụp trước nên không cùng điều kiện với bài báo. Bước 4, cập nhật các tài liệu của dự án. Nếu lần chạy lỗi thì báo nguyên nhân và hỏi người dùng trước khi chạy lại.

# Đối chiếu với bài báo gốc

## Cách phân loại
Tính mới ở đây được hiểu là so với bài báo META-CXR, không phải so với toàn bộ tài liệu; các kỹ thuật lấy từ công trình khác đều được ghi nguồn. Loại A là điểm khác bài báo, có số đo cho thấy tốt hơn và đã đưa vào pipeline. Loại B là điểm khác đã đưa vào pipeline nhưng chưa tách riêng đóng góp. Loại C là phát hiện mới từ phân tích. Loại D là thử nghiệm không cải thiện. Loại E là đề xuất. Loại K là điểm khác nhưng không phải tính mới, như phần cứng hay sửa lỗi của chính dự án. Trong luận văn, chỉ loại A được viết là cải thiện so với bài báo.

## Bảng chỉ mục
::table Chỉ mục 29 hạng mục khác biệt so với bài báo và trạng thái ngày 11 tháng 10
::widths 1.6, 7.6, 1.2, 6.1
::font 9.5
| Mã | Hạng mục | Loại | Trạng thái |
|---|---|---|---|
| TM-01 | Số chính Stage 1 dùng hai ngưỡng theo bệnh, fit trên val | A | Đang dùng |
| TM-02 | Giải mã beam 4 kèm length penalty 2 | A | Đã có trong bộ sinh chính thức |
| TM-03 | Làm sạch target FINDINGS, manifest v3 | A | Đang dùng |
| TM-04 | GradCache và SigLIP cho pha 1a | A | Đang dùng |
| TM-05 | Đa góc chụp: view fusion, MPC, StreamAdapter, view consistency | B | Đang dùng |
| TM-06 | Luồng PubMedCLIP: chuẩn hóa, trừ trung bình, tiền xử lý riêng | B | Đang dùng |
| TM-07 | Mở khóa một phần encoder ở pha 1c | B | Pha 1c không còn chạy mặc định |
| TM-08 | Chọn checkpoint theo macro recall; loại study không có nhãn | B | Đang dùng |
| TM-09 | Giải thích Stage 2 bằng attention rollout và hai cổng kiểm định | B | Đo trên adapter đã xóa |
| TM-10 | Bootstrap ghép cặp, mốc so sánh, nhóm con theo số ảnh | B | Đang dùng |
| TM-11 | Soft token sụp thành một vector ở pha 1a | C | Chương 2 |
| TM-12 | Pha 1b làm lệch đầu vào Q-Former; pha 1c phá căn chỉnh | C | Chương 2 và 3 |
| TM-13 | Soft token giữ bệnh phổ biến, mất bệnh khu trú hoặc hiếm | C | Chương 2 |
| TM-14 | Độ đo thưởng văn phong; mô hình không dùng thông tin phân loại | C | Chương 4 |
| TM-15 | Thiếu bối cảnh lâm sàng | C | Chương 4 |
| TM-16 | Ngưỡng phương trình 22 không tái lập được và làm giảm F1 | C | Mục 6.3 |
| TM-17 | Cờ EVOKE | D | Không dùng |
| TM-18 | Tối đa 3 ảnh phụ | D | Không dùng |
| TM-19 | Cue Stage 2 theo ngưỡng hai mức trên adapter v3 | D | Đang được thử lại công bằng ở lần huấn luyện P2 |
| TM-20 | Finding token học được | D | Không dùng |
| TM-21 | Logit-adjusted loss | D | Không dùng |
| TM-22 | ITC ở batch nhỏ; khung nhị phân và mention gate | D | Đã gỡ |
| TM-23 | Indication và FINDINGS lần chụp trước trong prompt | E | Đã làm; pilot dương tính; số test chờ lần chạy |
| TM-24 | Kênh ảnh gốc MedGemma thay 32 soft token | E | Chưa làm, không ưu tiên |
| TM-25 | Sửa sụp META-Former; bỏ pha 1c; pha 1b còn 3 epoch | E | Làm một phần; sửa sụp chưa làm |
| TM-26 | Đo CE bằng CheXbert | E | Đã làm |
| TM-27 | MedGemma 1.5 4B QLoRA thay Vicuna-7B | K | Cài đặt |
| TM-28 | Sửa stop token của Stage 2 | K | Sửa lỗi của dự án |
| TM-29 | Độ dài pha theo epoch; gradient checkpointing; bf16 | K | Cài đặt |
::note Cột trạng thái là cập nhật của tài liệu này. Tài liệu chỉ mục gốc còn ghi TM-02, TM-23, TM-25 và TM-26 theo trạng thái trước pilot.

Bốn kết quả của đợt này chưa có mã trong chỉ mục: hai lỗi BioViL kế thừa từ mã nguồn gốc, nhiễu nhãn so với nhãn bác sĩ, kết quả âm của RAD-DINO và của pilot sửa BioViL, và cách huấn luyện cue bằng nhãn thật.

## Các hạng mục có bằng chứng tốt hơn bài báo
TM-01 đã trình bày ở mục 3.2. TM-02: bài báo không nêu cách giải mã; trên 2.800 study test với cùng adapter v3, beam so với greedy tăng BERTScore +0,0083 [+0,0037; +0,0129], ROUGE-L +0,0150 [+0,0112; +0,0188] và CIDEr +0,0439 [+0,0230; +0,0653]. TM-03: 9,8% target train bắt đầu bằng dòng tiêu đề phiếu chụp và 3,4% chỉ có dòng đó; mô hình học lại và 57% báo cáo test chỉ là dòng tiêu đề, BLEU-1 0,056. Sau khi parser bỏ dòng này, không còn báo cáo nào chỉ có tiêu đề và BLEU-1 với greedy đạt 0,285; hai lần chạy này khác cohort test, 2.984 và 2.800 study, nên không phải so sánh ghép cặp. TM-04: trên GPU 16 GB, ITC ở batch 8 cho kết quả ngẫu nhiên trong cả bốn lần đo; với GradCache [14] ở batch 128 và loss sigmoid kiểu SigLIP [15], ITC lần đầu vượt ngẫu nhiên, R@5 đạt 0,664 và 0,641 trên 256 cặp val. Căn chỉnh vượt ngẫu nhiên nhưng token vẫn sụp.

TM-16, thuộc loại C: bài báo không nêu ngưỡng của phương trình 22 fit trên tập nào và không công bố file ngưỡng. Khi fit trên val của dự án và chấm test, weighted F1 là 0,702 và F1 năm bệnh là 0,637, thấp hơn argmax với 0,780 và 0,669.

## Các hạng mục khác biệt chưa tách đóng góp
TM-05: bài báo chỉ dùng ảnh thẳng; dự án dùng một ảnh neo và tối đa một ảnh phụ, với ý tưởng lấy từ EVOKE [13]. Chưa có ablation đa góc chụp so với chỉ ảnh thẳng. TM-06: token patch của PubMedCLIP được đọc qua lớp chuẩn hóa cuối rồi trừ trung bình theo ảnh, đưa cosine trung bình giữa các patch từ 0,674 về -0,014. TM-07: bằng chứng duy nhất, +0,020 AUROC, đo trong khung nhị phân đã gỡ, nên không dùng được làm bằng chứng hiện tại. TM-08: ô trống được coi là âm tính như bài báo, nhưng study không có thông tin CheXpert nào bị loại thay vì thành 14 ô âm tính. TM-09: hai cổng kiểm định gồm thay ảnh của study khác, làm NLL tăng +0,2160 [+0,1860; +0,2468] với n bằng 100, và ngẫu nhiên hóa trọng số, làm hệ số tương quan hạng giảm từ 0,986 xuống 0,295. TM-10: mọi so sánh đều có khoảng tin cậy bootstrap ghép cặp.

## Các thử nghiệm không cải thiện
::table Các thử nghiệm đã đo và không cải thiện
::widths 1.6, 5.6, 9.3
::font 9.5
| Mã | Thử nghiệm | Kết quả |
|---|---|---|
| TM-17 | Cắt gradient ảnh phụ và nhiệt độ MPC 0,5 như EVOKE | F1 năm bệnh -0,0160 [-0,0193; -0,0122]; weighted F1 -0,0033; test |
| TM-18 | Tối đa 3 ảnh phụ | AUROC -0,0085 [-0,0132; -0,0028]; F1 năm bệnh -0,0114; test |
| TM-19 | Cue theo ngưỡng hai mức trên adapter v3 | Beam: mọi chỉ số ngang; greedy: ROUGE-L -0,0030 [-0,0051; -0,0009]; test |
| TM-20 | 13 finding token học được | Không cải thiện; BERTScore -0,0075 so với không cue; 300 study val |
| TM-21 | Logit-adjusted loss | Mô phỏng: hệ số 1,0 sụp về lớp Uncertain; không áp dụng |
| TM-22 | ITC batch 8; khung nhị phân và mention gate | ITC ở mức ngẫu nhiên trong bốn lần đo; khung nhị phân đã gỡ |
| chưa có | RAD-DINO, probe | 0,721 so với MedCLIP-Swin 0,734; 2.780 study test |
| chưa có | Sửa luồng BioViL, pilot pha 1b | AUROC -0,0197 [-0,0288; -0,0103]; 1.808 study val |
::note Kết quả âm của TM-19 đo trên adapter v3, vốn huấn luyện với cue phương trình 22 và bỏ qua cue. Lần huấn luyện P2 là phép thử công bằng mà TM-19 chưa có.

## Phần giống bài báo
Dự án giữ ba encoder đóng băng cùng họ với bài báo; chiếu về 1408 chiều rồi nối theo trục token; che ngẫu nhiên 10% đặc trưng; META-Former khởi tạo từ Q-Former của BLIP-2; MHCAC 6 tầng với 8 token chuyên gia chung và 14 token chuyên gia riêng; ba lớp Positive, Negative, Uncertain với ô trống là âm tính; LoRA hạng 8, alpha 16; tập chia chính thức với 1.808 study val và 3.269 study test; cùng bộ độ đo. Lần huấn luyện P2 khác bài báo ở ba điểm: prompt có thêm indication và FINDINGS lần chụp trước, cue lúc suy luận theo ngưỡng hai mức thay vì phương trình 22, và cue lúc huấn luyện lấy từ nhãn thật. Mô hình ngôn ngữ là MedGemma 1.5 4B thay cho Vicuna-7B vì giới hạn phần cứng 16 GB và giấy phép.

# Tài liệu liên quan

## Sinh báo cáo
::table Kết quả trên MIMIC-CXR của các công trình liên quan tới sinh báo cáo
::widths 3.6, 5.4, 7.5
::font 9.5
| Nguồn | Thay đổi | Hiệu quả được báo cáo |
|---|---|---|
| MAIRA-2 [2] | Ảnh và báo cáo lần trước, mục comparison | Bỏ lúc suy luận: ROUGE-L giảm 28,9%; huấn luyện không có: giảm 11,7% |
| MAIRA-2 [2] | Ảnh nghiêng và technique | Bỏ lúc suy luận: ROUGE-L giảm 22% trên tập có ảnh nghiêng |
| RaDialog [3] | Thêm indication | BLEU-4 từ 9,5 lên 14,8; ROUGE-L từ 27,1 lên 31,6 |
| RaDialog [3] | Danh sách bệnh có cấu trúc | CE từ 26,1 lên 39,4; BLEU-4 từ 9,0 lên 9,5 |
| RaDialog [3] | Mô hình ngôn ngữ 7B, 13B, 33B | BLEU-4 đều 9,5 |
| PromptMRG [4] | Prompt chẩn đoán | CE F1 từ 0,370 lên 0,444; BLEU-4 từ 0,116 xuống 0,106 |
| MedGemma [5] | 4B không fine-tune; fine-tune với ảnh và indication | RadGraph F1 29,5 và 30,3 |
| MedGemma 1.5 [6] | 4B, so sánh hai thời điểm | RadGraph F1 27,2 |
| DeCo [7] | Q-Former so với pooling | Nén bằng query gây thiếu hụt ngữ nghĩa thị giác |
| ViT Registers [8] | Token hút attention | Cần token đệm để giữ attention đúng chỗ |
| METransformer [9] | Loss trực giao giữa các token học được | Giữ các token khác nhau |
| TRRG [10] | Số query 4, 8, 16, 32 | Tốt nhất ở 16 |
| MLRG [11] | Đa góc chụp và lần trước; CE bằng CheXbert | Dẫn đầu ở BLEU-4 và RadGraph |
| ReRef [12] | Đổi văn phong báo cáo tham chiếu | Thứ hạng các mô hình thay đổi |
::note Số của các công trình khác được chép từ tài liệu tổng hợp nguồn của dự án; tài liệu này không kiểm chứng lại từ bài báo gốc.

Mức của v3 với beam, BLEU-4 0,098 và ROUGE-L 0,265, ngang RaDialog ở giai đoạn căn chỉnh, một hệ Q-Former 32 query gần giống META-CXR mà Q-Former không sụp. Vì vậy chỉ sửa Q-Former khó kéo các độ đo sinh văn bản vượt mức này; đòn bẩy lớn nằm ở bối cảnh đầu vào.

## Phân loại và encoder
::table Các công trình liên quan tới thích nghi encoder và nhiễu nhãn
::widths 3.6, 7.4, 5.5
::font 9.5
| Nguồn | Nội dung | Hàm ý |
|---|---|---|
| Tayebi Arasteh và cộng sự [17] | Trên MIMIC-CXR, LoRA thua fine-tune toàn phần 2,1 đến 4,1 điểm AUROC; 512 điểm ảnh tốt hơn 224 | Đóng băng encoder là trần thấp |
| So sánh chiến lược fine-tune [18] | Linear probe khoảng 80 đến 82, fine-tune khoảng 87 đến 88 AUROC | Cùng chiều |
| LP-FT [16] | Linear probe rồi fine-tune tốt hơn fine-tune khoảng 1% | Thứ tự đọc trước, tinh chỉnh sau là hợp lý |
| R2GenGPT [19] | Swin đóng băng sang fine-tune: BLEU-4 từ 0,117 lên 0,134 | Thích nghi encoder giúp cả sinh báo cáo |
| RAD-DINO và BioViL-T [20] | Đặc trưng đóng băng, 14 nhãn: 0,827; 0,805; nối hai 0,837 AUROC | RAD-DINO đáng đo; đã đo và không hơn |
| VisualCheXbert [22], Jain và cộng sự [23] | Nhãn từ báo cáo lệch nhãn từ ảnh | Nhiễu nhãn là giới hạn thật |
| Pham và cộng sự [24] | Làm mềm nhãn Uncertain cải thiện AUC trên CheXpert | Lớp Uncertain là lớp yếu nhất |
| Kiểm định nhãn 2025 [25] | Nhãn công khai khớp chuyên gia dưới 60% trên mẫu nhỏ | Cùng chiều |
::note Số của các công trình khác được chép từ tài liệu tổng hợp nguồn của dự án; tài liệu này không kiểm chứng lại từ bài báo gốc.

Bằng chứng trong dự án cho việc thích nghi encoder đến từ một run cũ theo cách chấm nhị phân đã gỡ: mở khóa encoder nông tăng macro AUROC +0,0201 [+0,0136; +0,0263] ở 14 trên 14 bệnh, còn mở khóa sâu hơn không thêm, +0,0049 [-0,0003; +0,0100]. Pha 1c của run hiện tại cũng mở khóa nông nhưng không tăng AUROC test, vì ở đó encoder bị kéo bởi mục tiêu căn chỉnh chứ không phải mục tiêu phân loại.

# Đề xuất, việc còn mở và giới hạn

## Bảy đề xuất ban đầu và kết cục
::table Bảy đề xuất của đợt điều tra và kết cục ngày 11 tháng 10
::widths 0.9, 6.4, 2.6, 6.6
::font 9.5
| Số | Đề xuất | Độ chắc chắn | Kết cục |
|---|---|---|---|
| 1 | FINDINGS lần chụp trước và indication trong prompt Stage 2 | Cao | Đã làm; pilot đạt quy tắc; đang huấn luyện toàn bộ |
| 2 | Huấn luyện Stage 2 để dùng cue | Trung bình | Đã làm; pilot chưa đạt ngưỡng CE lexicon; người dùng chọn dùng |
| 3 | Beam 4 kèm length penalty 2 trong bộ sinh | Cao | Đã làm |
| 4 | CE bằng CheXbert | Cao | Đã làm |
| 5 | Pha 1b còn 3 epoch; bỏ pha 1c | Trung bình đến cao | Đã làm; lịch 3 epoch chưa chạy |
| 6 | Sửa sụp META-Former | Trung bình | Chưa làm |
| 7 | Kênh ảnh gốc MedGemma thay soft token | Thấp | Chưa làm, không ưu tiên |

Đề xuất 6 gồm bỏ token toàn cục khỏi đầu vào cross-attention của Q-Former hoặc cộng lại embedding vị trí của query ở mỗi tầng, phạt độ giống nhau giữa các query, thêm LayerNorm sau StreamAdapter, bỏ ITM và rút pha 1a còn 2 epoch. Các thay đổi này gần như chắc chắn giảm sụp nhưng chưa chắc tăng độ đo sinh văn bản, và đòi huấn luyện lại pha 1a với cache đặc trưng khoảng 148 GB trong khi phân vùng làm việc còn 44 GB. Đề xuất 7 có tài liệu ủng hộ nhưng các phép đo tại chỗ không ủng hộ, và huấn luyện đắt hơn khoảng 4 lần.

Các hướng sau không được đề xuất vì đã đo và không có lợi: thêm epoch cho pha 1a và 1b, tăng số ảnh phụ, cờ EVOKE, finding token, logit-adjusted loss, pha 1c như cấu hình cũ, mở khóa encoder sâu, RAD-DINO, sửa luồng BioViL trong ngân sách pilot, và đổi sang mô hình ngôn ngữ lớn hơn.

## Việc còn mở
Còn tám việc chưa có câu trả lời. Thứ nhất, kết quả test của lần huấn luyện P2. Thứ hai, epoch thứ hai của Stage 2 có giúp không. Thứ ba, tách đóng góp của indication và của báo cáo lần chụp trước, vì pilot gộp hai thứ. Thứ tư, huấn luyện lại pha 1a trên bố cục patch của BioViL, điều mà pilot chưa thử. Thứ năm, sửa sụp META-Former. Thứ sáu, xác minh việc mở khóa hai khối cuối của CLIP có đổi trọng số không. Thứ bảy, Stage 2 nên lấy soft token từ checkpoint 1a hay 1c; lần huấn luyện P2 dùng checkpoint 1c. Thứ tám, ảnh hưởng của việc record train Stage 2 được tính từ ảnh có augmentation.

## Giới hạn
CE lexicon là xấp xỉ bằng luật và không so được với CE của CheXbert. Các mốc truy hồi và probe dùng 50.000 đến 60.000 study train chứ không phải toàn bộ. Mỗi pha chỉ có một checkpoint và một hạt giống; mỗi arm pilot chỉ có một lần chạy. Cue oracle được đo trên 1.000 study với greedy. Pilot Stage 2 dùng 10.000 study train và 500 study val, nên chỉ các hiệu ghép cặp có giá trị. Embedding MedGemma lấy trung bình 256 token không nói gì về trường hợp mô hình ngôn ngữ đọc đủ 256 token không gian. Số chính Stage 1 không theo giao thức argmax của bài báo, và mô hình Stage 2 mới không cùng điều kiện đầu vào với bài báo. MedCLIP được huấn luyện trước trên MIMIC-CXR và CheXpert với cách chia không kiểm chứng được. Nhãn CheXpert tự động vừa là nhãn huấn luyện vừa là nhãn chấm.

::refs
D. Edirisinghe et al., "Chest X-Ray Report Generation Using Abnormality Guided Vision Language Model," IEEE Access, vol. 13, 2025.
S. Bannur et al., "MAIRA-2: Grounded Radiology Report Generation," arXiv:2406.04449, 2024.
C. Pellegrini et al., "RaDialog: A Large Vision-Language Model for Radiology Report Generation and Conversational Assistance," arXiv:2311.18681, 2023.
H. Jin et al., "PromptMRG: Diagnosis-Driven Prompts for Medical Report Generation," in Proc. AAAI, 2024, arXiv:2308.12604.
"MedGemma Technical Report," arXiv:2507.05201, 2025.
"MedGemma 1.5 Technical Report," arXiv:2604.05081, 2026.
"DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models," arXiv:2405.20985, 2024.
T. Darcet et al., "Vision Transformers Need Registers," arXiv:2309.16588, 2023.
Z. Wang et al., "METransformer: Radiology Report Generation by Transformer with Multiple Learnable Expert Tokens," in Proc. CVPR, 2023.
"TRRG: Towards Truthful Radiology Report Generation with Cross-modal Disease Clue Enhanced Large Language Model," arXiv:2408.12141, 2024.
"MLRG: Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation," in Proc. CVPR, 2025, arXiv:2502.20056.
"Reporting Practice Matters (ReRef)," arXiv:2609.19093, 2026.
"EVOKE: Multi-view chest X-ray report generation," arXiv:2411.10224, 2024.
L. Gao et al., "Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup," arXiv:2101.06983, 2021.
X. Zhai et al., "Sigmoid Loss for Language Image Pre-Training," arXiv:2303.15343, 2023.
A. Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution," in Proc. ICLR, 2022, arXiv:2202.10054.
S. Tayebi Arasteh et al., nghiên cứu về độ phân giải và DINOv3 trên ảnh X-quang ngực, arXiv:2510.07191, 2025.
"Comparison of fine-tuning strategies for transfer learning in medical image classification," arXiv:2406.10050, 2024.
Z. Wang et al., "R2GenGPT: Radiology Report Generation with Frozen LLMs," arXiv:2309.09812, 2023.
Nghiên cứu về đặc trưng đóng băng của RAD-DINO và BioViL-T trên MIMIC-CXR, arXiv:2609.09185, 2026.
F. Perez-Garcia et al., "RAD-DINO: Exploring Scalable Medical Image Encoders Beyond Text Supervision," arXiv:2401.10815, 2024.
S. Jain et al., "VisualCheXbert: Addressing the Discrepancy Between Radiology Report Labels and Image Labels," arXiv:2102.11467, 2021.
S. Jain et al., nghiên cứu về nhãn từ báo cáo X-quang và nhãn từ ảnh, arXiv:2104.00793, 2021.
H. Pham et al., "Interpreting chest X-rays via CNNs that exploit hierarchical disease dependencies and uncertainty labels," arXiv:1911.06475, 2019.
Nghiên cứu kiểm định nhãn công khai của MIMIC-CXR và CheXpert, arXiv:2509.15107, 2025.
