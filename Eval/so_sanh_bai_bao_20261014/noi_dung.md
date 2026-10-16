---
title: SO SÁNH KẾT QUẢ VỚI BÀI BÁO META-CXR
subtitle: Hai checkpoint được báo cáo, cấu hình huấn luyện, kết quả theo các phép đo của bài báo và nguyên nhân sai khác
subtitle: Cập nhật ngày 14 tháng 10 năm 2026
chapter_new_page: false
---
<!-- Nguồn số: ~/eval_20261010_cutpoints/test, ~/stage2_p2_20261011/score/n2800_*.json, log và manifest của hai run trên máy train. Chỉ số tổng hợp. -->

::heading Tóm tắt

Tài liệu này so sánh hệ thống của đề tài với bài báo gốc META-CXR (D. Edirisinghe và cộng sự, IEEE Access, tập 13, năm 2025). Mỗi giai đoạn chỉ báo cáo một checkpoint chính: giai đoạn 1 là checkpoint phân loại bất thường, giai đoạn 2 là bộ adapter sinh báo cáo. Với mỗi checkpoint, tài liệu mô tả nó là gì, được huấn luyện với cấu hình nào, đạt kết quả nào trên tập kiểm tra, khác bài báo ở điểm nào và vì sao kết quả chênh lệch.

::table Kết quả chính của hai checkpoint so với bài báo trên tập kiểm tra MIMIC-CXR
::widths 6.3; 3.2; 3.2; 3.8
::font 10.5
| Chỉ số | Bài báo | Đề tài | Chênh lệch |
|---|---|---|---|
| F1 có trọng số, 14 bệnh (giai đoạn 1) | 0,73 | 0,820 | +0,090 |
| F1 trung bình 5 bệnh (giai đoạn 1) | 0,701 | 0,749 | +0,048 |
| BLEU-1 (giai đoạn 2) | 0,390 | 0,434 | +0,044 |
| BLEU-4 | 0,102 | 0,190 | +0,088 |
| METEOR | 0,173 | 0,190 | +0,017 |
| ROUGE-L | 0,280 | 0,355 | +0,075 |
| CIDEr | 0,291 | 0,416 | +0,125 |
| BERTScore | 0,426 | 0,466 | +0,040 |
| F1 hiệu quả lâm sàng (macro, 14 bệnh) | 0,428 | 0,425 | -0,003 |
::note Giai đoạn 1 chấm trên 3.269 nghiên cứu, giai đoạn 2 trên 2.800 nghiên cứu có phần FINDINGS hợp lệ. Bài báo không công bố khoảng tin cậy nên cột chênh lệch là hiệu của hai điểm ước lượng.

Hai điều kiện đi kèm bảng trên phải được nêu cùng với nó. Một là, số phân loại của đề tài dùng quy tắc hai ngưỡng theo từng bệnh, không phải quy tắc argmax của bài báo; với argmax, cùng checkpoint đạt F1 có trọng số 0,780 và F1 năm bệnh 0,669. Hai là, mô hình sinh báo cáo của đề tài nhận thêm chỉ định chụp và phần FINDINGS của lần chụp trước, là thông tin bài báo không dùng; vì vậy phần cao hơn bài báo ở các chỉ số sinh ngôn ngữ không chứng minh rằng kênh ảnh của đề tài tốt hơn.

# Hai checkpoint được báo cáo

## Checkpoint giai đoạn 1: phân loại bất thường

Checkpoint giai đoạn 1 có tên run_20261005_paper, hoàn thành ngày 06 tháng 10 năm 2026. Đây là mô hình biểu diễn và phân loại: ba bộ mã hóa ảnh, khối META-Former sinh 32 token ảnh cho mô hình ngôn ngữ, và khối MHCAC phân loại 14 bất thường theo ba lớp âm tính, dương tính và không chắc chắn. Tên gọi có chữ paper vì các tham số của MHCAC được đặt đúng theo bài báo: 8 expert token chung và không làm mượt nhãn. File được dùng là checkpoint tốt nhất của pha 1c, dung lượng 1,37 GB; mọi kết quả giai đoạn 1 và toàn bộ đầu vào của giai đoạn 2 đều lấy từ file này.

::table Kiến trúc và dữ liệu của checkpoint giai đoạn 1
::widths 4.3; 12.2
::font 10
| Thành phần | Cấu hình |
|---|---|
| Dữ liệu | MIMIC-CXR-JPG, tập chia chính thức; mỗi mẫu là một nghiên cứu; kiểm định 1.808 và kiểm tra 3.269 nghiên cứu |
| Ảnh đầu vào | Thu về cạnh ngắn 512 rồi cắt giữa 448 x 448; một ảnh neo ưu tiên PA, AP rồi lateral, kèm tối đa một ảnh phụ |
| Bộ mã hóa ảnh | BioViL-T ResNet50 (196 token), PubMedCLIP ViT (50 token), MedCLIP Swin-Tiny (50 token); chiếu về 1408 chiều, nối thành 296 token; che ngẫu nhiên 10% đặc trưng |
| META-Former | Q-Former khởi tạo từ BLIP-2, 32 truy vấn, cross-attention mỗi hai khối |
| MHCAC | 6 tầng, 8 expert token chung và 14 expert token chuyên biệt, dropout 0,2, không làm mượt nhãn |
| Nhãn | CheXpert ba lớp; ô trống coi là âm tính; nghiên cứu không có thông tin nhãn bị loại khỏi hàm mất mát phân loại |
| Hàm mất mát phân loại | Cross-entropy có trọng số theo tần suất lớp (chặn trên 10), cộng contrastive 0,3, orthogonality 0,7, sparsity 0,3 |
| Hàm mất mát đa góc chụp | Đối chiếu đa dương tính 0,02 và nhất quán giữa các góc chụp 0,05 |
| Thiết lập chung | AdamW, suy giảm trọng số 0,02, cắt gradient 1,0, độ chính xác hỗn hợp bf16, hạt giống 42, một GPU RTX 5060 Ti 16 GB |

Việc huấn luyện đi theo ba pha như bài báo mô tả. Pha 1a căn chỉnh ảnh và văn bản cho META-Former. Pha 1b chuyển trọng tâm sang MHCAC: trong 10% bước đầu META-Former còn học cùng, sau đó bị đóng băng. Pha 1c huấn luyện chung phân loại và căn chỉnh, đồng thời mở khóa một phần bộ mã hóa ảnh.

::table Cấu hình huấn luyện theo từng pha của checkpoint giai đoạn 1
::widths 3.4; 4.3; 4.4; 4.4
::font 9.5
| Hạng mục | Pha 1a | Pha 1b | Pha 1c |
|---|---|---|---|
| Mục tiêu | ITC, ITM và sinh văn bản, trọng số 1 mỗi loại | Phân loại và các hàm mất mát của MHCAC, đa góc chụp | Phân loại cộng căn chỉnh, trọng số căn chỉnh 1,0 |
| Số epoch | 4 | 5 | 1 |
| Kích thước lô | 128 bằng GradCache trên đặc trưng đã lưu | 16, tích lũy 4 (hiệu dụng 64) | 8, tích lũy 8 (hiệu dụng 64) |
| Tốc độ học | 1e-4 giảm về 1e-5 | Khởi động tới 2e-4 rồi giảm về 1e-5 | 5e-5 giảm về 1e-5; bộ mã hóa 1e-5 |
| Tham số được học | Q-Former, truy vấn, các lớp chiếu | 310,9 triệu ở 10% đầu, sau đó 122,0 triệu (MHCAC 80,6 triệu, hợp nhất góc chụp 38,0 triệu) | 342,8 triệu, gồm 31,9 triệu của bộ mã hóa (ResNet50 layer4, hai khối cuối của CLIP) |
| Chọn checkpoint | Theo hàm mất mát kiểm định | Macro recall ba lớp trên tập kiểm định; tốt nhất ở epoch thứ ba | Macro recall ba lớp trên tập kiểm định |
| Thời gian | Dùng lại từ lần chạy trước | 7 giờ 08 phút | Khoảng 4 giờ 40 phút |

Pha 1a không được huấn luyện lại cho lần chạy này mà dùng checkpoint pha 1a của lần chạy ba lớp trước đó (run_20260930_3class), vì pha này không huấn luyện MHCAC nên không phụ thuộc vào hai tham số được đổi theo bài báo. Ở pha 1a, hàm đối chiếu ITC dùng dạng sigmoid theo SigLIP thay cho softmax.

Quy tắc quyết định là một phần của checkpoint được báo cáo. Với mỗi bệnh, đề tài tính điểm s bằng xác suất dương tính chia cho tổng xác suất dương tính và âm tính, rồi dùng hai ngưỡng t1 và t2: dưới t1 là âm tính, từ t2 trở lên là dương tính, ở giữa là không chắc chắn. Hai ngưỡng được chọn trên tập kiểm định 1.808 nghiên cứu để tối đa F1 có trọng số của từng bệnh, và được lưu trong file configs/stage1_cutpoints/run_20261005_paper.json. Bốn bệnh có vùng không chắc chắn là Atelectasis, Consolidation, Edema và Pneumonia. Bài toán vẫn là ba lớp; tập kiểm tra không tham gia vào việc chọn ngưỡng.

## Checkpoint giai đoạn 2: sinh báo cáo

Checkpoint giai đoạn 2 là một bộ adapter QLoRA cho mô hình ngôn ngữ MedGemma 1.5 4B, kèm lớp chiếu đưa 32 token của META-Former vào không gian của mô hình ngôn ngữ. Nó được huấn luyện từ ngày 12 đến ngày 14 tháng 10 năm 2026, trong 47 giờ 45 phút, không gặp lỗi. Mô hình ngôn ngữ không nhìn ảnh gốc: thông tin hình ảnh chỉ đến qua 32 token của checkpoint giai đoạn 1, đúng như thiết kế của bài báo.

::table Cấu hình của checkpoint giai đoạn 2
::widths 4.3; 12.2
::font 10
| Thành phần | Cấu hình |
|---|---|
| Mô hình ngôn ngữ | google/medgemma-1.5-4b-it, lượng tử hóa 4 bit NF4 |
| Adapter | LoRA hạng 8, alpha 16, chỉ đặt ở tháp ngôn ngữ; 16,87 triệu tham số được học (LoRA 14,90 triệu và lớp chiếu 1,97 triệu), bằng 0,67% tổng số; tháp ảnh của MedGemma không được học và không được dùng |
| Đầu vào hình ảnh | 32 token của META-Former từ checkpoint giai đoạn 1, qua một lớp chiếu tuyến tính |
| Lời nhắc | Câu lệnh nguyên văn của bài báo và danh sách bệnh dương tính, âm tính, không chắc chắn; thêm dòng chỉ định chụp (tối đa 60 từ) và phần FINDINGS của lần chụp gần nhất trước đó của cùng bệnh nhân (tối đa 150 từ, kèm số ngày cách) |
| Danh sách bệnh lúc huấn luyện | 80% mẫu lấy từ nhãn CheXpert thật, 20% giữ dự đoán của MHCAC |
| Danh sách bệnh lúc kiểm định và kiểm tra | Luôn là dự đoán của MHCAC theo quy tắc hai ngưỡng |
| Văn bản đích | Chỉ phần FINDINGS, đã bỏ dòng tiêu đề phiếu chụp; 170.370 mẫu huấn luyện và 1.360 mẫu kiểm định; tối đa 768 token |
| Tối ưu | 2 epoch; lô 2, tích lũy 8 (hiệu dụng 16); tốc độ học 3e-4 cho LoRA và 1e-3 cho lớp chiếu; khởi động 3%; suy giảm trọng số 0,01; cắt gradient 1,0 |
| Chọn checkpoint | Hàm mất mát kiểm định: 0,99053 sau epoch 1 và 0,95619 sau epoch 2; giữ epoch 2 |
| Giải mã | Tìm kiếm chùm 4, hệ số độ dài 2,0, tối đa 256 token mới; khoảng 3,7 giây mỗi nghiên cứu |

Trong 2.800 nghiên cứu của tập kiểm tra, 2.566 nghiên cứu có chỉ định chụp và 2.508 nghiên cứu có lần chụp trước. Hàm mất mát huấn luyện giảm từ 1,024 xuống 0,855 trong khi hàm mất mát kiểm định giảm từ 0,991 xuống 0,956; epoch thứ hai vẫn có lợi nhưng khoảng cách giữa hai đường đã mở ra.

# Kết quả theo các phép đo của bài báo

## Phân loại bất thường

Phép đo theo bài báo: với mỗi bệnh tính precision, recall và F1 có trọng số theo số mẫu của ba lớp, rồi lấy trung bình trên 14 bệnh; F1 năm bệnh là trung bình trên Atelectasis, Cardiomegaly, Consolidation, Edema và Pleural Effusion. Tập kiểm tra gồm 3.269 nghiên cứu.

::table Kết quả phân loại của checkpoint giai đoạn 1 trên tập kiểm tra
::widths 4.2; 2.4; 5.3; 4.6
::font 10
| Chỉ số | Bài báo | Đề tài, hai ngưỡng (số chính) | Đề tài, argmax (giao thức bài báo) |
|---|---|---|---|
| Precision có trọng số | 0,87 | 0,827 [0,823; 0,832] | 0,839 |
| Recall có trọng số | 0,78 | 0,834 [0,830; 0,838] | 0,781 |
| F1 có trọng số | 0,73 | 0,820 [0,815; 0,824] | 0,780 |
| F1 trung bình 5 bệnh | 0,701 | 0,749 [0,741; 0,757] | 0,669 |
| AUROC lớp dương tính | chỉ nêu vài bệnh | 0,784 | 0,784 |
| AUROC lớp âm tính | chỉ nêu vài bệnh | 0,778 | 0,778 |
| AUROC lớp không chắc chắn | chỉ nêu vài bệnh | 0,663 | 0,663 |
::note Khoảng tin cậy 95% bằng bootstrap 1.000 lần. Hai cột của đề tài là cùng một checkpoint với hai quy tắc quyết định; AUROC không phụ thuộc quy tắc quyết định. Cột hai ngưỡng không phải giao thức của bài báo.

Số chính của đề tài là cột hai ngưỡng vì đó là kết quả cao nhất của checkpoint này và là quy tắc được dùng để tạo danh sách bệnh cho giai đoạn 2. Cột argmax được giữ lại vì đó là cách bài báo báo cáo.

::table AUROC theo bệnh mà bài báo nêu trong phần chữ, so với checkpoint giai đoạn 1
::widths 7.5; 4.5; 4.5
::font 10.5
| Bệnh và lớp | Bài báo | Đề tài |
|---|---|---|
| Support Devices, dương tính | 0,92 | 0,904 |
| Support Devices, âm tính | 0,84 | 0,903 |
| Pleural Effusion, dương tính | 0,74 | 0,893 |
| Pleural Effusion, âm tính | 0,85 | 0,893 |
| Edema, âm tính | 0,78 | 0,838 |

## Sinh báo cáo

Các chỉ số được tính theo cách phổ biến mà bài báo dùng: chữ thường, tách từ, bộ công cụ pycocoevalcap; BERTScore dùng distilroberta-base có chuẩn hóa theo đường cơ sở. Tập chấm gồm 2.800 nghiên cứu của tập kiểm tra có phần FINDINGS hợp lệ.

::table Chỉ số sinh ngôn ngữ của checkpoint giai đoạn 2 trên tập kiểm tra
::widths 4.5; 3.0; 3.0; 3.0; 3.0
::font 10.5
| Chỉ số | Bài báo | Đề tài | Chênh lệch | Tỉ lệ so với bài báo |
|---|---|---|---|---|
| BLEU-1 | 0,390 | 0,434 | +0,044 | 111% |
| BLEU-2 | không rõ | 0,308 | không có | không có |
| BLEU-3 | không rõ | 0,237 | không có | không có |
| BLEU-4 | 0,102 | 0,190 | +0,088 | 186% |
| METEOR | 0,173 | 0,190 | +0,017 | 110% |
| ROUGE-L | 0,280 | 0,355 | +0,075 | 127% |
| CIDEr | 0,291 | 0,416 | +0,125 | 143% |
| BERTScore | 0,426 | 0,466 | +0,040 | 109% |
::note Không rõ: bảng của bài báo là ảnh và hai số này không được nhắc trong phần chữ. Báo cáo sinh ra có độ dài trung vị 47 từ.

Bảng này không phải so sánh cùng điều kiện. Mô hình của đề tài đọc thêm chỉ định chụp và báo cáo của lần chụp trước; mô hình của bài báo chỉ đọc 32 token ảnh và danh sách bệnh.

## Hiệu quả lâm sàng

Hiệu quả lâm sàng đo việc báo cáo sinh ra có nêu đúng 14 bất thường như báo cáo của bác sĩ hay không. Bài báo dùng bộ gán nhãn CheXpert; đề tài dùng CheXbert, bộ gán nhãn thông dụng trong các công trình gần đây.

::table Chỉ số lâm sàng của checkpoint giai đoạn 2 trên tập kiểm tra
::widths 6.5; 3.3; 3.3; 3.4
::font 10.5
| Chỉ số | Bài báo | Đề tài | Chênh lệch |
|---|---|---|---|
| Precision (macro, 14 bệnh) | 0,411 | 0,517 | +0,106 |
| Recall (macro, 14 bệnh) | 0,465 | 0,398 | -0,067 |
| F1 (macro, 14 bệnh) | 0,428 | 0,425 | -0,003 |
| F1 (micro, 14 bệnh) | không báo | 0,564 | không có |
| F1 (macro, 5 bệnh phổ biến) | không báo | 0,551 | không có |
| RadGraph F1 | 0,224 | chưa đo | không có |
| RadCliQ (thấp hơn là tốt hơn) | 1,211 | chưa đo | không có |
::note Hai bên dùng hai bộ gán nhãn khác nhau nên F1 0,425 so với 0,428 chỉ là so sánh gần đúng. Chưa đo nghĩa là chưa cài công cụ, không phải bằng 0.

## Các phép đo của bài báo chưa thực hiện

::table Các bảng và hình của bài báo mà đề tài chưa tái lập
::widths 6.0; 4.0; 6.5
::font 10
| Phép đo của bài báo | Số của bài báo | Tình trạng ở đề tài |
|---|---|---|
| IU-Xray, ngoài phân phối (Bảng 2) | BLEU-4 0,121; ROUGE-L 0,287 | Chưa chạy |
| CheXpert, 200 ảnh kiểm định (Bảng 4) | F1 trung bình 0,699 | Có mã chấm điểm, chưa có bộ nạp dữ liệu |
| Bỏ từng bộ mã hóa (Bảng 5 và 6) | Bảng là ảnh, không trích được | Chưa chạy trên checkpoint này |
| Bỏ từng hàm mất mát của MHCAC (Bảng 7) | F1 5 bệnh 0,701 khi đủ | Chưa chạy; mỗi dòng là một lần huấn luyện |
| Chi phí suy luận (Bảng 9) | Khoảng 3,5 giây; 14,9 GB | Riêng bước sinh: 3,7 giây, đỉnh 8,9 GiB; không cùng phạm vi đo |
| Chuyên gia chấm (Hình 12) | 86,75% đến 88,75% | Không có đánh giá của bác sĩ |

# Sai khác so với bài báo

## Sai khác ở giai đoạn 1

::table Sai khác của checkpoint giai đoạn 1 so với bài báo và lý do
::widths 2.6; 3.3; 4.6; 6.0
::font 9.5
| Hạng mục | Bài báo | Đề tài | Lý do |
|---|---|---|---|
| Quy tắc quyết định | argmax ba lớp | Hai ngưỡng theo từng bệnh, chọn trên tập kiểm định | argmax gần như không bao giờ chọn lớp không chắc chắn và bỏ sót dương tính; đổi quy tắc làm F1 5 bệnh tăng 0,081 trên cùng checkpoint |
| Góc chụp | Chỉ ảnh frontal | Một ảnh neo và tối đa một ảnh phụ, có khối hợp nhất góc chụp và hai hàm mất mát đa góc | Hơn một nửa số nghiên cứu huấn luyện có nhiều ảnh; đóng góp riêng chưa được tách |
| Hàm đối chiếu ở pha 1a | ITC của BLIP-2, không nêu kích thước lô | ITC dạng sigmoid, lô 128 bằng GradCache | Trên GPU 16 GB, ITC với lô 8 cho kết quả ngẫu nhiên trong cả bốn lần đo; cách này lần đầu vượt ngẫu nhiên |
| Độ dài huấn luyện | 20.000 bước | Theo epoch: 4, 5 và 1 | Bài báo không nêu kích thước lô nên số bước không quy đổi được |
| Mở khóa ở pha 1c | Lớp chiếu của bộ mã hóa | Thêm ResNet50 layer4 và hai khối cuối của CLIP | Cho bộ mã hóa thích nghi với nhiệm vụ; đóng góp riêng chưa được tách |
| Token PubMedCLIP | Không nêu chi tiết | Đọc sau lớp chuẩn hóa cuối, trừ trung bình theo ảnh, tiền xử lý riêng | Token thô gần trùng nhau (độ tương đồng 0,674) nên attention gần như phẳng |
| Chọn checkpoint | Không nêu | Macro recall ba lớp trên tập kiểm định | Hàm mất mát bị lớp đa số chi phối |
| Nghiên cứu không có nhãn | Mọi ô trống là âm tính | Như bài báo, nhưng loại nghiên cứu không có thông tin nhãn nào | Tránh biến nghiên cứu không nhãn thành 14 ô âm tính |

Các phần giống bài báo: ba họ bộ mã hóa, 296 token ở 1408 chiều, che 10% đặc trưng, META-Former khởi tạo từ BLIP-2 với ba mục tiêu căn chỉnh, MHCAC 6 tầng với 8 và 14 expert token, dropout 0,2, không làm mượt nhãn, bốn hàm mất mát của MHCAC, ba lớp nhãn, lịch ba pha và dải tốc độ học của pha 1b.

## Sai khác ở giai đoạn 2

::table Sai khác của checkpoint giai đoạn 2 so với bài báo và lý do
::widths 2.6; 3.5; 4.4; 6.0
::font 9.5
| Hạng mục | Bài báo | Đề tài | Lý do |
|---|---|---|---|
| Mô hình ngôn ngữ | Vicuna-7B, lượng tử hóa 8 bit, LoRA khởi tạo từ RaDialog đã tinh chỉnh trên MIMIC-CXR | MedGemma 1.5 4B, 4 bit NF4, LoRA huấn luyện từ đầu | Giới hạn 16 GB bộ nhớ GPU và giấy phép sử dụng |
| Bối cảnh trong lời nhắc | Không có | Chỉ định chụp và FINDINGS của lần chụp trước | 90,0% báo cáo kiểm tra có ngôn ngữ so sánh theo thời gian; thử nghiệm nhỏ cho ROUGE-L +0,036 [+0,023; +0,049] |
| Danh sách bệnh lúc huấn luyện | Dự đoán của MHCAC | 80% từ nhãn thật, 20% từ MHCAC | Mô hình huấn luyện kiểu bài báo bỏ qua danh sách bệnh: thay bằng nhãn thật không đổi chỉ số nào |
| Ngưỡng cho danh sách bệnh | Ngưỡng theo đường ROC (công thức 22), không nêu tập dùng để chọn | Hai ngưỡng của giai đoạn 1 | Bài báo không công bố file ngưỡng; công thức 22 chọn trên tập kiểm định làm F1 có trọng số còn 0,702 |
| Độ dài huấn luyện | 5.000 bước | 2 epoch, khoảng 21.300 bước cập nhật | Hàm mất mát còn giảm sau một epoch; tốc độ học 3e-4 đạt tiêu chí trong thử nghiệm nhỏ |
| Giải mã | Không nêu | Tìm kiếm chùm 4, hệ số độ dài 2,0 | Giải mã tham lam viết ngắn hơn báo cáo của bác sĩ |
| Văn bản đích | Không mô tả cách lấy FINDINGS | Bỏ dòng tiêu đề phiếu chụp | 9,8% văn bản đích bắt đầu bằng dòng tiêu đề và mô hình học lại dòng đó |

Các phần giống bài báo: mô hình ngôn ngữ chỉ nhận 32 token của META-Former qua một lớp chiếu, lời nhắc gồm truy vấn và danh sách bệnh ba lớp, LoRA hạng 8 và alpha 16, chỉ lớp chiếu và LoRA được học.

## Sai khác ở cách đánh giá

::table Sai khác về cách đánh giá và lý do
::widths 3.0; 4.0; 4.3; 5.2
::font 9.5
| Hạng mục | Bài báo | Đề tài | Lý do |
|---|---|---|---|
| Tập chấm giai đoạn 2 | 3.269 | 2.800 nghiên cứu có FINDINGS hợp lệ | Nghiên cứu không có FINDINGS không có văn bản tham chiếu để chấm |
| BERTScore | 0,426, không nêu cấu hình | distilroberta-base, chuẩn hóa theo đường cơ sở | Cấu hình của bộ công cụ mà bài báo trích dẫn; BERTScore thô ở thang khác |
| Hiệu quả lâm sàng | Bộ gán nhãn CheXpert | CheXbert, dương tính hoặc không chắc chắn tính là 1 | Bộ gán nhãn CheXpert không có trong kho mã |
| RadGraph, RadCliQ | Có | Chưa đo | Công cụ có giấy phép riêng, chưa cài |
| Khoảng tin cậy | Không có | Bootstrap theo nghiên cứu | Để biết chênh lệch có vượt nhiễu lấy mẫu không |

# Vì sao kết quả chênh với bài báo

## Phân loại

F1 có trọng số của đề tài cao hơn bài báo ở cả hai quy tắc quyết định (0,820 và 0,780 so với 0,73). Chỉ số này lấy trung bình 14 bệnh và có trọng số theo tần suất lớp, nên chủ yếu phản ánh việc đoán đúng lớp âm tính, vốn chiếm phần lớn ở các bệnh hiếm. Bài báo có precision 0,87 và recall 0,78 nhưng F1 chỉ 0,73; đề tài không xác định được nguyên nhân vì bài báo không công bố trọng số lớp hay dự đoán. Vì vậy phần cao hơn này không được coi là bằng chứng rằng bộ phân loại của đề tài mạnh hơn.

F1 năm bệnh là thước đo sát hơn vì năm bệnh này có nhiều mẫu dương tính và không chắc chắn. Với argmax, đề tài thấp hơn bài báo 0,032 (0,669 so với 0,701). Nguyên nhân có bằng chứng trực tiếp là điểm cắt của argmax: khi chuyển sang hai ngưỡng trên cùng checkpoint, AUROC không đổi nhưng F1 năm bệnh tăng 0,0805 với khoảng tin cậy [+0,0743; +0,0866]. Nói cách khác, khả năng xếp hạng đã có sẵn, argmax chỉ chọn điểm cắt kém. Cái giá của hai ngưỡng là macro recall giảm 0,0149.

Mức tuyệt đối còn thấp do hai giới hạn đã đo được. Một là, nhãn CheXpert tự động có nhiễu: so với nhãn của bác sĩ trên 685 nghiên cứu kiểm tra, độ chính xác của nhãn dương tính chỉ từ 0,42 đến 0,46 ở bốn bệnh, và cùng checkpoint khi chấm bằng nhãn bác sĩ có AUROC dương tính cao hơn 0,022. Hai là, lớp không chắc chắn có rất ít mẫu và AUROC chỉ 0,663. Hai giới hạn này áp dụng cho cả bài báo nên chúng giải thích mức tuyệt đối, không giải thích khoảng cách với bài báo.

## Sinh báo cáo

Phần cao hơn bài báo ở mọi chỉ số sinh ngôn ngữ đến chủ yếu từ thông tin đầu vào mà bài báo không có. Có ba bằng chứng. Thứ nhất, 90,0% báo cáo của tập kiểm tra dùng ngôn ngữ so sánh với lần chụp trước, và 89,6% nghiên cứu kiểm tra có lần chụp trước; không có báo cáo cũ thì mô hình không thể viết đúng các câu so sánh. Thứ hai, chỉ cần chép nguyên phần FINDINGS của lần chụp trước, không huấn luyện gì, đã đạt BLEU-1 0,391 và BLEU-4 0,106, ngang với bài báo. Thứ ba, trong thử nghiệm nhỏ chỉ thay đổi một yếu tố là thêm bối cảnh, ROUGE-L tăng 0,036 [+0,023; +0,049] và BERTScore tăng 0,028 [+0,014; +0,041].

BLEU-4 và CIDEr tăng mạnh nhất (bằng 186% và 143% của bài báo) vì hai chỉ số này thưởng các cụm từ dài và các cụm từ hiếm; báo cáo lần trước cung cấp đúng những cụm từ mô tả tình trạng riêng của bệnh nhân. METEOR và BERTScore tăng ít hơn vì chúng đo mức trùng nghĩa ở cấp từ, vốn đã khá cao khi mô hình viết đúng văn phong báo cáo.

Hai yếu tố còn lại cũng đóng góp nhưng chưa được tách riêng ở lần huấn luyện đầy đủ: huấn luyện với danh sách bệnh lấy từ nhãn thật để mô hình ngôn ngữ học cách dùng danh sách này, và huấn luyện dài hơn với tốc độ học cao hơn. Cả ba thay đổi đi cùng nhau trong một lần chạy duy nhất.

Kênh ảnh của đề tài không mạnh hơn bài báo. 32 token của META-Former trong checkpoint giai đoạn 1 gần như trùng nhau (độ tương đồng cosine trung bình 0,9997 giữa các token của một nghiên cứu), nên mô hình ngôn ngữ thực chất chỉ nhận một vector ảnh. Vector này giữ được thông tin của các bệnh phổ biến nhưng mất các bệnh khu trú và hiếm. Khi đầu vào giống hệt bài báo, tức không có bối cảnh, phiên bản trước của giai đoạn 2 thấp hơn bài báo ở mọi chỉ số (BLEU-4 0,098; CIDEr 0,128; BERTScore 0,369). Đây là lý do không được viết rằng hệ thống vượt bài báo mà không kèm điều kiện về đầu vào.

## Hiệu quả lâm sàng

F1 lâm sàng ngang bài báo (0,425 so với 0,428) nhưng cấu trúc khác: precision cao hơn 0,106 và recall thấp hơn 0,067. Mô hình của đề tài nêu ít bất thường hơn, nhưng khi nêu thì đúng hơn. Recall thấp là do mô hình viết thiếu: báo cáo sinh ra có độ dài trung vị 47 từ so với khoảng 56 từ của bác sĩ, và các bệnh hiếm ít khi được nhắc vì vector ảnh không còn mang thông tin về chúng. Precision cao là do báo cáo lần trước và danh sách bệnh theo hai ngưỡng cho mô hình căn cứ chắc hơn khi khẳng định một bất thường. Vì hai bên dùng hai bộ gán nhãn khác nhau, chênh lệch 0,003 không có ý nghĩa theo hướng nào.

## Tóm tắt nguyên nhân

::table Nguyên nhân chính của từng chênh lệch so với bài báo
::widths 4.6; 2.0; 5.6; 4.3
::font 9.5
| Chênh lệch | Hướng | Nguyên nhân chính | Mức bằng chứng |
|---|---|---|---|
| F1 có trọng số 0,820 so với 0,73 | Cao hơn | Chỉ số bị lớp đa số chi phối; quy tắc hai ngưỡng | Không giải thích được đầy đủ vì thiếu thông tin từ bài báo |
| F1 5 bệnh 0,749 so với 0,701 | Cao hơn | Quy tắc hai ngưỡng chọn trên tập kiểm định | Đo trực tiếp; không cùng giao thức |
| F1 5 bệnh với argmax 0,669 so với 0,701 | Thấp hơn | Điểm cắt của argmax | Đo trực tiếp trên cùng checkpoint |
| Chỉ số sinh ngôn ngữ | Cao hơn | Thêm chỉ định chụp và báo cáo lần chụp trước | Thử nghiệm tách riêng và mốc chép báo cáo cũ |
| F1 lâm sàng 0,425 so với 0,428 | Ngang | Precision cao hơn, recall thấp hơn; bộ gán nhãn khác | So sánh gần đúng |

# Giới hạn

Mỗi checkpoint là kết quả của một lần chạy với một hạt giống; bài báo cũng không nêu số lần chạy. Ba thay đổi của giai đoạn 2 chưa được tách riêng ở lần huấn luyện đầy đủ. Giai đoạn 2 chấm trên 2.800 nghiên cứu trong khi bài báo ghi 3.269. Hiệu quả lâm sàng dùng CheXbert trong khi bài báo dùng bộ gán nhãn CheXpert. Bài báo không nêu cấu hình BERTScore, cách giải mã, kích thước lô và tập dùng để chọn ngưỡng. Các bảng trong bài báo là ảnh nên chỉ trích được những số được nhắc trong phần chữ. RadGraph, RadCliQ, IU-Xray và CheXpert kiểm định chưa được đo. Nhãn CheXpert tự động vừa dùng để huấn luyện vừa dùng để chấm và có nhiễu.

::check abs((0.434 - 0.390) - 0.044) < 1e-9 | sai hiệu BLEU-1
::check abs((0.190 - 0.102) - 0.088) < 1e-9 | sai hiệu BLEU-4
::check abs((0.355 - 0.280) - 0.075) < 1e-9 | sai hiệu ROUGE-L
::check abs((0.416 - 0.291) - 0.125) < 1e-9 | sai hiệu CIDEr
::check abs((0.466 - 0.426) - 0.040) < 1e-9 | sai hiệu BERTScore
::check round(0.190 / 0.102 * 100) == 186 and round(0.416 / 0.291 * 100) == 143 | sai tỉ lệ BLEU-4 hoặc CIDEr
::check round(0.434 / 0.390 * 100) == 111 and round(0.190 / 0.173 * 100) == 110 and round(0.355 / 0.280 * 100) == 127 and round(0.466 / 0.426 * 100) == 109 | sai tỉ lệ
::check abs((0.517 - 0.411) - 0.106) < 1e-9 and abs((0.465 - 0.398) - 0.067) < 1e-9 | sai hiệu CE
