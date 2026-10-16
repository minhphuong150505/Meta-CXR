---
title: Các độ đo đánh giá sử dụng trong đề tài
subtitle: Mô tả độ đo, cách tính và lý do lựa chọn cho hai giai đoạn của pipeline META-CXR
data: so_lieu.json
chapter_new_page: false
---

::heading Phạm vi tài liệu
Tài liệu này chỉ mô tả các độ đo mà đề tài dùng để đánh giá hai giai đoạn của pipeline: Giai đoạn 1 phân
loại 14 bất thường trên ảnh X-quang ngực, và Giai đoạn 2 sinh phần FINDINGS của báo cáo. Với mỗi độ đo,
tài liệu nêu đại lượng mà độ đo phản ánh, cách tính trong đề tài, lý do lựa chọn, và đối chiếu với bài báo
META-CXR gốc [1]. Kết quả số chỉ được dùng khi cần minh họa lý do lựa chọn; kết quả đầy đủ nằm trong tài
liệu đánh giá thực nghiệm riêng.

# Độ đo cho Giai đoạn 1: phân loại bất thường

## Cách diễn giải nhãn trước khi tính độ đo
Mỗi ô nhãn CheXpert có bốn trạng thái: dương tính, âm tính, không chắc chắn, và để trống khi báo cáo không
nhắc tới bất thường đó [2]. Trên MIMIC-CXR-JPG, {{data.blank_fraction_raw|pct1}} số ô để trống. Vì vậy trước
khi tính bất kỳ độ đo nào phải chọn câu hỏi mà độ đo trả lời, và đề tài ghi rõ lựa chọn này trong mọi tệp kết quả.

Cách thứ nhất, gọi là phân cực có che (masked polarity), bỏ qua ô trống và chỉ hỏi phân cực của bất thường khi
báo cáo đã nhắc tới nó. Đây là mẫu số đúng cho hàm mất mát phân loại ba lớp, nhưng là mẫu số sai cho F1: khi chỉ
xét các ô được nhắc tới, dương tính chiếm đa số ở phần lớn bất thường, nên một bộ phân loại hằng số cũng đạt F1
cao. Cách thứ hai, gọi là hiện diện trong nghiên cứu (study presence), hỏi câu hỏi lâm sàng: nghiên cứu này có
bất thường X hay không; ô trống, âm tính và không chắc chắn đều được coi là không có. Cách này trùng với chính
sách U-Zeros của CheXpert [2] và với cách CheXclusion dùng khi đánh giá trên MIMIC-CXR [3].

::table F1 dương tính macro của mô hình và của các bộ phân loại hằng số dưới hai cách diễn giải (tập kiểm tra, {{s1.n|n}} nghiên cứu)
::widths 7.4, 4.5, 4.5
| Bộ phân loại | Phân cực có che | Hiện diện trong nghiên cứu |
|---|---|---|
| Mô hình của đề tài | {{framing.masked_model_f1|f4}} | {{framing.presence_model_f1|f4}} |
| Luôn dự đoán dương tính | {{framing.masked_all_positive_f1|f4}} | {{framing.presence_all_positive_f1|f4}} |
| Dự đoán lớp đa số của từng nhãn | {{framing.masked_majority_f1|f4}} | {{framing.presence_majority_f1|f4}} |
::note Cùng một tệp dự đoán, chỉ khác cách diễn giải nhãn. Ngưỡng được hiệu chuẩn trên tập kiểm định với cùng cách diễn giải.

Dưới phân cực có che, bộ phân loại luôn dự đoán dương tính đạt {{framing.masked_all_positive_f1|f4}}, chỉ thấp
hơn mô hình {{framing.masked_model_f1 - framing.masked_all_positive_f1|f4}}. Dưới hiện diện trong nghiên cứu, cả
hai bộ phân loại hằng số rơi xuống mức thấp và khoảng cách với mô hình phản ánh đúng năng lực của mô hình. Đề tài
vì vậy chỉ báo cáo F1, độ chính xác, độ nhạy và độ đặc hiệu dưới cách diễn giải hiện diện trong nghiên cứu.
::check D["framing"]["masked_all_positive_f1"] > 0.8 and D["framing"]["presence_all_positive_f1"] < 0.3 | Bảng cách diễn giải không còn minh họa được lạm phát F1
::check D["framing"]["masked_model_f1"] - D["framing"]["masked_all_positive_f1"] < 0.06 | Khoảng cách dưới phân cực có che không còn nhỏ

Bài báo gốc coi ô trống là âm tính (mục III-A của [1]) và báo cáo độ chính xác, độ nhạy, F1 trung bình cho bài
toán ba lớp dương tính, âm tính, không chắc chắn. Bài báo không nêu kiểu trung bình trong bảng kết quả chính,
còn nghiên cứu cắt bỏ số token chuyên gia ghi là trung bình có trọng số (mục IV-D.4 của [1]). Vì cách diễn giải
nhãn và cách lấy trung bình khác nhau, các con số F1 của hai công trình không so sánh trực tiếp được.

## Diện tích dưới đường cong ROC
AUROC đo xác suất mô hình xếp một nghiên cứu có bất thường cao hơn một nghiên cứu không có bất thường, trên mọi
ngưỡng quyết định [4], [5]. Đề tài tính AUROC riêng cho từng nhãn rồi báo cáo hai dạng tổng hợp. AUROC macro là
trung bình AUROC của các nhãn, mỗi nhãn một trọng số như nhau, nên phản ánh khả năng phân biệt của từng bất
thường kể cả bất thường hiếm. AUROC micro gộp mọi cặp nghiên cứu và nhãn vào một bảng xếp hạng chung, nên nhạy với
việc điểm số của các nhãn khác nhau có cùng thang đo hay không.

Lý do chọn AUROC làm độ đo chính của Giai đoạn 1 là nó không phụ thuộc ngưỡng và không đổi khi tỉ lệ bệnh thay đổi,
nên tách được năng lực phân biệt khỏi cách chọn điểm hoạt động. Việc báo cáo cả hai dạng là có chủ đích: trong
thực nghiệm của đề tài, hàm mất mát phân cấp làm AUROC micro tăng có ý nghĩa thống kê
({{s1pair.hier_shallow.micro_auroc.delta|sf4}}) trong khi AUROC macro không tăng có ý nghĩa
({{s1pair.hier_shallow.macro_auroc.delta|sf4}}). Chỉ nhìn AUROC micro sẽ nhầm một cải thiện về hiệu chuẩn giữa các
nhãn thành một cải thiện về khả năng phân biệt. Bài báo gốc có vẽ đường cong ROC cho từng lớp nhưng không báo cáo
AUROC tổng hợp trên MIMIC-CXR.
::check D["s1pair"]["hier_shallow"]["micro_auroc"]["sig"] and not D["s1pair"]["hier_shallow"]["macro_auroc"]["sig"] | Ví dụ macro và micro không còn đúng

## Diện tích dưới đường cong độ chính xác và độ nhạy
AUPRC macro là trung bình diện tích dưới đường cong độ chính xác theo độ nhạy của các nhãn. Khi lớp dương tính
hiếm, AUROC có thể cao dù phần lớn các dự đoán dương tính là sai, còn đường cong độ chính xác và độ nhạy phản ánh
trực tiếp chi phí của các báo động giả [6]. Dưới cách diễn giải hiện diện trong nghiên cứu, tỉ lệ dương tính của
các nhãn chỉ khoảng 2% đến 34%, nên AUPRC là độ đo bổ sung cần thiết cho AUROC. Giá trị tham chiếu của AUPRC với
một bộ phân loại ngẫu nhiên bằng tỉ lệ dương tính của nhãn, nên AUPRC luôn được đọc cùng tỉ lệ này.

## Độ chính xác, độ nhạy, F1 và độ đặc hiệu tại ngưỡng đã hiệu chuẩn
Các độ đo phụ thuộc ngưỡng mô tả điểm hoạt động thực tế khi mô hình phải đưa ra quyết định có hoặc không. Đề tài
báo cáo F1 dương tính macro cùng độ chính xác và độ nhạy tương ứng, và độ đặc hiệu macro để thấy mô hình có đang
đổi độ đặc hiệu lấy độ nhạy hay không. Hai điểm trong cách tính quyết định giá trị các độ đo này.

Thứ nhất, ngưỡng của từng nhãn chỉ được chọn trên tập kiểm định rồi áp dụng một lần lên tập kiểm tra. Đề tài chọn
ngưỡng tối đa hóa F1 theo hướng dẫn của Lipton và cộng sự [7], nhưng thay vì đứng tại đỉnh, ngưỡng được đặt ở trung
vị của vùng các ngưỡng đạt ít nhất 95% F1 cực đại, vì đỉnh trên 1.808 nghiên cứu kiểm định chịu ảnh hưởng của vài
nghiên cứu tình cờ nằm quanh nó. Nhãn cần ít nhất 5 mẫu dương tính trong tập kiểm định để được hiệu chuẩn; với
ngưỡng tối thiểu 20, hai nhãn hiếm bị gán ngưỡng cố định 0,5, và việc hạ ngưỡng tối thiểu xuống 5 làm F1 macro
thay đổi {{s1pair.minpos.positive_macro_f1.delta|sf4}}, khoảng tin cậy
[{{s1pair.minpos.positive_macro_f1.lo|sf4}}; {{s1pair.minpos.positive_macro_f1.hi|sf4}}].

Thứ hai, điểm số dùng để so với ngưỡng là xác suất hiện diện, bằng xác suất bất thường được nhắc tới nhân với xác
suất dương tính khi được nhắc tới. Cách chấm này tương ứng đúng với câu hỏi hiện diện trong nghiên cứu. Bài báo gốc
chọn ngưỡng tại điểm trên đường cong ROC gần góc trên bên trái nhất (mục V-C của [1]), một biến thể của chỉ số
Youden [8], và không nêu ngưỡng được chọn trên tập nào.
::check D["s1pair"]["minpos"]["positive_macro_f1"]["sig"] and D["s1pair"]["minpos"]["positive_macro_f1"]["delta"] > 0 | Ví dụ ngưỡng tối thiểu không còn có ý nghĩa

## Khoảng tin cậy bootstrap ghép cặp
Mọi so sánh giữa hai cấu hình được thực hiện trên cùng một tập nghiên cứu và báo cáo kèm khoảng tin cậy 95% bằng
bootstrap theo nghiên cứu [9]: mỗi lần lặp lấy mẫu có hoàn lại các nghiên cứu, tính độ đo cho cả hai cấu hình trên
đúng mẫu đó, rồi lấy hiệu. Đề tài dùng 2.000 lần lặp với hạt giống 16. Lấy mẫu ghép cặp loại bỏ phần biến thiên do
độ khó của từng nghiên cứu, nên nhạy hơn nhiều so với việc so hai khoảng tin cậy độc lập. Một khác biệt chỉ được
coi là có ý nghĩa khi khoảng tin cậy không chứa 0 và cận gần 0 nhất cách 0 ít nhất 0,0005; cận nằm sát 0 hơn mức này
có thể đổi dấu chỉ vì nhiễu Monte Carlo. Bài báo gốc báo cáo giá trị điểm mà không kèm khoảng tin cậy.

## Tóm tắt độ đo Giai đoạn 1
::table Các độ đo phân loại và đối chiếu với bài báo gốc
::widths 3.6, 5.6, 3.9, 3.4
::font 10
| Độ đo | Phản ánh | Cách tính trong đề tài | Bài báo gốc |
|---|---|---|---|
| AUROC macro | Khả năng phân biệt của từng bất thường | Hiện diện trong nghiên cứu, không cần ngưỡng | Chỉ vẽ đường cong |
| AUROC micro | Tính nhất quán thang điểm giữa các nhãn | Gộp mọi ô nghiên cứu và nhãn | Không báo cáo |
| AUPRC macro | Chi phí báo động giả khi lớp hiếm | Hiện diện trong nghiên cứu | Không báo cáo |
| F1, độ chính xác, độ nhạy dương tính macro | Điểm hoạt động thực tế | Ngưỡng hiệu chuẩn trên tập kiểm định | Có, cho ba lớp, ô trống là âm tính |
| Độ đặc hiệu macro | Cái giá của độ nhạy | Cùng ngưỡng như trên | Không báo cáo |
| Khoảng tin cậy ghép cặp | Mức chắc chắn của một khác biệt | Bootstrap 2.000 lần theo nghiên cứu | Không báo cáo |

# Độ đo cho Giai đoạn 2: sinh báo cáo

## Giao thức chấm điểm
Các độ đo sinh văn bản rất nhạy với cách tách từ và với cài đặt cụ thể, nên hai con số chỉ so được khi cùng giao
thức. Đề tài chấm theo giao thức chuẩn của lĩnh vực sinh báo cáo X-quang: tách từ bằng PTBTokenizer và tính BLEU,
METEOR, ROUGE-L, CIDEr bằng thư viện pycocoevalcap [10], là bộ công cụ mà R2Gen [11] và nhiều công trình sau dùng
để báo cáo trên tập kiểm tra chính thức của MIMIC-CXR. BERTScore được tính bằng thư viện bert_score với mô hình
RoBERTa-large. Bộ chấm nội bộ của mã nguồn dùng METEOR của nltk và BERTScore với DistilBERT, cho giá trị khác hẳn
giao thức chuẩn, nên chỉ được dùng cho so sánh nội bộ. Bài báo gốc không nêu bộ công cụ chấm điểm.

## BLEU
BLEU-n đo độ chính xác của các n-gram trong báo cáo sinh ra so với báo cáo tham chiếu, kèm hệ số phạt khi báo cáo
ngắn hơn tham chiếu [12]. BLEU-1 phản ánh việc dùng đúng từ vựng, BLEU-4 phản ánh việc tái tạo đúng cụm từ dài. Đề
tài báo cáo BLEU-1 đến BLEU-4 vì đây là độ đo mà mọi công trình sinh báo cáo trên MIMIC-CXR đều báo cáo, kể cả bài
báo gốc, nên là cơ sở chung tối thiểu để đặt kết quả cạnh nhau. Hạn chế của BLEU là không phân biệt được một câu
phủ định với câu khẳng định có cùng từ vựng, nên không phản ánh độ đúng lâm sàng.

## METEOR
METEOR tính trung bình điều hòa có trọng số của độ chính xác và độ nhạy ở mức từ đơn, có khớp theo gốc từ và từ
đồng nghĩa, cộng hình phạt khi thứ tự từ bị phân mảnh [13]. Đề tài dùng METEOR 1.5, phiên bản trong pycocoevalcap.
METEOR bổ sung cho BLEU ở hai điểm: nó thưởng độ nhạy, tức việc nhắc lại được nội dung của tham chiếu, và chấp nhận
cách diễn đạt khác từ. Vì thưởng độ nhạy, METEOR giảm mạnh khi báo cáo sinh ra quá ngắn, nên luôn được đọc cùng độ
dài báo cáo.

## ROUGE-L
ROUGE-L dựa trên dãy con chung dài nhất giữa báo cáo sinh ra và tham chiếu, tính dưới dạng F-measure [14]. Khác
BLEU, ROUGE-L không yêu cầu các từ khớp phải liền nhau, nên phản ánh việc giữ đúng trình tự nội dung ở mức câu,
một đặc điểm quan trọng của báo cáo X-quang vốn thường mô tả theo trình tự giải phẫu.

## CIDEr
CIDEr biểu diễn mỗi báo cáo bằng vector n-gram có trọng số TF-IDF trên toàn tập tham chiếu và đo độ tương đồng
cosin với tham chiếu [15]. Trọng số IDF làm giảm đóng góp của các cụm từ lặp lại ở hầu hết báo cáo, như các câu mô
tả phổi bình thường, và tăng đóng góp của các cụm từ đặc trưng cho từng ca. Đề tài dùng CIDEr như độ đo ít bị chi
phối bởi văn mẫu nhất trong nhóm độ đo từ vựng. Tuy vậy CIDEr cũng có thể tăng khi mô hình sinh các câu ngắn khớp
đúng cụm từ đặc trưng, nên không được dùng riêng lẻ để kết luận.

## BERTScore
BERTScore so khớp từng token của báo cáo sinh ra với token gần nhất của tham chiếu trong không gian biểu diễn ngữ
cảnh của một mô hình ngôn ngữ, rồi tính F1 trên các độ tương đồng đó [16]. Độ đo này ghi nhận các cách diễn đạt
khác từ nhưng cùng nghĩa mà các độ đo n-gram bỏ sót. Đề tài báo cáo cả giá trị thô và giá trị đã co giãn theo
đường cơ sở. Giá trị thô với RoBERTa-large tập trung quanh 0,85 cho mọi văn bản tiếng Anh trôi chảy, còn giá trị
co giãn đưa mức của hai văn bản không liên quan về gần 0 để chênh lệch dễ đọc hơn. Bài báo gốc báo cáo BERTScore
{{paper.bertscore|f3}} mà không nêu mô hình nền hay việc co giãn; giá trị này nằm xa mức 0,85 của giá trị thô, nên
không đặt được cạnh con số của đề tài khi chưa biết giao thức.

## Độ dài và tỉ lệ lặp của báo cáo sinh ra
Đề tài báo cáo thêm hai đại lượng chẩn đoán không phải độ đo chất lượng. Trung vị số từ của báo cáo sinh ra được
đặt cạnh trung vị của báo cáo tham chiếu, vì BLEU và METEOR đều phụ thuộc mạnh vào độ dài. Tỉ lệ báo cáo chứa một
5-gram lặp lại từ 3 lần trở lên đo hiện tượng suy biến khi giải mã tham lam, một lỗi đã được ghi nhận rộng rãi ở
các mô hình ngôn ngữ [17]. Chính đại lượng này giúp phát hiện lỗi điều kiện dừng sinh của đề tài: trên
{{gen.stop_probe_25.n|n}} nghiên cứu kiểm định, tỉ lệ lặp giảm từ
{{gen.stop_probe_25.res.none_old.repeated_5gram_3x|pct0}} xuống {{gen.stop_probe_25.res.none_fixed.repeated_5gram_3x|pct0}}
sau khi sửa lỗi. Độ dài 5-gram được chọn vì báo cáo của bác sĩ hiếm khi lặp một cụm năm từ, trong khi các
cụm ba từ lặp lại là bình thường trong văn phong X-quang.
::check D["gen"]["stop_probe_25"]["res"]["none_old"]["repeated_5gram_3x"] > D["gen"]["stop_probe_25"]["res"]["none_fixed"]["repeated_5gram_3x"] | Ví dụ lỗi dừng không còn đúng

## Độ đo lâm sàng chưa được triển khai
Các độ đo từ vựng không đo độ đúng lâm sàng: một báo cáo nói "không có tràn dịch màng phổi" và một báo cáo nói "có
tràn dịch màng phổi" có BLEU gần như nhau. Các công trình gần đây vì vậy báo cáo thêm độ đo hiệu quả lâm sàng, tính
F1 giữa nhãn do CheXpert hoặc CheXbert trích từ báo cáo sinh ra và từ báo cáo tham chiếu [11], [18]; RadGraph F1,
đo mức trùng khớp thực thể và quan hệ lâm sàng [19]; và RadCliQ, độ đo tổng hợp được hiệu chỉnh theo đánh giá của
bác sĩ [20]. Bài báo gốc báo cáo cả ba nhóm: RadGraph F1 {{paper.radgraph_f1|f3}}, RadCliQ {{paper.radcliq|f3}} và F1
hiệu quả lâm sàng {{paper.ce_f1|f3}}. Đề tài chưa triển khai các độ đo này vì chúng là mã nghiên cứu với giấy phép và
phiên bản riêng, chưa được đối chiếu với điểm tham chiếu đã công bố. Theo nguyên tắc của đề tài, một độ đo chưa có
được ghi là không có, không bao giờ ghi là 0, và các độ đo từ vựng không được trình bày như độ đúng lâm sàng.

## Kiểm tra độ tin cậy của lời giải thích
Với bản đồ giải thích của Giai đoạn 2, đề tài không dùng độ đo chất lượng hình ảnh mà dùng hai phép kiểm tra hợp lệ
của Adebayo và cộng sự [21]. Phép thay ảnh đo mức tăng log-likelihood âm của báo cáo đích khi thay ảnh của nghiên cứu
bằng ảnh của một nghiên cứu khác; khoảng tin cậy bootstrap phải loại trừ 0 thì mới kết luận mô hình dùng thông tin
ảnh. Phép ngẫu nhiên hóa trọng số khởi tạo lại dần các lớp của mô hình ngôn ngữ từ trên xuống và đo tương quan hạng
Spearman giữa bản đồ mới và bản đồ gốc; tương quan phải giảm dần về gần 0, nếu không bản đồ chỉ phản ánh đầu vào
chứ không phản ánh mô hình. Bài báo gốc trình bày bản đồ chú ý của token chuyên gia và đánh giá của chuyên gia trên
8 ảnh, nhưng không có phép kiểm tra hợp lệ nào.

## Tóm tắt độ đo Giai đoạn 2
::table Các độ đo sinh báo cáo và đối chiếu với bài báo gốc
::widths 3.4, 5.8, 3.9, 3.4
::font 10
| Độ đo | Phản ánh | Trong đề tài | Bài báo gốc |
|---|---|---|---|
| BLEU-1 đến BLEU-4 | Độ chính xác n-gram | pycocoevalcap | Có |
| METEOR | Độ nhạy từ, có đồng nghĩa | METEOR 1.5 | Có |
| ROUGE-L | Trình tự nội dung | pycocoevalcap | Có |
| CIDEr | Cụm từ đặc trưng từng ca | pycocoevalcap | Có |
| BERTScore | Tương đồng ngữ nghĩa | RoBERTa-large, thô và co giãn | Có, không nêu cài đặt |
| Độ dài, tỉ lệ lặp | Chẩn đoán lỗi sinh | Trung vị từ, 5-gram lặp 3 lần | Không |
| F1 hiệu quả lâm sàng | Độ đúng nhãn bệnh | Chưa có | Có |
| RadGraph F1, RadCliQ | Độ đúng thực thể lâm sàng | Chưa có | Có |
| Phép thay ảnh, ngẫu nhiên hóa | Độ tin cậy lời giải thích | Có, kèm khoảng tin cậy | Không |

::refs
D. Edirisinghe, W. Nimalsiri, M. Hennayake, D. Meedeniya, and G. Lim, "Chest X-ray report generation using abnormality guided vision language model," IEEE Access, vol. 13, pp. 157651-157673, 2025, doi: 10.1109/ACCESS.2025.3606961.
J. Irvin et al., "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," in Proc. AAAI Conf. Artif. Intell., vol. 33, 2019, pp. 590-597.
L. Seyyed-Kalantari, G. Liu, M. McDermott, I. Y. Chen, and M. Ghassemi, "CheXclusion: Fairness gaps in deep chest X-ray classifiers," in Proc. Pacific Symp. Biocomputing, 2021, pp. 232-243.
J. A. Hanley and B. J. McNeil, "The meaning and use of the area under a receiver operating characteristic (ROC) curve," Radiology, vol. 143, no. 1, pp. 29-36, 1982.
T. Fawcett, "An introduction to ROC analysis," Pattern Recognit. Lett., vol. 27, no. 8, pp. 861-874, 2006.
T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," PLoS ONE, vol. 10, no. 3, Art. no. e0118432, 2015.
Z. C. Lipton, C. Elkan, and B. Naryanaswamy, "Optimal thresholding of classifiers to maximize F1 measure," in Proc. ECML PKDD, 2014, pp. 225-239.
W. J. Youden, "Index for rating diagnostic tests," Cancer, vol. 3, no. 1, pp. 32-35, 1950.
B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman and Hall, 1993.
X. Chen et al., "Microsoft COCO captions: Data collection and evaluation server," arXiv:1504.00325, 2015.
Z. Chen, Y. Song, T.-H. Chang, and X. Wan, "Generating radiology reports via memory-driven transformer," in Proc. EMNLP, 2020, pp. 1439-1449.
K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "BLEU: A method for automatic evaluation of machine translation," in Proc. 40th Annu. Meeting Assoc. Comput. Linguistics, 2002, pp. 311-318.
M. Denkowski and A. Lavie, "Meteor Universal: Language specific translation evaluation for any target language," in Proc. 9th Workshop Statistical Machine Translation, 2014, pp. 376-380.
C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in Text Summarization Branches Out, 2004, pp. 74-81.
R. Vedantam, C. L. Zitnick, and D. Parikh, "CIDEr: Consensus-based image description evaluation," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2015, pp. 4566-4575.
T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, "BERTScore: Evaluating text generation with BERT," in Proc. Int. Conf. Learn. Represent., 2020.
A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The curious case of neural text degeneration," in Proc. Int. Conf. Learn. Represent., 2020.
A. Smit, S. Jain, P. Rajpurkar, A. Pareek, A. Y. Ng, and M. P. Lungren, "CheXbert: Combining automatic labelers and expert annotations for accurate radiology report labeling using BERT," in Proc. EMNLP, 2020, pp. 1500-1519.
S. Jain et al., "RadGraph: Extracting clinical entities and relations from radiology reports," in Proc. NeurIPS Datasets and Benchmarks Track, 2021.
F. Yu et al., "Evaluating progress in automatic chest X-ray radiology report generation," Patterns, vol. 4, no. 9, Art. no. 100802, 2023.
J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, "Sanity checks for saliency maps," in Proc. Adv. Neural Inf. Process. Syst., 2018, pp. 9505-9515.
