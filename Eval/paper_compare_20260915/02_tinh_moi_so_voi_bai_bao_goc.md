---
title: Những điểm mới của đề tài so với bài báo META-CXR gốc
subtitle: Đối chiếu từng thành phần, lý do thay đổi và bằng chứng thực nghiệm
data: so_lieu.json
chapter_new_page: false
---

::heading Phạm vi và nguồn đối chiếu
Tài liệu này so sánh pipeline của đề tài với bài báo META-CXR gốc của Edirisinghe và cộng sự, công bố trên IEEE
Access năm 2025 [1]. Nội dung bài báo được đọc từ bản toàn văn, gồm phần phương pháp, các bảng kết quả và các
nghiên cứu cắt bỏ. Ở những điểm bài báo mô tả khác với mã nguồn công khai của chính nhóm tác giả [2], tài liệu nêu
cả hai. Cả hai công trình huấn luyện và đánh giá trên MIMIC-CXR-JPG theo phân chia chính thức [3]. Với mỗi thay đổi, tài liệu trình bày bài báo gốc làm gì, đề tài làm gì, vì sao thay đổi, và bằng chứng đo
được. Các số liệu so sánh ghép cặp được tính trên cùng tập nghiên cứu với bootstrap 2.000 lần theo nghiên cứu; số
liệu đánh dấu là ghi nhận được lấy từ nhật ký thực nghiệm của đề tài.

# Tổng quan các khác biệt

::table Đối chiếu các thành phần chính giữa bài báo gốc và đề tài
::widths 3.2, 4.6, 4.6, 4.1
::font 9.5
| Thành phần | Bài báo gốc [1] | Đề tài | Lý do chính |
|---|---|---|---|
| Ô nhãn để trống | Coi là âm tính | Bị che, không tính mất mát | Không nhắc tới khác với loại trừ |
| Đơn vị lấy mẫu | Ảnh thẳng | Nghiên cứu, ảnh chính và ảnh phụ | Một báo cáo cho mỗi nghiên cứu |
| Bộ mã hóa ảnh | ResNet50 BioViL-T, ViT PubMedCLIP, Swin MedCLIP, đóng băng | BioViL-T và PubMedCLIP, mở đông lớp trên | Đo được lợi ích khi mở đông |
| Nhánh văn bản MHCAC | Dropout văn bản ở các lớp đầu | Giáo viên có văn bản, học sinh chỉ có ảnh | Khớp huấn luyện với suy luận |
| Cổng nhắc tới | Không có | Có, điểm hiện diện bằng m nhân q | Học từ ô để trống |
| ITC, ITM, sinh văn bản | Có trong phương pháp, tắt trong mã nguồn | Tắt | Truy hồi đo được ở mức ngẫu nhiên |
| Chọn ngưỡng | Điểm ROC gần góc trên trái | Vùng ổn định F1 trên tập kiểm định | Ngưỡng chuyển được sang tập kiểm tra |
| Mô hình ngôn ngữ | Vicuna-7B, LoRA hạng 8, INT8 | MedGemma 1.5 4B, QLoRA NF4 hạng 16 | Tiền huấn luyện y khoa, có bộ mã hóa ảnh riêng |
| Đường thị giác Giai đoạn 2 | Token mềm từ META-Former | Bộ mã hóa ảnh của MedGemma, thêm token mềm | Token mềm chưa được căn chỉnh |
| Đánh giá | Giá trị điểm | Giao thức chuẩn, khoảng tin cậy ghép cặp | So sánh được và kiểm định được |

# Thay đổi ở dữ liệu và nhãn

## Ô nhãn để trống được che thay vì coi là âm tính
Bài báo gốc dùng nhãn CheXpert và coi các ô để trống là lớp âm tính để nhất quán (mục III-A của [1]). Ô để trống
nghĩa là bộ gán nhãn không tìm thấy bất thường được nhắc tới trong báo cáo [4], không có nghĩa bác sĩ đã loại trừ
bất thường đó. Trên MIMIC-CXR-JPG, {{data.blank_fraction_raw|pct1}} số ô để trống, nên gán chúng thành âm tính biến
phần lớn các nhãn âm tính trong dữ liệu huấn luyện thành sự vắng mặt của thông tin. Đề tài gán ô để trống giá trị bỏ
qua và loại chúng khỏi hàm mất mát phân loại theo từng ô. Thông tin mà các ô này mang, tức bất thường có được nhắc
tới hay không, được khai thác bằng một đầu dự đoán riêng mô tả ở mục 3.3.

Hệ quả là nhãn huấn luyện của đề tài không còn chứa các nhãn âm tính được tạo ra từ ô để trống. Đổi lại, bài toán ba lớp khi chỉ
xét ô được nhắc tới lại có dương tính chiếm đa số, nên đề tài báo cáo F1 theo cách diễn giải hiện diện trong nghiên
cứu, trùng với chính sách U-Zeros của CheXpert và cách đánh giá của CheXclusion [4], [5].

## Lấy mẫu theo nghiên cứu và dùng mọi tư thế chụp
Bài báo gốc chỉ dùng ảnh chụp thẳng và mô tả số lượng theo ảnh (Bảng 1 của [1]). Một nghiên cứu trong MIMIC-CXR có
thể gồm nhiều ảnh nhưng chỉ có một báo cáo, nên lấy mẫu theo ảnh làm các nghiên cứu nhiều ảnh bị lặp báo cáo và có
trọng số lớn hơn. Đề tài lấy mẫu theo nghiên cứu: mỗi nghiên cứu chọn một ảnh chính theo thứ tự PA, AP, nghiêng và
tối đa một ảnh phụ có tư thế khác. Trên tập huấn luyện, {{data.train_aux_view|n}} trong {{data.train_studies|n}}
nghiên cứu có ảnh phụ như vậy. Ảnh phụ được đưa vào mô-đun hợp nhất góc nhìn, trong đó token của ảnh chính làm truy
vấn và token của ảnh phụ làm khóa và giá trị. Lớp chiếu đầu ra của mô-đun được khởi tạo bằng 0 để mô-đun là phép
đồng nhất ở bước đầu, nên nghiên cứu một ảnh không bị ảnh hưởng. Lý do là ảnh nghiêng cho thấy những vùng mà ảnh
thẳng che khuất, như góc sườn hoành phía sau. Đề tài chưa có nghiên cứu cắt bỏ riêng cho mô-đun này, nên đây là thay
đổi có cơ sở thiết kế nhưng chưa có bằng chứng định lượng tách riêng.

## Trích xuất phần FINDINGS có kiểm soát
Bài báo gốc không mô tả cách lấy văn bản đích từ báo cáo. Đề tài dùng bộ phân tích riêng để tách FINDINGS và
IMPRESSION, khôi phục phần tường thuật khi báo cáo không có tiêu đề FINDINGS thay vì lấy toàn bộ báo cáo, không bao
giờ dùng IMPRESSION thay cho FINDINGS, và loại các đích quá ngắn hoặc dài hơn phân vị 99% tính trên tập huấn luyện.
Nghiên cứu không có đích hợp lệ vẫn được dùng cho phân loại nhưng bị che khỏi hàm mất mát sinh báo cáo.

# Thay đổi ở Giai đoạn 1

## Bộ mã hóa ảnh và cách giữ nguyên cấu trúc token
Bài báo gốc mô tả ba bộ mã hóa đóng băng: BioViL-T với lõi ResNet50, PubMedCLIP với lõi ViT và MedCLIP với lõi Swin,
chiếu về 1.408 chiều rồi nối lại (mục III-B và III-E của [1]). Mã nguồn công khai của nhóm tác giả chỉ đưa đặc trưng
của BioViL-T và PubMedCLIP vào MHCAC [2]. Đề tài dùng hai bộ mã hóa này [6], [7] và tắt Swin, vì với bộ nhớ 16 GB việc
giữ Swin buộc phải giảm kích thước lô, và cấu hình hai bộ mã hóa cho phép giữ nguyên cấu trúc token gốc của từng bộ.

Đề tài sửa ba điểm trong cách đưa token vào MHCAC. Thứ nhất, mỗi bộ mã hóa giữ nguyên độ phân giải gốc: BioViL-T cho
lưới 14 nhân 14, PubMedCLIP cho token CLS toàn cục và lưới 7 nhân 7, thay vì thu nhỏ về cùng kích thước và bỏ token
CLS. Token CLS của PubMedCLIP không khôi phục được từ các token vùng, còn đầu ra toàn cục của BioViL-T chính là trung
bình các token vùng, nên việc bỏ CLS đã bỏ đi phần thông tin không thể thay thế. Thứ hai, token vùng của PubMedCLIP được
đưa qua lớp chuẩn hóa cuối và trừ trung bình theo ảnh. Theo số liệu ghi nhận, độ tương đồng cosin trung bình giữa các
token vùng của PubMedCLIP khi chưa xử lý là 0,674, so với 0,0017 của BioViL-T, tức các token gần như trùng nhau và
chú ý trên chúng gần như phẳng; sau xử lý con số này là -0,014. Thứ ba, mỗi bộ mã hóa có mã hóa vị trí riêng, khởi tạo
với độ lệch chuẩn 0,02, để token thứ k của hai bộ mã hóa không nhận cùng một vector vị trí.

## Mở đông một phần bộ mã hóa ảnh
Bài báo gốc giữ đóng băng các bộ mã hóa ảnh; ở pha cuối chỉ các lớp chiếu phía sau bộ mã hóa được tinh chỉnh cùng
META-Former (mục III-E của [1]). Với bộ mã hóa đóng băng, hàm mất mát phân loại chỉ có thể điều chỉnh các lớp chiếu và
MHCAC đọc trên một bản đồ đặc trưng cố định. Đề tài mở đông khối layer4 và lớp
chiếu của BioViL-T cùng hai khối Transformer cuối và lớp chuẩn hóa cuối của PubMedCLIP, tổng cộng 31,85 triệu tham số,
với tốc độ học 1e-5, thấp hơn mười lần so với phần còn lại. So với cấu hình đóng băng trên {{s1pair.unfreeze.n|n}}
nghiên cứu kiểm tra, AUROC macro thay đổi {{s1pair.unfreeze.macro_auroc.delta|sf4}}, khoảng tin cậy
[{{s1pair.unfreeze.macro_auroc.lo|sf4}}; {{s1pair.unfreeze.macro_auroc.hi|sf4}}], và F1 dương tính macro thay đổi
{{s1pair.unfreeze.positive_macro_f1.delta|sf4}}, khoảng tin cậy
[{{s1pair.unfreeze.positive_macro_f1.lo|sf4}}; {{s1pair.unfreeze.positive_macro_f1.hi|sf4}}]. Trong các thay đổi huấn
luyện đã được đo có đối chứng, đây là thay đổi duy nhất làm AUROC macro tăng có ý nghĩa thống kê. Cần lưu ý trọng số lớp được đổi
cùng lần chạy này, nên đóng góp của hai thay đổi chưa được tách riêng hoàn toàn; lập luận quy tăng AUROC cho việc mở
đông dựa trên việc trọng số lớp chỉ dời điểm hoạt động chứ không đổi được AUROC. Mở đông sâu hơn không cho thêm lợi
ích có ý nghĩa: AUROC macro thay đổi {{s1pair.depth.macro_auroc.delta|sf4}}, khoảng tin cậy
[{{s1pair.depth.macro_auroc.lo|sf4}}; {{s1pair.depth.macro_auroc.hi|sf4}}].
::check D["s1pair"]["unfreeze"]["macro_auroc"]["sig"] and D["s1pair"]["unfreeze"]["macro_auroc"]["delta"] > 0 | Mở đông không còn là cải thiện có ý nghĩa
::check D["s1pair"]["unfreeze"]["positive_macro_f1"]["sig"] and D["s1pair"]["unfreeze"]["positive_macro_f1"]["delta"] > 0 | F1 khi mở đông không còn tăng có ý nghĩa
::check not D["s1pair"]["depth"]["macro_auroc"]["sig"] | Mở đông sâu đã trở thành có ý nghĩa

## Đầu cổng nhắc tới
Đây là thành phần mới không có trong bài báo gốc. Bên cạnh đầu phân loại ba lớp, MHCAC có thêm một đầu nhị phân
cho mỗi bất thường, dự đoán bất thường đó có được nhắc tới trong báo cáo hay không. Đầu này là thành phần duy nhất học
từ các ô để trống, và được huấn luyện bằng entropy chéo nhị phân có trọng số theo tỉ lệ ô không nhắc tới trên ô được
nhắc tới nhân hệ số 2, giới hạn trên bằng 10 để tránh thiên lệch về lớp đa số. Khi đánh giá, xác suất hiện diện được tính bằng
xác suất được nhắc tới nhân với xác suất dương tính khi được nhắc tới, đúng với câu hỏi nghiên cứu có bất thường hay
không. Lý do cần đầu này là trước khi có nó, mô hình buộc phải chọn một trong ba lớp cho cả 14 bất thường trên mọi ảnh
và theo số liệu ghi nhận đã gọi dương tính trung bình 10,8 trên 14 bất thường mỗi nghiên cứu. Trên cùng mô hình và
cùng {{s1pair.gate.n|n}} nghiên cứu kiểm tra, dùng điểm hiện diện thay cho xác suất dương tính có điều kiện làm AUROC
macro thay đổi {{s1pair.gate.macro_auroc.delta|sf4}}, khoảng tin cậy
[{{s1pair.gate.macro_auroc.lo|sf4}}; {{s1pair.gate.macro_auroc.hi|sf4}}].
::check D["s1pair"]["gate"]["macro_auroc"]["sig"] and D["s1pair"]["gate"]["macro_auroc"]["delta"] > 0.04 | Cổng nhắc tới không còn cải thiện rõ

## Tách giáo viên và học sinh cho nhánh văn bản
Trong bài báo gốc, các token chuyên gia chung của MHCAC chú ý chéo với văn bản báo cáo trong các lớp đầu, có áp dụng
dropout lên đặc trưng văn bản, trong khi lúc suy luận không có văn bản (mục III-D của [1]). Thiết kế này để một mạng
duy nhất gặp hai phân phối đầu vào khác nhau giữa huấn luyện và suy luận. Đề tài tách thành hai nhánh: nhánh giáo viên
đọc cả ảnh và văn bản, chỉ tồn tại khi huấn luyện; nhánh học sinh chỉ đọc ảnh và là nhánh dùng khi suy luận. Nhánh học
sinh học từ nhãn và từ phân phối dự đoán của giáo viên qua chưng cất tri thức với nhiệt độ 2 [8], theo tinh thần học
với thông tin đặc quyền, trong đó thông tin chỉ có lúc huấn luyện được dùng để hướng dẫn một mô hình không có thông tin
đó [9]. Nhờ vậy đường suy luận không bao giờ nhìn thấy văn bản và không có sai lệch phân phối giữa hai giai đoạn.
Đề tài chưa có nghiên cứu cắt bỏ riêng cho việc tách này.

## Tắt khối căn chỉnh ảnh và văn bản của Q-Former
Bài báo gốc mô tả META-Former được huấn luyện với ba mục tiêu của BLIP-2 là đối sánh tương phản ảnh và văn bản, khớp
ảnh và văn bản, và sinh văn bản dựa trên ảnh [1], [10]. Tuy vậy, trong mã nguồn công khai của nhóm tác giả toàn bộ khối
tính các mất mát này bị chú thích bỏ, và hàm mất mát trả về chỉ gồm mất mát phân loại cộng 0,3 lần mất mát tương phản,
0,7 lần mất mát trực giao và 0,3 lần mất mát thưa của MHCAC [2]. Đề tài giữ đúng ba trọng số này.

Đề tài đã bật lại khối căn chỉnh và đo trực tiếp bằng phép kiểm tra truy hồi trên 256 cặp ảnh và báo cáo kiểm định.
Theo số liệu ghi nhận, ba lần đo sau khoảng 500 bước cập nhật, với nhiệt độ học được và với nhiệt độ cố định, đều cho
thứ hạng truy hồi trung bình từ 127,4 đến 128,4, trong khi mức ngẫu nhiên là 127,5. Nguyên nhân
phù hợp nhất là kích thước lô: BLIP-2 huấn luyện đối sánh tương phản với lô 1.680 đến 2.320 cặp [10], trong khi với GPU
16 GB lô của đề tài chỉ là 8, nên các mẫu âm trong lô quá dễ để tạo tín hiệu học. Đề tài vì vậy tắt khối này. Hệ quả
cần nêu rõ là 32 token mềm của Q-Former không được căn chỉnh với ảnh y khoa, và điều này định hướng lại thiết kế Giai
đoạn 2 ở mục 4.2. Một phép thăm dò tuyến tính trên các token này vẫn đạt AUROC macro 0,6847 trên 10 bất thường so với
0,4838 khi xáo trộn nhãn, nên chúng mang tín hiệu nhưng chỉ là một phép đọc cố định.

## Các mất mát bổ sung
Ngoài bốn mất mát của MHCAC, đề tài thêm ba mất mát. Mất mát cổng nhắc tới và mất mát chưng cất đã mô tả ở trên. Mất
mát tương phản đa dương tính giữa hai góc nhìn của cùng nghiên cứu, theo dạng của SimCLR [11], với trọng số 0,02 và
một bộ điều hợp dư sau mỗi bộ mã hóa. Bộ điều hợp là cần thiết vì mất mát này ban đầu được tính trên đặc trưng của bộ
mã hóa đóng băng nên không có tham số nào phía trước nhận gradient; theo số liệu ghi nhận, giá trị mất mát đứng yên ở
khoảng 3,99 qua bốn epoch. Cuối cùng là mất mát nhất quán góc nhìn dạng mềm, chỉ phạt khi phân kỳ vượt một ngưỡng và
bỏ qua các ô mà việc hợp nhất làm dự đoán tự tin hơn, vì ảnh phụ tồn tại chính để thay đổi dự đoán ở những vùng ảnh
chính không thấy. Các mất mát bổ sung này chưa có nghiên cứu cắt bỏ riêng.

## Hiệu chuẩn ngưỡng và cách chấm phân loại
Bài báo gốc chọn ngưỡng cho từng lớp tại điểm trên đường cong ROC gần góc trên bên trái nhất và báo cáo độ chính xác,
độ nhạy, F1 trung bình cho bài toán ba lớp (mục IV-B và V-C của [1]), không nêu tập dùng để chọn ngưỡng. Đề tài chỉ
chọn ngưỡng trên tập kiểm định, tối đa hóa F1 dưới cách diễn giải hiện diện trong nghiên cứu, đặt ngưỡng ở trung vị
vùng đạt ít nhất 95% F1 cực đại và yêu cầu ít nhất 5 mẫu dương tính. Lý do thứ nhất là F1 chỉ có ý nghĩa khi các bộ
phân loại hằng số không đạt được nó. Khi ô trống bị che và chỉ xét các ô được nhắc tới, một bộ phân loại luôn dự đoán
dương tính đã đạt F1 {{framing.masked_all_positive_f1|f4}} trên tập kiểm tra, so với
{{framing.presence_all_positive_f1|f4}} dưới cách diễn giải hiện diện trong nghiên cứu. Với cách làm của bài báo gốc,
ô trống được tính là âm tính nên lớp âm tính chiếm phần lớn số ô; nếu độ chính xác, độ nhạy và F1 được lấy trung bình
có trọng số theo số mẫu như nghiên cứu cắt bỏ của bài báo ghi, con số trung bình sẽ bị chi phối bởi lớp âm tính, vốn là
lớp dễ nhất. Lý do thứ hai là ngưỡng tối thiểu 5 mẫu giúp hiệu chuẩn được hai nhãn hiếm Pleural
Other và Fracture, làm F1 macro thay đổi {{s1pair.minpos.positive_macro_f1.delta|sf4}}, khoảng tin cậy
[{{s1pair.minpos.positive_macro_f1.lo|sf4}}; {{s1pair.minpos.positive_macro_f1.hi|sf4}}]; riêng việc chọn vùng ổn định
thay cho đỉnh chỉ thay đổi {{s1pair.plateau.positive_macro_f1.delta|sf4}} và chưa có ý nghĩa thống kê.
::check D["s1pair"]["minpos"]["positive_macro_f1"]["sig"] and not D["s1pair"]["plateau"]["positive_macro_f1"]["sig"] | Kết luận về ngưỡng không còn khớp số liệu

# Thay đổi ở Giai đoạn 2

## Mô hình ngôn ngữ
Bài báo gốc dùng Vicuna-7B lượng tử hóa INT8 với LoRA hạng {{paper.lora_r|raw}}, alpha {{paper.lora_alpha|raw}}, chỉ cập
nhật lớp chiếu đầu ra của META-Former và LoRA (mục III-E và V-B của [1]). Đề tài dùng MedGemma 1.5 4B-it [12] lượng tử
hóa NF4 theo QLoRA [13], với LoRA [14] hạng 16, alpha 32 chỉ trên tháp ngôn ngữ, tổng cộng 31,77 triệu tham số huấn
luyện, bằng 1,26% mô hình. Có ba lý do. MedGemma được tiền huấn luyện trên dữ liệu y khoa gồm cả ảnh X-quang ngực, nên
xuất phát điểm gần miền bài toán hơn một mô hình trò chuyện tổng quát. MedGemma có bộ mã hóa ảnh riêng, cho phép thiết
kế đường thị giác không phụ thuộc token mềm chưa được căn chỉnh. Cuối cùng, mô hình 4 tỉ tham số lượng tử hóa 4 bit
vừa với GPU 16 GB khi huấn luyện toàn bộ một epoch trên tập huấn luyện.

## Đường thị giác của Giai đoạn 2
Trong bài báo gốc, mô hình ngôn ngữ chỉ nhìn ảnh qua các token truy vấn của META-Former, kèm danh sách bất thường dương
tính, âm tính và không chắc chắn vượt ngưỡng trong lời nhắc (mục III-E.3 của [1]). Vì khối căn chỉnh của Q-Former
không học được như trình bày ở mục 3.5, đề tài xây dựng hai hệ thống. Hệ thống A dùng trực tiếp bộ mã hóa ảnh của
MedGemma trên ảnh chính, không dùng thông tin Giai đoạn 1. Hệ thống C giữ bộ mã hóa ảnh của MedGemma và bổ sung 32 token
mềm cùng gợi ý từ MHCAC, tức trả lời câu hỏi token mềm và gợi ý đóng góp thêm được gì, thay vì câu hỏi token mềm có thay
thế được ảnh hay không. Việc thay token mềm vào đúng vị trí của từng mẫu được kiểm tra theo từng hàng và dừng hẳn khi
sai số lượng, vì lỗi chỉ số ở đây không làm mất mát bất thường nhưng khiến mỗi báo cáo mô tả ảnh của một nghiên cứu
khác.

Kết quả trên {{gen.test_main.n|n}} nghiên cứu kiểm tra cho thấy tinh chỉnh hệ thống A so với MedGemma không tinh chỉnh
làm CIDEr thay đổi {{gen.test_main.pair.armA_minus_zeroshot.cider.delta|sf4}}, khoảng tin cậy
[{{gen.test_main.pair.armA_minus_zeroshot.cider.lo|sf4}}; {{gen.test_main.pair.armA_minus_zeroshot.cider.hi|sf4}}],
nhưng METEOR thay đổi {{gen.test_main.pair.armA_minus_zeroshot.meteor.delta|sf4}}, khoảng tin cậy
[{{gen.test_main.pair.armA_minus_zeroshot.meteor.lo|sf4}}; {{gen.test_main.pair.armA_minus_zeroshot.meteor.hi|sf4}}],
chủ yếu vì báo cáo sinh ra ngắn hơn: trung vị {{gen.test_main.res.armA.gen_median_words|n}} từ so với
{{gen.test_main.res.armA.ref_median_words|n}} từ của tham chiếu. Hệ thống A chỉ hoàn thành 0,86 epoch và không có bước
chọn checkpoint, còn hệ thống C hoàn thành trọn một epoch.
::check D["gen"]["test_main"]["pair"]["armA_minus_zeroshot"]["cider"]["sig"] and D["gen"]["test_main"]["pair"]["armA_minus_zeroshot"]["cider"]["delta"] > 0 | CIDEr của hệ thống A không còn tăng có ý nghĩa
::check D["gen"]["test_main"]["pair"]["armA_minus_zeroshot"]["meteor"]["sig"] and D["gen"]["test_main"]["pair"]["armA_minus_zeroshot"]["meteor"]["delta"] < 0 | METEOR của hệ thống A không còn giảm có ý nghĩa

## Quy tắc tạo gợi ý và trạng thái gợi ý
Bài báo gốc đưa vào lời nhắc mọi bất thường có độ tin cậy vượt ngưỡng của từng lớp [1]. Đề tài tách ba trạng thái mà
cách làm cũ gộp chung: không cung cấp gợi ý, mô hình từ chối kết luận vì không bất thường nào vượt ngưỡng, và có dự
đoán. Hai trạng thái đầu không sinh khối gợi ý nào, còn tóm tắt dạng bình thường chỉ được phép khi cả 13 bất thường
đều được dự đoán âm tính. Lý do là trong cách làm cũ, danh sách rỗng bị diễn đạt thành một câu khẳng định bình thường,
tức đưa vào lời nhắc một kết luận mà mô hình phân loại chưa đưa ra. Quy tắc mặc định dùng xác suất hiện diện thay cho
xác suất dương tính có điều kiện.

Bằng chứng thực nghiệm về gợi ý là âm tính và cần được nêu rõ. Trên {{gen.cue_rules_100.n|n}} nghiên cứu kiểm định với
cùng một checkpoint, không quy tắc nào trong năm quy tắc tạo gợi ý, kể cả các quy tắc có độ chính xác cao gấp đôi quy
tắc cũ, cải thiện có ý nghĩa so với không dùng gợi ý trên METEOR, ROUGE-L, CIDEr hay BERTScore. Thử nghiệm đưa thông
tin Giai đoạn 1 qua 13 token học được thay cho câu chữ, trên {{gen.pilot_300.n|n}} nghiên cứu, còn làm BERTScore thô
thay đổi {{gen.pilot_300.pair.D_minus_A.bertscore_raw.delta|sf4}}, khoảng tin cậy
[{{gen.pilot_300.pair.D_minus_A.bertscore_raw.lo|sf4}}; {{gen.pilot_300.pair.D_minus_A.bertscore_raw.hi|sf4}}], so với
không dùng gợi ý. Điểm mới của đề tài ở đây vì vậy là đã kiểm định có đối chứng và bác bỏ, trong chế độ huấn luyện đã
thử, giả định rằng đưa kết quả phân loại vào lời nhắc giúp sinh báo cáo tốt hơn, một giả định mà bài báo gốc chỉ minh
họa định tính.
::check all(not (v["sig"] and v["delta"] > 0) for k, p in D["gen"]["cue_rules_100"]["pair"].items() for m, v in p.items() if m in ("meteor", "rouge_l", "cider", "bertscore_raw")) | Có quy tắc gợi ý cải thiện có ý nghĩa
::check D["gen"]["pilot_300"]["pair"]["D_minus_A"]["bertscore_raw"]["sig"] and D["gen"]["pilot_300"]["pair"]["D_minus_A"]["bertscore_raw"]["delta"] < 0 | Kết quả token học được đã đổi

## Sửa điều kiện dừng sinh
Đây là sửa lỗi kỹ thuật chứ không phải cải tiến phương pháp, nhưng ảnh hưởng tới mọi số liệu Giai đoạn 2. Báo cáo đích
khi huấn luyện kết thúc bằng mã end_of_turn của MedGemma, trong khi mã nguồn ban đầu ghi đè danh sách mã dừng bằng mã
kết thúc của bộ tách từ. Theo số liệu ghi nhận trên 25 nghiên cứu kiểm định, 3.112 trong 4.000 token sinh ra nằm sau
mã end_of_turn. Sau khi sửa, trên cùng {{gen.stop_probe_25.n|n}} nghiên cứu, CIDEr thay đổi
{{gen.stop_probe_25.pair.none_fixed_minus_none_old.cider.delta|sf4}}, khoảng tin cậy
[{{gen.stop_probe_25.pair.none_fixed_minus_none_old.cider.lo|sf4}}; {{gen.stop_probe_25.pair.none_fixed_minus_none_old.cider.hi|sf4}}],
và tỉ lệ báo cáo có 5-gram lặp giảm từ {{gen.stop_probe_25.res.none_old.repeated_5gram_3x|pct0}} xuống
{{gen.stop_probe_25.res.none_fixed.repeated_5gram_3x|pct0}}.
::check D["gen"]["stop_probe_25"]["pair"]["none_fixed_minus_none_old"]["cider"]["sig"] | Sửa điều kiện dừng không còn cải thiện CIDEr có ý nghĩa

## Giao thức đánh giá
Bài báo gốc báo cáo giá trị điểm của các độ đo, không kèm khoảng tin cậy, không nêu bộ công cụ chấm điểm và không nêu
cài đặt BERTScore [1]. Đề tài chấm bằng giao thức chuẩn pycocoevalcap [15] và BERTScore RoBERTa-large ở cả dạng thô và
co giãn, so sánh mọi cặp hệ thống trên đúng một tập nghiên cứu được kiểm tra trùng khớp định danh băm, và báo cáo khoảng
tin cậy bootstrap ghép cặp [16]. Đề tài cũng theo dõi tỉ lệ lặp n-gram [17], là đại lượng đã phát hiện lỗi điều kiện
dừng. Với lời giải thích, bài báo gốc trình bày bản đồ chú ý của token chuyên gia; đề tài chỉ xuất bản đồ khi hai phép kiểm tra hợp lệ
đạt [18], với bản đồ tính bằng lan truyền chú ý có trọng số gradient [19] trên MedGemma: thay ảnh bằng ảnh nghiên cứu
khác phải làm log-likelihood âm tăng có ý nghĩa, và ngẫu nhiên hóa dần trọng số phải làm bản đồ suy biến.

# Đặt kết quả cạnh bài báo gốc

::table Độ đo sinh báo cáo trên tập kiểm tra MIMIC-CXR của bài báo gốc và của đề tài
::widths 5.6, 2.2, 2.2, 2.2, 2.2, 2.1
::font 10
| Hệ thống | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|
| META-CXR, bài báo gốc [1] | {{paper.bleu_1|f3}} | {{paper.bleu_4|f3}} | {{paper.meteor|f3}} | {{paper.rouge_l|f3}} | {{paper.cider|f3}} |
| Đề tài, hệ thống A | {{gen.test_main.res.armA.bleu_1|f3}} | {{gen.test_main.res.armA.bleu_4|f3}} | {{gen.test_main.res.armA.meteor|f3}} | {{gen.test_main.res.armA.rouge_l|f3}} | {{gen.test_main.res.armA.cider|f3}} |
| Đề tài, hệ thống C | {{gen.test_main.res.armC.bleu_1|f3}} | {{gen.test_main.res.armC.bleu_4|f3}} | {{gen.test_main.res.armC.meteor|f3}} | {{gen.test_main.res.armC.rouge_l|f3}} | {{gen.test_main.res.armC.cider|f3}} |
::note Bài báo gốc: {{paper.n_test|n}} nghiên cứu kiểm tra, không nêu bộ công cụ và phần báo cáo được chấm. Đề tài: {{gen.test_main.n|n}} nghiên cứu kiểm tra có FINDINGS hợp lệ, chấm phần FINDINGS bằng pycocoevalcap.

Trên các độ đo từ vựng, cả hai hệ thống của đề tài đều thấp hơn con số bài báo gốc công bố. Hai bên dùng cùng phân chia
chính thức của MIMIC-CXR-JPG [3], nhưng không cùng tập con nghiên cứu, không chắc cùng phần báo cáo đích và không chắc
cùng bộ công cụ, nên bảng trên cho thấy vị trí tương đối chứ không phải một phép so sánh có kiểm soát. Một phần khoảng
cách đến từ việc báo cáo sinh ra của đề tài ngắn hơn tham chiếu, làm giảm BLEU qua hệ số phạt độ ngắn và giảm METEOR vốn
thưởng độ nhạy. Ở Giai đoạn 1, bài báo gốc báo cáo F1 {{paper.cls_f1|f2}} cho bài toán ba lớp với ô trống là âm tính,
còn đề tài báo cáo AUROC macro {{s1.macro_auroc|f4}} và F1 dương tính macro {{s1.positive_macro_f1|f4}} dưới cách diễn
giải hiện diện trong nghiên cứu; hai con số F1 trả lời hai câu hỏi khác nhau và không đặt cạnh nhau được.
::check all(D["gen"]["test_main"]["res"][a][m] < D["paper"][m] for a in ("armA", "armC") for m in ("bleu_1", "bleu_4", "meteor", "rouge_l")) | Có độ đo từ vựng của đề tài không còn thấp hơn bài báo gốc

# Các hướng đã thử nhưng không giữ lại

::table Các thay đổi đã thử có đối chứng nhưng không đưa vào mô hình chính
::widths 4.4, 7.0, 5.1
::font 10
| Thay đổi | Kết quả | Kết luận |
|---|---|---|
| Mở đông sâu hơn | AUROC macro {{s1pair.depth.macro_auroc.delta|sf4}}, chưa có ý nghĩa | Không giữ |
| Hàm mất mát phân cấp | AUROC micro {{s1pair.hier_shallow.micro_auroc.delta|sf4}} có ý nghĩa, AUROC macro {{s1pair.hier_shallow.macro_auroc.delta|sf4}} chưa có ý nghĩa | Chỉ cải thiện hiệu chuẩn giữa nhãn |
| Mất mát giải thích Grad-CAM | Không tăng phân loại, bản đồ ở mức ngẫu nhiên, ghi nhận | Tắt |
| Huấn luyện thêm 5 epoch | AUROC macro trên tập kiểm tra -0,0005, ghi nhận | Không giữ |
| Bật lại ITC và ITM | Truy hồi ở mức ngẫu nhiên qua các lần đo, ghi nhận | Tắt |
| Năm quy tắc tạo gợi ý | Không quy tắc nào cải thiện có ý nghĩa | Không chứng minh được lợi ích |
| Token phát hiện học được | BERTScore thô {{gen.pilot_300.pair.D_minus_A.bertscore_raw.delta|sf4}} so với không gợi ý | Không giữ |
::check not D["s1pair"]["depth"]["macro_auroc"]["sig"] and D["s1pair"]["hier_shallow"]["micro_auroc"]["sig"] and not D["s1pair"]["hier_shallow"]["macro_auroc"]["sig"] | Bảng hướng đã thử không còn khớp số liệu

::refs
D. Edirisinghe, W. Nimalsiri, M. Hennayake, D. Meedeniya, and G. Lim, "Chest X-ray report generation using abnormality guided vision language model," IEEE Access, vol. 13, pp. 157651-157673, 2025, doi: 10.1109/ACCESS.2025.3606961.
D. Edirisinghe, "META-CXR: Multimodal expert tokens-based vision-language model for abnormality-guided chest X-ray reporting," GitHub repository, 2025. [Online]. Available: https://github.com/DasithEdirisinghe/META-CXR
A. E. W. Johnson et al., "MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs," arXiv:1901.07042, 2019.
J. Irvin et al., "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," in Proc. AAAI Conf. Artif. Intell., vol. 33, 2019, pp. 590-597.
L. Seyyed-Kalantari, G. Liu, M. McDermott, I. Y. Chen, and M. Ghassemi, "CheXclusion: Fairness gaps in deep chest X-ray classifiers," in Proc. Pacific Symp. Biocomputing, 2021, pp. 232-243.
S. Bannur et al., "Learning to exploit temporal structure for biomedical vision-language processing," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 15016-15027.
S. Eslami, C. Meinel, and G. de Melo, "PubMedCLIP: How much does CLIP benefit visual question answering in the medical domain?" in Findings Assoc. Comput. Linguistics: EACL, 2023, pp. 1181-1193.
G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," arXiv:1503.02531, 2015.
V. Vapnik and A. Vashist, "A new learning paradigm: Learning using privileged information," Neural Netw., vol. 22, no. 5-6, pp. 544-557, 2009.
J. Li, D. Li, S. Savarese, and S. Hoi, "BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models," in Proc. Int. Conf. Mach. Learn., 2023, pp. 19730-19742.
T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework for contrastive learning of visual representations," in Proc. Int. Conf. Mach. Learn., 2020, pp. 1597-1607.
A. Sellergren et al., "MedGemma technical report," arXiv:2507.05201, 2025.
T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in Proc. Adv. Neural Inf. Process. Syst., 2023.
E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," in Proc. Int. Conf. Learn. Represent., 2022.
X. Chen et al., "Microsoft COCO captions: Data collection and evaluation server," arXiv:1504.00325, 2015.
B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman and Hall, 1993.
A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The curious case of neural text degeneration," in Proc. Int. Conf. Learn. Represent., 2020.
J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, "Sanity checks for saliency maps," in Proc. Adv. Neural Inf. Process. Syst., 2018, pp. 9505-9515.
H. Chefer, S. Gur, and L. Wolf, "Generic attention-model explainability for interpreting bi-modal and encoder-decoder transformers," in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2021, pp. 397-406.
