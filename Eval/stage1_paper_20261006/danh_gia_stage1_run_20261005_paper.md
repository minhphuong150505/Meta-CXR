---
title: Đánh giá Giai đoạn 1 theo giao thức của bài báo META-CXR
subtitle: Checkpoint run_20261005_paper (cấu hình MHCAC theo bài báo, D-026), huấn luyện xong ngày 06/10/2026
subtitle: So sánh với số liệu công bố của bài báo gốc và với checkpoint trước đó run_20260930_3class
data: so_lieu.json
chapter_new_page: false
---

::heading Tóm tắt
Tài liệu này đánh giá checkpoint Giai đoạn 1 vừa huấn luyện xong lúc 02 giờ 27 ngày 06/10/2026 (run_20261005_paper,
pha 1c, checkpoint_best) trên {{n_test|n}} nghiên cứu của tập test chính thức MIMIC-CXR, bằng đúng các độ đo mà bài
báo META-CXR [1] sử dụng: precision, recall và F1 kiểu weighted cho từng bệnh rồi lấy trung bình 14 bệnh (Fig. 10),
F1 trung bình năm bệnh (Bảng 5 và Bảng 7) và AUROC một lớp so với phần còn lại cho từng lớp (Fig. 5). Mô hình đạt
precision {{new.weighted_precision|f4}}, recall {{new.weighted_recall|f4}} và F1 {{new.weighted_f1|f4}}, so với
0,87, 0,78 và 0,73 của bài báo; F1 trung bình năm bệnh đạt {{new.mean_weighted_f1_5|f4}}, so với 0,701. Như vậy F1
cao hơn bài báo, recall tương đương, còn precision và F1 năm bệnh thấp hơn. So với checkpoint trước
(run_20260930_3class) trên cùng các nghiên cứu, checkpoint mới tốt hơn có ý nghĩa thống kê ở bảy trên tám độ đo chính,
rõ nhất là F1 năm bệnh tăng {{paired.mean_weighted_f1_5.delta|sf4}} và AUROC lớp Không chắc chắn tăng
{{paired.auroc_uncertain_mean.delta|sf4}}.
::check D["ci"]["weighted_f1"]["lo"] > D["paper"]["f1"] | F1 không cao hơn bài báo một cách chắc chắn
::check D["ci"]["weighted_recall"]["lo"] <= D["paper"]["recall"] <= D["ci"]["weighted_recall"]["hi"] | Recall bài báo không nằm trong khoảng tin cậy, không được viết là tương đương
::check D["ci"]["weighted_precision"]["hi"] < D["paper"]["precision"] | Precision không thấp hơn bài báo một cách chắc chắn
::check D["ci"]["mean_weighted_f1_5"]["hi"] < D["paper"]["f1_5"] | F1 năm bệnh không thấp hơn bài báo một cách chắc chắn
::check sum(1 for k in ["weighted_precision","weighted_recall","weighted_f1","mean_weighted_f1_5","macro_recall","auroc_positive_mean","auroc_negative_mean","auroc_uncertain_mean"] if D["paired"][k]["lo"] > 0.0005) == 7 | Số độ đo tăng có ý nghĩa không phải là bảy

# Thiết lập đánh giá

## Checkpoint được đánh giá
Checkpoint là kết quả của lịch huấn luyện ba pha của Giai đoạn 1. Pha 1a (Q-Former với ITC, ITM và LM) được dùng
lại từ run_20260930_3class. Pha 1b huấn luyện MHCAC trong năm epoch và chọn epoch theo macro_recall trên tập
validation; epoch được chọn là epoch {{best_1b_epoch|raw}} (đánh số từ 0). Pha 1c huấn luyện toàn bộ thêm một epoch,
và checkpoint_best của pha này là checkpoint được đánh giá. So với run_20260930_3class, có ba điểm khác nhau. Thứ nhất,
số expert token chung của MHCAC giảm từ 14 xuống 8 như Fig. 10 của bài báo. Thứ hai, label smoothing của hàm
cross-entropy phân loại giảm từ 0,05 về 0 như phương trình (10) của bài báo. Thứ ba, tiêu chí chọn epoch ở pha 1b
là macro_recall thay cho tổng loss trên validation. Ba thay đổi này diễn ra cùng lúc nên tài liệu không tách được
đóng góp riêng của từng thay đổi.

## Dữ liệu
Tập test gồm {{n_test|n}} nghiên cứu, tập validation gồm {{n_val|n}} nghiên cứu, đều theo phân chia chính thức của
MIMIC-CXR và lấy mẫu theo nghiên cứu. Mỗi bệnh trong 14 bệnh của CheXpert [2] có ba lớp: Âm tính, Dương tính và Không
chắc chắn. Ô nhãn trống được coi là Âm tính như mục III-A của bài báo; nghiên cứu không có thông tin CheXpert nào bị
loại khỏi phép tính của bệnh tương ứng.

## Độ đo theo giao thức của bài báo
Với mỗi bệnh, lớp dự đoán là lớp có xác suất lớn nhất trong ba lớp (argmax). Precision, recall và F1 được tính kiểu
weighted của sklearn với zero_division bằng 1 cho từng bệnh, rồi lấy trung bình cộng trên 14 bệnh; đây là ba số của
Fig. 10 trong bài báo. F1 năm bệnh là trung bình F1 weighted của Atelectasis, Cardiomegaly, Consolidation, Edema và
Pleural Effusion, là số 0,701 trong Bảng 5 và Bảng 7. AUROC được tính một lớp so với phần còn lại cho từng lớp của
từng bệnh như Fig. 5; bài báo chỉ vẽ các đường ROC mà không công bố số trung bình nên cột bài báo ghi "không có".
Khoảng tin cậy 95% được ước lượng bằng bootstrap percentile 1.000 lần lấy mẫu lại theo nghiên cứu, hạt giống 42.
Lưu ý rằng recall weighted về mặt toán học bằng đúng accuracy, và với zero_division bằng 1 một lớp mà mô hình không
bao giờ dự đoán vẫn có precision bằng 1, nên precision weighted có xu hướng cao khi mô hình bỏ qua lớp hiếm.

# Kết quả

## So sánh với số liệu công bố của bài báo
::table Độ đo Giai đoạn 1 trên tập test theo giao thức của bài báo, n = {{n_test|n}} nghiên cứu
::widths 6, 2,3, 2,5, 3,7, 2
::font 11
| Độ đo | Bài báo | Mô hình | KTC 95% | Chênh lệch |
|---|---|---|---|---|
| Precision weighted, TB 14 bệnh (Fig. 10) | 0,87 | {{new.weighted_precision|f4}} | [{{ci.weighted_precision.lo|f4}}; {{ci.weighted_precision.hi|f4}}] | {{new.weighted_precision - paper.precision|sf4}} |
| Recall weighted, TB 14 bệnh (Fig. 10) | 0,78 | {{new.weighted_recall|f4}} | [{{ci.weighted_recall.lo|f4}}; {{ci.weighted_recall.hi|f4}}] | {{new.weighted_recall - paper.recall|sf4}} |
| F1 weighted, TB 14 bệnh (Fig. 10) | 0,73 | **{{new.weighted_f1|f4}}** | [{{ci.weighted_f1.lo|f4}}; {{ci.weighted_f1.hi|f4}}] | {{new.weighted_f1 - paper.f1|sf4}} |
| F1 trung bình 5 bệnh (Bảng 5, 7) | 0,701 | {{new.mean_weighted_f1_5|f4}} | [{{ci.mean_weighted_f1_5.lo|f4}}; {{ci.mean_weighted_f1_5.hi|f4}}] | {{new.mean_weighted_f1_5 - paper.f1_5|sf4}} |
| AUROC lớp Dương, TB 14 bệnh (Fig. 5) | không có | {{new.auroc_positive_mean|f4}} | [{{ci.auroc_positive_mean.lo|f4}}; {{ci.auroc_positive_mean.hi|f4}}] | không có |
| AUROC lớp Âm, TB 14 bệnh (Fig. 5) | không có | {{new.auroc_negative_mean|f4}} | [{{ci.auroc_negative_mean.lo|f4}}; {{ci.auroc_negative_mean.hi|f4}}] | không có |
| AUROC lớp Không chắc, TB 13 bệnh (Fig. 5) | không có | {{new.auroc_uncertain_mean|f4}} | [{{ci.auroc_uncertain_mean.lo|f4}}; {{ci.auroc_uncertain_mean.hi|f4}}] | không có |
::note Giá trị in đậm cao hơn bài báo với khoảng tin cậy không chứa số của bài báo. No Finding không có nhãn Không chắc chắn trong tập test nên AUROC lớp này lấy trung bình trên 13 bệnh.

F1 weighted của mô hình cao hơn bài báo {{new.weighted_f1 - paper.f1|sf4}}, và cận dưới khoảng tin cậy
({{ci.weighted_f1.lo|f4}}) vẫn cao hơn 0,73. Recall của bài báo (0,78) nằm trong khoảng tin cậy của mô hình, nên hai
giá trị là tương đương. Precision thấp hơn bài báo {{paper.precision - new.weighted_precision|f4}} và F1 năm bệnh thấp
hơn {{paper.f1_5 - new.mean_weighted_f1_5|f4}}; trong cả hai trường hợp cận trên khoảng tin cậy vẫn dưới số của bài
báo. Đối chiếu này chỉ mang tính tham khảo vì bài báo không công bố danh sách nghiên cứu của tập test và code tham
chiếu lấy trung bình độ đo theo từng batch rồi trung bình các batch, trong khi ở đây mỗi bệnh được tính trên toàn bộ
tập test.

Độ đo weighted bị chi phối bởi lớp Âm tính vì lớp này chiếm đa số ở mọi bệnh, nên cần đặt cạnh các mô hình tham
chiếu tầm thường trên cùng tập test. Bảng dưới đây cho thấy một mô hình luôn dự đoán Âm tính đã đạt F1 weighted
{{base.all_negative.weighted_f1|f4}}, gần bằng con số 0,73 của bài báo. Mô hình vượt mô hình tham chiếu này
{{new.weighted_f1 - base.all_negative.weighted_f1|sf4}} về F1 và {{new.mean_weighted_f1_5 - base.all_negative.mean_weighted_f1_5|sf4}}
về F1 năm bệnh, nhưng có precision và recall weighted thấp hơn nó, vì mô hình chấp nhận một số dự đoán Dương tính sai
để tìm ra các ca Dương tính.
::check D["new"]["weighted_f1"] > D["base"]["all_negative"]["weighted_f1"] | Mô hình không vượt mô hình luôn dự đoán Âm tính về F1
::check D["new"]["weighted_precision"] < D["base"]["all_negative"]["weighted_precision"] and D["new"]["weighted_recall"] < D["base"]["all_negative"]["weighted_recall"] | Văn bản nói precision và recall thấp hơn mô hình luôn Âm tính nhưng số liệu không như vậy

::table Mô hình so với các mô hình tham chiếu tầm thường trên cùng tập test, cùng giao thức
::widths 5,5, 2,75, 2,75, 2,75, 2,75
::font 11
| Mô hình | Precision w | Recall w | F1 w | F1 5 bệnh |
|---|---|---|---|---|
| Mô hình đánh giá | {{new.weighted_precision|f4}} | {{new.weighted_recall|f4}} | **{{new.weighted_f1|f4}}** | **{{new.mean_weighted_f1_5|f4}}** |
| Luôn dự đoán Âm tính | {{base.all_negative.weighted_precision|f4}} | {{base.all_negative.weighted_recall|f4}} | {{base.all_negative.weighted_f1|f4}} | {{base.all_negative.mean_weighted_f1_5|f4}} |
| Ngẫu nhiên theo tần suất lớp | {{base.prior_random.weighted_precision|f4}} | {{base.prior_random.weighted_recall|f4}} | {{base.prior_random.weighted_f1|f4}} | {{base.prior_random.mean_weighted_f1_5|f4}} |
| Luôn dự đoán Dương tính | {{base.all_positive.weighted_precision|f4}} | {{base.all_positive.weighted_recall|f4}} | {{base.all_positive.weighted_f1|f4}} | {{base.all_positive.mean_weighted_f1_5|f4}} |
::note Mô hình luôn dự đoán Âm tính trùng với mô hình dự đoán lớp đa số, vì lớp Âm tính là lớp đa số ở cả 14 bệnh.

## Kết quả theo từng bệnh
::table Precision, recall, F1 weighted và AUROC theo lớp cho từng bệnh trên tập test (giao thức Fig. 10 và Fig. 5)
::widths 4,2, 3,3, 1,4, 1,4, 1,4, 1,6, 1,6, 1,6
::font 9,5
| Bệnh | n Âm / Dương / Không chắc | P w | R w | F1 w | AUROC Âm | AUROC Dương | AUROC Không chắc |
|---|---|---|---|---|---|---|---|
| No Finding | {{per.0.nN|n}} / {{per.0.nP|n}} / {{per.0.nU|n}} | {{per.0.wP|f4}} | {{per.0.wR|f4}} | {{per.0.wF1|f4}} | {{per.0.aN|f4}} | {{per.0.aP|f4}} | không có |
| Enlarged Cardiomediastinum | {{per.1.nN|n}} / {{per.1.nP|n}} / {{per.1.nU|n}} | {{per.1.wP|f4}} | {{per.1.wR|f4}} | {{per.1.wF1|f4}} | {{per.1.aN|f4}} | {{per.1.aP|f4}} | {{per.1.aU|f4}} |
| Cardiomegaly (5) | {{per.2.nN|n}} / {{per.2.nP|n}} / {{per.2.nU|n}} | {{per.2.wP|f4}} | {{per.2.wR|f4}} | {{per.2.wF1|f4}} | {{per.2.aN|f4}} | {{per.2.aP|f4}} | {{per.2.aU|f4}} |
| Lung Opacity | {{per.3.nN|n}} / {{per.3.nP|n}} / {{per.3.nU|n}} | {{per.3.wP|f4}} | {{per.3.wR|f4}} | {{per.3.wF1|f4}} | {{per.3.aN|f4}} | {{per.3.aP|f4}} | {{per.3.aU|f4}} |
| Lung Lesion | {{per.4.nN|n}} / {{per.4.nP|n}} / {{per.4.nU|n}} | {{per.4.wP|f4}} | {{per.4.wR|f4}} | {{per.4.wF1|f4}} | {{per.4.aN|f4}} | {{per.4.aP|f4}} | {{per.4.aU|f4}} |
| Edema (5) | {{per.5.nN|n}} / {{per.5.nP|n}} / {{per.5.nU|n}} | {{per.5.wP|f4}} | {{per.5.wR|f4}} | {{per.5.wF1|f4}} | {{per.5.aN|f4}} | {{per.5.aP|f4}} | {{per.5.aU|f4}} |
| Consolidation (5) | {{per.6.nN|n}} / {{per.6.nP|n}} / {{per.6.nU|n}} | {{per.6.wP|f4}} | {{per.6.wR|f4}} | {{per.6.wF1|f4}} | {{per.6.aN|f4}} | {{per.6.aP|f4}} | {{per.6.aU|f4}} |
| Pneumonia | {{per.7.nN|n}} / {{per.7.nP|n}} / {{per.7.nU|n}} | {{per.7.wP|f4}} | {{per.7.wR|f4}} | {{per.7.wF1|f4}} | {{per.7.aN|f4}} | {{per.7.aP|f4}} | {{per.7.aU|f4}} |
| Atelectasis (5) | {{per.8.nN|n}} / {{per.8.nP|n}} / {{per.8.nU|n}} | {{per.8.wP|f4}} | {{per.8.wR|f4}} | {{per.8.wF1|f4}} | {{per.8.aN|f4}} | {{per.8.aP|f4}} | {{per.8.aU|f4}} |
| Pneumothorax | {{per.9.nN|n}} / {{per.9.nP|n}} / {{per.9.nU|n}} | {{per.9.wP|f4}} | {{per.9.wR|f4}} | {{per.9.wF1|f4}} | {{per.9.aN|f4}} | {{per.9.aP|f4}} | {{per.9.aU|f4}} |
| Pleural Effusion (5) | {{per.10.nN|n}} / {{per.10.nP|n}} / {{per.10.nU|n}} | {{per.10.wP|f4}} | {{per.10.wR|f4}} | {{per.10.wF1|f4}} | {{per.10.aN|f4}} | {{per.10.aP|f4}} | {{per.10.aU|f4}} |
| Pleural Other | {{per.11.nN|n}} / {{per.11.nP|n}} / {{per.11.nU|n}} | {{per.11.wP|f4}} | {{per.11.wR|f4}} | {{per.11.wF1|f4}} | {{per.11.aN|f4}} | {{per.11.aP|f4}} | {{per.11.aU|f4}} |
| Fracture | {{per.12.nN|n}} / {{per.12.nP|n}} / {{per.12.nU|n}} | {{per.12.wP|f4}} | {{per.12.wR|f4}} | {{per.12.wF1|f4}} | {{per.12.aN|f4}} | {{per.12.aP|f4}} | {{per.12.aU|f4}} |
| Support Devices | {{per.13.nN|n}} / {{per.13.nP|n}} / {{per.13.nU|n}} | {{per.13.wP|f4}} | {{per.13.wR|f4}} | {{per.13.wF1|f4}} | {{per.13.aN|f4}} | {{per.13.aP|f4}} | {{per.13.aU|f4}} |
::note (5) đánh dấu năm bệnh của F1 năm bệnh. "không có" là lớp không có mẫu nào trong tập test nên AUROC không xác định. Fracture có 4 mẫu Không chắc chắn và Support Devices có 4 mẫu, nên AUROC lớp Không chắc của hai bệnh này rất không ổn định.
::check D["per"][0]["name"] == "No Finding" and D["per"][2]["name"] == "Cardiomegaly" and D["per"][13]["name"] == "Support Devices" | Thứ tự bệnh trong bảng không khớp số liệu
::check D["per"][0]["aU"] is None | No Finding có AUROC Không chắc nhưng bảng ghi không có

Cardiomegaly là bệnh có F1 weighted thấp nhất ({{per.2.wF1|f4}}). Nguyên nhân là mô hình gọi Dương tính quá nhiều ở
bệnh này: recall lớp Dương tính đạt {{per.2.rP|f4}} nhưng recall lớp Âm tính chỉ {{per.2.rN|f4}}, trong khi lớp Âm tính
chiếm {{per.2.nN|n}} trên {{D["per"][2]["nN"] + D["per"][2]["nP"] + D["per"][2]["nU"]|n}} nghiên cứu. Ở chiều ngược lại, Enlarged Cardiomediastinum
và Fracture có F1 weighted cao (lần lượt {{per.1.wF1|f4}} và {{per.12.wF1|f4}}) nhưng mô hình không dự đoán Dương
tính cho nghiên cứu nào của hai bệnh này (recall lớp Dương tính bằng {{per.1.rP|f4}} và {{per.12.rP|f4}}); F1 cao chỉ
phản ánh việc lớp Âm tính chiếm đa số. Vì vậy {{n_wF1_above_paper|raw}} trên 14 bệnh có F1 weighted trên 0,73 không
có nghĩa là mô hình nhận ra tốt từng bệnh đó; AUROC theo lớp, không phụ thuộc ngưỡng, là chỉ báo đáng tin hơn về khả
năng phân biệt.
::check min(r["wF1"] for r in D["per"]) == D["per"][2]["wF1"] | Cardiomegaly không phải bệnh có F1 thấp nhất
::check D["per"][1]["rP"] == 0 and D["per"][12]["rP"] == 0 | Văn bản nói mô hình không dự đoán Dương tính cho hai bệnh nhưng recall khác 0

## Vì sao F1 weighted cao hơn bài báo khi precision và recall không cao hơn
Bảng so sánh với bài báo cho thấy một kết quả thoạt nhìn mâu thuẫn: recall weighted của mô hình tương đương bài báo,
precision weighted thấp hơn, nhưng F1 weighted lại cao hơn {{new.weighted_f1 - paper.f1|sf4}}. Điều này không mâu
thuẫn, vì F1 weighted không được tính từ precision weighted và recall weighted. Với mỗi bệnh, F1 được tính riêng cho
từng lớp theo công thức F1 = 2PR/(P + R), rồi lấy trung bình có trọng số theo số mẫu thật của lớp, sau đó lấy trung
bình qua 14 bệnh. Precision weighted và recall weighted là các trung bình cộng nên cho phép precision cao của lớp này
bù cho precision thấp của lớp kia, còn F1 là trung bình điều hòa nên chỉ cao khi precision và recall cùng cao trong
cùng một lớp. Ngoài ra, recall weighted của một bệnh luôn bằng accuracy của bệnh đó, nên hai mô hình có cùng recall
weighted là hai mô hình dự đoán đúng cùng tỷ lệ mẫu; khác biệt về F1 giữa chúng chỉ đến từ cách các lỗi phân bố giữa
các lớp.
::check abs(D["new"]["weighted_recall"] - D["new"]["accuracy"]) < 1e-9 if "accuracy" in D["new"] else True | Recall weighted không bằng accuracy
::check D["ci"]["weighted_recall"]["lo"] <= D["paper"]["recall"] <= D["ci"]["weighted_recall"]["hi"] | Recall bài báo không nằm trong khoảng tin cậy
::check D["new"]["weighted_precision"] < D["paper"]["precision"] and D["new"]["weighted_f1"] > D["paper"]["f1"] | Chiều chênh lệch precision và F1 không như mô tả

Ví dụ minh họa tự dựng sau đây, không phải dữ liệu của đề tài, cho thấy cơ chế này. Một bệnh có 200 mẫu, 100 mẫu lớp
A và 100 mẫu lớp B. Mô hình X chỉ dự đoán A cho 50 mẫu nó chắc chắn nhất và dự đoán B cho toàn bộ phần còn lại; mô
hình Y dự đoán đúng 75 mẫu mỗi lớp và nhầm 25 mẫu mỗi lớp sang lớp kia. Cả hai đều dự đoán đúng 150 trên 200 mẫu.
::table Ví dụ minh họa tự dựng: hai mô hình có cùng recall weighted nhưng F1 weighted khác nhau
::widths 3,4, 2,6, 2,6, 2,6, 2,6
::font 11
| Mô hình, lớp | Precision | Recall | F1 | Trọng số |
|---|---|---|---|---|
| X, lớp A | {{toy.X.A.p|f3}} | {{toy.X.A.r|f3}} | {{toy.X.A.f1|f3}} | {{toy.X.A.w|f1}} |
| X, lớp B | {{toy.X.B.p|f3}} | {{toy.X.B.r|f3}} | {{toy.X.B.f1|f3}} | {{toy.X.B.w|f1}} |
| X, weighted | **{{toy.X.p|f3}}** | {{toy.X.r|f3}} | {{toy.X.f1|f3}} | |
| Y, lớp A | {{toy.Y.A.p|f3}} | {{toy.Y.A.r|f3}} | {{toy.Y.A.f1|f3}} | {{toy.Y.A.w|f1}} |
| Y, lớp B | {{toy.Y.B.p|f3}} | {{toy.Y.B.r|f3}} | {{toy.Y.B.f1|f3}} | {{toy.Y.B.w|f1}} |
| Y, weighted | {{toy.Y.p|f3}} | {{toy.Y.r|f3}} | **{{toy.Y.f1|f3}}** | |
::note Số liệu tự dựng để minh họa cách tính, không lấy từ tập test.
::check D["toy"]["X"]["r"] == D["toy"]["Y"]["r"] | Hai mô hình minh họa không có cùng recall weighted
::check D["toy"]["X"]["p"] > D["toy"]["Y"]["p"] and D["toy"]["X"]["f1"] < D["toy"]["Y"]["f1"] | Ví dụ minh họa không cho precision X cao hơn và F1 X thấp hơn

Hai mô hình có cùng recall weighted ({{toy.X.r|f3}}), mô hình X có precision weighted cao hơn ({{toy.X.p|f3}} so với
{{toy.Y.p|f3}}), nhưng mô hình Y có F1 weighted cao hơn ({{toy.Y.f1|f3}} so với {{toy.X.f1|f3}}). Mô hình X dồn lỗi về
một phía: lớp A có precision tuyệt đối nhưng recall chỉ {{toy.X.A.r|f3}}, lớp B có recall tuyệt đối nhưng precision
chỉ {{toy.X.B.p|f3}}. Trung bình cộng che đi sự lệch này, còn F1 của từng lớp thì không. Mô hình Y phân bố lỗi đều nên
precision và recall cân bằng trong từng lớp.

Số của bài báo mang dấu hiệu của kiểu mô hình X. Nếu F1 là trung bình điều hòa của precision 0,87 và recall 0,78 thì
phải đạt {{paper.f1_from_pr|f3}}, nhưng bài báo công bố 0,73, thấp hơn cả precision lẫn recall. Điều đó chỉ xảy ra khi
trong từng bệnh có lớp precision cao nhưng recall thấp và lớp ngược lại, hoặc có lớp không bao giờ được dự đoán. Ở mô
hình của đề tài, F1 weighted ({{new.weighted_f1|f4}}) nằm giữa precision ({{new.weighted_precision|f4}}) và recall
({{new.weighted_recall|f4}}), tức lỗi được phân bố cân bằng hơn khi tính trung bình qua 14 bệnh. Vì bài báo không công
bố độ đo theo từng lớp, nhận định về mô hình của bài báo là suy luận từ ba con số tổng hợp, không kiểm chứng trực tiếp
được.
::check D["paper"]["f1"] < min(D["paper"]["precision"], D["paper"]["recall"]) | F1 bài báo không thấp hơn cả P và R
::check D["new"]["weighted_recall"] >= D["new"]["weighted_f1"] - 0.002 and D["new"]["weighted_f1"] < D["new"]["weighted_precision"] | F1 của mô hình không nằm giữa P và R như mô tả

Mô hình của đề tài cũng không cân bằng ở mọi bệnh. Có {{zd.n_below|raw}} trên 14 bệnh mà F1 weighted thấp hơn cả
precision lẫn recall weighted, gồm Enlarged Cardiomediastinum, Cardiomegaly, Pleural Effusion và Fracture. Cardiomegaly
là ví dụ rõ nhất của kiểu mô hình X trong dữ liệu thật.
::check D["zd"]["below"] == ["Enlarged Cardiomediastinum", "Cardiomegaly", "Pleural Effusion", "Fracture"] | Danh sách bệnh có F1 thấp hơn cả P và R không khớp
::table Precision, recall và F1 theo lớp của Cardiomegaly trên tập test
::widths 4,4, 2,8, 2,8, 2,8, 2,8
::font 11
| Lớp | n | Precision | Recall | F1 |
|---|---|---|---|---|
| Âm tính | {{cardio.negative.n|n}} | {{cardio.negative.p|f3}} | {{cardio.negative.r|f3}} | {{cardio.negative.f1|f3}} |
| Dương tính | {{cardio.positive.n|n}} | {{cardio.positive.p|f3}} | {{cardio.positive.r|f3}} | {{cardio.positive.f1|f3}} |
| Không chắc chắn | {{cardio.uncertain.n|n}} | {{cardio.uncertain.p|f3}} | {{cardio.uncertain.r|f3}} | {{cardio.uncertain.f1|f3}} |
| Weighted | | {{per.2.wP|f3}} | {{per.2.wR|f3}} | {{per.2.wF1|f3}} |
::note Precision 1,000 của lớp Không chắc chắn là do mô hình không dự đoán lớp này lần nào; xem đoạn dưới về zero_division.
::check D["cardio"]["uncertain"]["r"] == 0 and D["cardio"]["uncertain"]["p"] == 1.0 | Cardiomegaly có dự đoán Không chắc
::check D["cardio"]["positive"]["r"] > 0.9 and D["cardio"]["negative"]["r"] < 0.4 | Mô tả lệch lớp của Cardiomegaly không khớp

Giao thức của bài báo tính precision bằng sklearn với zero_division bằng 1: một lớp có mẫu thật nhưng không bao giờ
được dự đoán nhận precision bằng 1, recall bằng 0 và F1 bằng 0. Precision weighted vì thế được cộng một phần không phản
ánh khả năng của mô hình, còn F1 thì không. Trên tập test, mô hình của đề tài có {{zd.n_never|raw}} cặp bệnh và lớp
như vậy: lớp Không chắc chắn ở {{zd.n_never_unc_findings|raw}} bệnh, cùng lớp Dương tính của Enlarged
Cardiomediastinum và Fracture. Nếu không cộng phần này (zero_division bằng 0), precision weighted trung bình 14 bệnh
giảm từ {{zd.p_zd1|f4}} xuống {{zd.p_zd0|f4}}. Precision 0,87 của bài báo cũng chứa phần cộng này nếu mô hình của họ có lớp
không bao giờ được dự đoán, nhưng không xác định được có hay không và bao nhiêu vì bài báo không công bố số liệu theo lớp. Do đó chênh lệch precision giữa hai bên không
nên được đọc là mô hình của đề tài kém chính xác hơn; F1 weighted, vốn không được cộng điểm khi bỏ qua một lớp, là
con số đáng tin hơn trong ba con số của Fig. 10.
::check D["zd"]["n_never"] == 12 and D["zd"]["n_never_unc"] == 10 and D["zd"]["n_never_unc_findings"] == 10 | Số cặp lớp không bao giờ được dự đoán không khớp văn bản
::check sorted((x["name"], x["cls"]) for x in D["zd"]["never"] if x["cls"] == "positive") == [("Enlarged Cardiomediastinum", "positive"), ("Fracture", "positive")] | Các lớp Dương tính không bao giờ được dự đoán không khớp
::check abs(D["zd"]["p_zd1"] - D["new"]["weighted_precision"]) < 1e-6 | Precision tính lại không khớp precision trong bảng

## So sánh có ghép cặp với checkpoint trước
Hai checkpoint được chấm trên cùng {{n_test|n}} nghiên cứu (khóa nghiên cứu và nhãn được kiểm tra trùng khớp từng
dòng trước khi so sánh). Khoảng tin cậy của hiệu số được ước lượng bằng bootstrap ghép cặp {{paired_resamples|n}}
lần, mỗi lần lấy mẫu lại cùng một tập chỉ số nghiên cứu cho cả hai checkpoint, hạt giống 16. Theo quy ước của đề tài,
khoảng tin cậy có cận cách 0 dưới 0,0005 được coi là chưa xác lập.

::table Checkpoint mới so với run_20260930_3class trên cùng tập test, bootstrap ghép cặp
::widths 5,2, 2,6, 2,6, 2,4, 3,7
::font 11
| Độ đo | Run trước | Run mới | Hiệu số | KTC 95% |
|---|---|---|---|---|
| Precision weighted | {{paired.weighted_precision.old|f4}} | {{paired.weighted_precision.new|f4}} | {{paired.weighted_precision.delta|sf4}} | [{{paired.weighted_precision.lo|sf4}}; {{paired.weighted_precision.hi|sf4}}] |
| Recall weighted | {{paired.weighted_recall.old|f4}} | {{paired.weighted_recall.new|f4}} | **{{paired.weighted_recall.delta|sf4}}** | [{{paired.weighted_recall.lo|sf4}}; {{paired.weighted_recall.hi|sf4}}] |
| F1 weighted | {{paired.weighted_f1.old|f4}} | {{paired.weighted_f1.new|f4}} | **{{paired.weighted_f1.delta|sf4}}** | [{{paired.weighted_f1.lo|sf4}}; {{paired.weighted_f1.hi|sf4}}] |
| F1 trung bình 5 bệnh | {{paired.mean_weighted_f1_5.old|f4}} | {{paired.mean_weighted_f1_5.new|f4}} | **{{paired.mean_weighted_f1_5.delta|sf4}}** | [{{paired.mean_weighted_f1_5.lo|sf4}}; {{paired.mean_weighted_f1_5.hi|sf4}}] |
| macro_recall | {{paired.macro_recall.old|f4}} | {{paired.macro_recall.new|f4}} | **{{paired.macro_recall.delta|sf4}}** | [{{paired.macro_recall.lo|sf4}}; {{paired.macro_recall.hi|sf4}}] |
| AUROC lớp Dương | {{paired.auroc_positive_mean.old|f4}} | {{paired.auroc_positive_mean.new|f4}} | **{{paired.auroc_positive_mean.delta|sf4}}** | [{{paired.auroc_positive_mean.lo|sf4}}; {{paired.auroc_positive_mean.hi|sf4}}] |
| AUROC lớp Âm | {{paired.auroc_negative_mean.old|f4}} | {{paired.auroc_negative_mean.new|f4}} | **{{paired.auroc_negative_mean.delta|sf4}}** | [{{paired.auroc_negative_mean.lo|sf4}}; {{paired.auroc_negative_mean.hi|sf4}}] |
| AUROC lớp Không chắc | {{paired.auroc_uncertain_mean.old|f4}} | {{paired.auroc_uncertain_mean.new|f4}} | **{{paired.auroc_uncertain_mean.delta|sf4}}** | [{{paired.auroc_uncertain_mean.lo|sf4}}; {{paired.auroc_uncertain_mean.hi|sf4}}] |
::note Run trước là run_20260930_3class, run mới là run_20261005_paper. Hiệu số in đậm có khoảng tin cậy nằm hoàn toàn trên 0. macro_recall là trung bình không trọng số của recall ba lớp, lấy trung bình 14 bệnh; đây là độ đo chọn epoch của đề tài, không phải độ đo của bài báo.
::check D["paired"]["weighted_precision"]["lo"] < 0 < D["paired"]["weighted_precision"]["hi"] | Precision được trình bày là không đổi nhưng khoảng tin cậy không chứa 0
::check all(D["paired"][k]["lo"] > 0.0005 for k in ["weighted_recall","weighted_f1","mean_weighted_f1_5","macro_recall","auroc_positive_mean","auroc_negative_mean","auroc_uncertain_mean"]) | Một hiệu số in đậm có cận dưới không vượt 0,0005

Checkpoint mới tốt hơn ở bảy trên tám độ đo với khoảng tin cậy nằm hoàn toàn trên 0; chỉ precision weighted là không
đổi. Khác với lần so sánh run_20260930_3class với run_20260925b_3phase, trong đó hai checkpoint chỉ khác nhau trong
phạm vi nhiễu, lần này các độ đo không phụ thuộc ngưỡng cũng tăng: AUROC lớp Dương tăng
{{paired.auroc_positive_mean.delta|sf4}}, lớp Âm tăng {{paired.auroc_negative_mean.delta|sf4}}. Đây là dấu hiệu khả năng
phân biệt thực sự tốt hơn, không chỉ là dịch chuyển điểm làm việc. Theo từng bệnh, checkpoint mới có F1 weighted cao
hơn ở {{win_wF1|raw}} trên 14 bệnh, AUROC lớp Dương cao hơn ở {{win_aP|raw}} trên 14 bệnh và AUROC lớp Không chắc cao
hơn ở {{win_aU|raw}} trên {{n_aU|raw}} bệnh. Mức tăng lớn nhất là AUROC lớp Không chắc chắn
({{paired.auroc_uncertain_mean.old|f4}} lên {{paired.auroc_uncertain_mean.new|f4}}), nhưng khoảng tin cậy của nó rộng
vì lớp này có ít mẫu. Do ba thay đổi (8 expert token, bỏ label smoothing, chọn epoch theo macro_recall) diễn ra cùng
lúc, kết quả này không cho biết thay đổi nào tạo ra mức tăng.
::check D["paired"]["auroc_uncertain_mean"]["delta"] == max(v["delta"] for v in D["paired"].values()) | AUROC Không chắc không phải mức tăng lớn nhất

## Độ đo bổ sung ngoài giao thức của bài báo
Hai cách ra quyết định thay cho argmax được chấm thêm để đặt cạnh kết quả chính, không thay thế nó. Cách thứ nhất dùng
ngưỡng riêng cho từng cặp bệnh và lớp theo phương trình (22) của bài báo (khoảng cách nhỏ nhất tới góc trên bên trái
của đường ROC), ước lượng trên tập validation rồi áp dụng cho tập test; bài báo dùng các ngưỡng này để dựng prompt cho
Giai đoạn 2, không dùng để báo cáo độ đo phân loại. Cách thứ hai đặt một hoặc hai điểm cắt trên điểm
p_dương / (p_dương + p_âm) cho từng bệnh, ước lượng trên validation để tối đa F1 weighted.

::table Kết quả trên tập test khi thay argmax bằng các quy tắc quyết định ước lượng trên validation
::widths 4,4, 2,1, 2,1, 2,1, 2,1, 2,1, 1,6
::font 10
| Quy tắc quyết định | P w | R w | F1 w | F1 5 bệnh | macro_recall | Số lần dự đoán Không chắc |
|---|---|---|---|---|---|---|
| Argmax (bài báo) | {{new.weighted_precision|f4}} | {{new.weighted_recall|f4}} | {{new.weighted_f1|f4}} | {{new.mean_weighted_f1_5|f4}} | {{new.macro_recall|f4}} | {{cutA.uncertain_predicted|n}} |
| Ngưỡng phương trình (22) | {{eq22.weighted_precision|f4}} | {{eq22.weighted_recall|f4}} | {{eq22.weighted_f1|f4}} | {{eq22.mean_weighted_f1_5|f4}} | {{eq22.macro_recall|f4}} | không có |
| Một điểm cắt | {{cut1.metrics.weighted_precision|f4}} | {{cut1.metrics.weighted_recall|f4}} | {{cut1.metrics.weighted_f1|f4}} | {{cut1.metrics.mean_weighted_f1_5|f4}} | {{cut1.metrics.macro_recall|f4}} | {{cut1.uncertain_predicted|n}} |
| Hai điểm cắt | {{cut2.metrics.weighted_precision|f4}} | {{cut2.metrics.weighted_recall|f4}} | {{cut2.metrics.weighted_f1|f4}} | {{cut2.metrics.mean_weighted_f1_5|f4}} | {{cut2.metrics.macro_recall|f4}} | {{cut2.uncertain_predicted|n}} |
::note Tập test có {{cutA.uncertain_true|n}} ô nhãn Không chắc chắn trên 14 bệnh. Số lần dự đoán Không chắc của quy tắc phương trình (22) không được script chấm điểm xuất ra nên ghi "không có".

Một điểm cắt nâng F1 năm bệnh lên {{cut1.metrics.mean_weighted_f1_5|f4}}, cao hơn con số 0,701 của bài báo, và F1
weighted lên {{cut1.metrics.weighted_f1|f4}}; hiệu số so với argmax là {{cut1.minus_argmax.mean_weighted_f1_5.delta|sf4}}
cho F1 năm bệnh, khoảng tin cậy [{{cut1.minus_argmax.mean_weighted_f1_5.ci95.0|sf4}}; {{cut1.minus_argmax.mean_weighted_f1_5.ci95.1|sf4}}]
(bootstrap ghép cặp 1.000 lần, hạt giống 16). Cái giá là macro_recall giảm và quy tắc này không bao giờ dự đoán lớp
Không chắc chắn. Mức tăng đến từ việc dịch điểm cắt giữa Dương tính và Âm tính, tức là sửa xu hướng gọi Dương tính quá
nhiều mà trọng số lớp đưa vào argmax. Ngưỡng phương trình (22) làm điều ngược lại: macro_recall tăng lên
{{eq22.macro_recall|f4}} nhưng F1 weighted giảm còn {{eq22.weighted_f1|f4}} vì recall lớp Âm tính giảm. Các quy tắc này
không phải giao thức của bài báo, nên con số dùng để so với bài báo vẫn là dòng argmax.
::check D["cut1"]["metrics"]["mean_weighted_f1_5"] > D["paper"]["f1_5"] | Một điểm cắt không vượt 0,701
::check D["cut1"]["minus_argmax"]["mean_weighted_f1_5"]["ci95"][0] > 0.0005 | Mức tăng F1 năm bệnh của một điểm cắt chưa xác lập
::check D["cut1"]["metrics"]["macro_recall"] < D["new"]["macro_recall"] | Một điểm cắt không làm giảm macro_recall
::check D["eq22"]["macro_recall"] > D["new"]["macro_recall"] and D["eq22"]["weighted_f1"] < D["new"]["weighted_f1"] | Mô tả ngưỡng phương trình (22) không khớp số liệu

## Kết quả trên validation và quá trình chọn epoch
::table Độ đo trên validation ở từng epoch của pha 1b và ở pha 1c, n = {{n_val|n}} nghiên cứu
::widths 3,4, 2,6, 2,6, 2,6, 2,6, 2,6
::font 11
| Epoch | macro_recall | F1 w | F1 5 bệnh | AUROC Dương | AUROC Không chắc |
|---|---|---|---|---|---|
| 1b, epoch 0 | {{p1b.0.macro_recall|f4}} | {{p1b.0.weighted_f1|f4}} | {{p1b.0.mean_weighted_f1_5|f4}} | {{p1b.0.auroc_positive_mean|f4}} | {{p1b.0.auroc_uncertain_mean|f4}} |
| 1b, epoch 1 | {{p1b.1.macro_recall|f4}} | {{p1b.1.weighted_f1|f4}} | {{p1b.1.mean_weighted_f1_5|f4}} | {{p1b.1.auroc_positive_mean|f4}} | {{p1b.1.auroc_uncertain_mean|f4}} |
| 1b, epoch 2 (chọn) | **{{p1b.2.macro_recall|f4}}** | {{p1b.2.weighted_f1|f4}} | {{p1b.2.mean_weighted_f1_5|f4}} | {{p1b.2.auroc_positive_mean|f4}} | {{p1b.2.auroc_uncertain_mean|f4}} |
| 1b, epoch 3 | {{p1b.3.macro_recall|f4}} | {{p1b.3.weighted_f1|f4}} | {{p1b.3.mean_weighted_f1_5|f4}} | {{p1b.3.auroc_positive_mean|f4}} | {{p1b.3.auroc_uncertain_mean|f4}} |
| 1b, epoch 4 | {{p1b.4.macro_recall|f4}} | {{p1b.4.weighted_f1|f4}} | {{p1b.4.mean_weighted_f1_5|f4}} | {{p1b.4.auroc_positive_mean|f4}} | {{p1b.4.auroc_uncertain_mean|f4}} |
| 1c, epoch 0 | {{p1c.0.macro_recall|f4}} | {{p1c.0.weighted_f1|f4}} | {{p1c.0.mean_weighted_f1_5|f4}} | {{p1c.0.auroc_positive_mean|f4}} | {{p1c.0.auroc_uncertain_mean|f4}} |
::check D["best_1b_epoch"] == 2 | Epoch được chọn không phải epoch 2

Trên validation, F1 năm bệnh của checkpoint cuối là {{val.mean_weighted_f1_5|f4}}, cao hơn 0,701 của bài báo, trong khi
trên test chỉ đạt {{new.mean_weighted_f1_5|f4}}. Khoảng cách {{val.mean_weighted_f1_5 - new.mean_weighted_f1_5|f4}} giữa
hai tập lớn hơn nhiều so với sai khác giữa các epoch, nên không thể dùng số trên validation để khẳng định mô hình đạt
hoặc vượt bài báo. Pha 1c làm macro_recall trên validation giảm nhẹ so với epoch được chọn của pha 1b
({{p1c.0.macro_recall|f4}} so với {{p1b.2.macro_recall|f4}}), phù hợp với ghi nhận trước đây rằng pha 1c hầu như không
thay đổi phần phân loại.
::check D["val"]["mean_weighted_f1_5"] > D["paper"]["f1_5"] > D["new"]["mean_weighted_f1_5"] | Mô tả chênh lệch validation và test không khớp
::check D["p1c"][0]["macro_recall"] < D["p1b"][2]["macro_recall"] | Pha 1c không làm giảm macro_recall

# Thảo luận và giới hạn
Điểm yếu rõ nhất của mô hình vẫn là lớp Không chắc chắn. Với argmax, mô hình chỉ dự đoán lớp này
{{cutA.uncertain_predicted|n}} lần cho {{cutA.uncertain_true|n}} ô nhãn Không chắc chắn, recall
{{cutA.uncertain_recall|f4}} và precision {{cutA.uncertain_precision|f4}}, nên macro_recall chỉ đạt
{{new.macro_recall|f4}} so với mức ngẫu nhiên 0,3333. Các độ đo weighted của bài báo che khuất điểm yếu này vì lớp Không
chắc chắn chiếm tỷ lệ nhỏ trong trọng số. AUROC lớp Không chắc đã tăng rõ so với checkpoint trước nhưng vẫn thấp hơn
nhiều so với hai lớp còn lại.

Có bốn giới hạn khi đặt các con số này cạnh bài báo. Thứ nhất, bài báo không công bố danh sách nghiên cứu của tập test,
nên chênh lệch có thể một phần do khác tập dữ liệu. Thứ hai, code tham chiếu của bài báo lấy trung bình độ đo theo batch,
còn đề tài tính trên toàn bộ tập test. Thứ ba, đề tài chỉ huấn luyện một lần với một hạt giống, nên khoảng tin cậy chỉ
phản ánh biến thiên do lấy mẫu dữ liệu, không phản ánh biến thiên do huấn luyện. Thứ tư, hai độ đo phân loại khác của
bài báo chưa tính được: F1 liên miền trên CheXpert validation (Bảng 4, 0,699) vì repo chưa có bộ nạp tập này cho Giai
đoạn 1, và Clinical Efficacy (Bảng 3) vì repo chưa có bộ gán nhãn CheXpert cho báo cáo sinh ra; cả hai được ghi là
không có, không phải bằng 0.

# Kết luận
Theo đúng giao thức của bài báo, checkpoint run_20261005_paper đạt F1 weighted {{new.weighted_f1|f4}}, cao hơn con số
0,73 của bài báo, recall {{new.weighted_recall|f4}} tương đương 0,78, còn precision {{new.weighted_precision|f4}} và F1
năm bệnh {{new.mean_weighted_f1_5|f4}} thấp hơn 0,87 và 0,701. So với checkpoint trước trên cùng tập test, checkpoint
mới tốt hơn có ý nghĩa thống kê ở bảy trên tám độ đo, kể cả các độ đo AUROC không phụ thuộc ngưỡng, nên đây là
checkpoint Giai đoạn 1 tốt nhất hiện có theo giao thức ba lớp và là checkpoint đang được dùng cho lượt huấn luyện Giai
đoạn 2 theo cấu hình của bài báo. Khoảng cách còn lại so với bài báo ở F1 năm bệnh nằm chủ yếu ở cách ra quyết định:
khi điểm cắt giữa Dương tính và Âm tính được ước lượng trên validation, F1 năm bệnh vượt 0,701, nhưng kết quả đó nằm
ngoài giao thức của bài báo và chỉ được báo cáo như số bổ sung.

::refs
D. Edirisinghe, W. Nimalsiri, M. Hennayake, D. Meedeniya, and G. Lim, "Chest X-ray report generation using abnormality guided vision language model," IEEE Access, vol. 13, pp. 157651-157673, 2025, doi: 10.1109/ACCESS.2025.3606961.
J. Irvin et al., "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," in Proc. AAAI Conf. Artif. Intell., vol. 33, 2019, pp. 590-597.
A. E. W. Johnson et al., "MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs," arXiv:1901.07042, 2019.
