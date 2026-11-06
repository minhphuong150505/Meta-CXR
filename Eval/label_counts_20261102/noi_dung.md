---
title: PHÂN BỐ NHÃN CHEXPERT SAU TIỀN XỬ LÝ DỮ LIỆU MIMIC-CXR
subtitle: Số nghiên cứu Positive, Negative, Uncertain theo từng lớp trên các tập train, test, val
subtitle: Bộ chia full_allviews_v2, số liệu đếm ngày 02/11/2026
data: so_lieu.json
---

::heading Tóm tắt
Tài liệu này trình bày số nghiên cứu (study) mang nhãn Positive (P), Negative (N) và
Uncertain (U) của 14 lớp CheXpert trên ba tập train, test và val sau khi tiền xử lý dữ liệu
MIMIC-CXR. Số liệu được đếm theo hai cách: theo nhãn gốc, trong đó ô trống không được tính,
và theo cách mô hình được huấn luyện, trong đó ô trống được coi là Negative.

# Dữ liệu và cách đếm

## Nguồn dữ liệu
Số liệu được ghép từ hai nguồn. Thứ nhất là file nhãn CheXpert gốc của MIMIC-CXR-JPG
(mimic-cxr-2.0.0-chexpert.csv.gz), mỗi dòng là một nghiên cứu với 14 cột nhãn, trong đó giá
trị 1 là Positive, 0 là Negative, -1 là Uncertain và ô trống nghĩa là báo cáo không nhắc tới
bệnh đó. Thứ hai là ba file chia tập sau tiền xử lý train.csv, val.csv và test.csv của bộ chia
full_allviews_v2, cũng là các file mô hình đọc khi huấn luyện. Bộ chia này giữ nguyên phân chia
chính thức của MIMIC-CXR và giữ mọi góc chụp.

## Cách đếm
Việc đếm thực hiện ở mức nghiên cứu: mỗi nghiên cứu chỉ được tính một lần dù có nhiều ảnh.
Danh sách nghiên cứu của từng tập được nối với file nhãn theo mã bệnh nhân và mã nghiên cứu,
sau đó nhãn được chuyển đổi đúng như khi huấn luyện, bằng cùng một hàm của mã nguồn. Tập train
có {{splits.train.studies|n}} nghiên cứu, tập val có {{splits.val.studies|n}} và tập
test có {{splits.test.studies|n}}. Trong đó lần lượt {{splits.train.unlabelled|n}},
{{splits.val.unlabelled|n}} và {{splits.test.unlabelled|n}} nghiên cứu không có thông
tin CheXpert, do không có bản ghi nhãn hoặc có bản ghi nhưng cả 14 ô đều trống. Các nghiên cứu
này không được tính vào bất kỳ ô nào của hai bảng dưới đây, nên số nghiên cứu có nhãn là
{{splits.train.labelled|n}} ở train, {{splits.val.labelled|n}} ở val và
{{splits.test.labelled|n}} ở test.
::check all(D["splits"][s]["studies"] - D["splits"][s]["unlabelled"] == D["splits"][s]["labelled"] for s in ("train", "val", "test")) | Số nghiên cứu có nhãn không khớp

# Phân bố nhãn

## Theo nhãn gốc
Bảng 2.1 đếm nhãn đúng như file CheXpert ghi, ô trống không được tính. Đây là cùng cách đếm với
bảng thống kê nhãn trong bài báo META-CXR: lớp No Finding chỉ có Positive vì công cụ gán nhãn
không bao giờ ghi Negative cho lớp này.

::table Số nghiên cứu theo nhãn gốc, ô trống không tính
::widths 3.3, 1.6, 1.6, 1.6, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25
::font 9
| Lớp | Train P | Train N | Train U | Test P | Test N | Test U | Val P | Val N | Val U |
|---|---|---|---|---|---|---|---|---|---|
| Atelectasis | {{labels.atelectasis.ignore.train.pos|n}} | {{labels.atelectasis.ignore.train.neg|n}} | {{labels.atelectasis.ignore.train.unc|n}} | {{labels.atelectasis.ignore.test.pos|n}} | {{labels.atelectasis.ignore.test.neg|n}} | {{labels.atelectasis.ignore.test.unc|n}} | {{labels.atelectasis.ignore.val.pos|n}} | {{labels.atelectasis.ignore.val.neg|n}} | {{labels.atelectasis.ignore.val.unc|n}} |
| Cardiomegaly | {{labels.cardiomegaly.ignore.train.pos|n}} | {{labels.cardiomegaly.ignore.train.neg|n}} | {{labels.cardiomegaly.ignore.train.unc|n}} | {{labels.cardiomegaly.ignore.test.pos|n}} | {{labels.cardiomegaly.ignore.test.neg|n}} | {{labels.cardiomegaly.ignore.test.unc|n}} | {{labels.cardiomegaly.ignore.val.pos|n}} | {{labels.cardiomegaly.ignore.val.neg|n}} | {{labels.cardiomegaly.ignore.val.unc|n}} |
| Consolidation | {{labels.consolidation.ignore.train.pos|n}} | {{labels.consolidation.ignore.train.neg|n}} | {{labels.consolidation.ignore.train.unc|n}} | {{labels.consolidation.ignore.test.pos|n}} | {{labels.consolidation.ignore.test.neg|n}} | {{labels.consolidation.ignore.test.unc|n}} | {{labels.consolidation.ignore.val.pos|n}} | {{labels.consolidation.ignore.val.neg|n}} | {{labels.consolidation.ignore.val.unc|n}} |
| Edema | {{labels.edema.ignore.train.pos|n}} | {{labels.edema.ignore.train.neg|n}} | {{labels.edema.ignore.train.unc|n}} | {{labels.edema.ignore.test.pos|n}} | {{labels.edema.ignore.test.neg|n}} | {{labels.edema.ignore.test.unc|n}} | {{labels.edema.ignore.val.pos|n}} | {{labels.edema.ignore.val.neg|n}} | {{labels.edema.ignore.val.unc|n}} |
| Enlarged Cardio | {{labels.enlarged_cardio.ignore.train.pos|n}} | {{labels.enlarged_cardio.ignore.train.neg|n}} | {{labels.enlarged_cardio.ignore.train.unc|n}} | {{labels.enlarged_cardio.ignore.test.pos|n}} | {{labels.enlarged_cardio.ignore.test.neg|n}} | {{labels.enlarged_cardio.ignore.test.unc|n}} | {{labels.enlarged_cardio.ignore.val.pos|n}} | {{labels.enlarged_cardio.ignore.val.neg|n}} | {{labels.enlarged_cardio.ignore.val.unc|n}} |
| Fracture | {{labels.fracture.ignore.train.pos|n}} | {{labels.fracture.ignore.train.neg|n}} | {{labels.fracture.ignore.train.unc|n}} | {{labels.fracture.ignore.test.pos|n}} | {{labels.fracture.ignore.test.neg|n}} | {{labels.fracture.ignore.test.unc|n}} | {{labels.fracture.ignore.val.pos|n}} | {{labels.fracture.ignore.val.neg|n}} | {{labels.fracture.ignore.val.unc|n}} |
| Lung Lesion | {{labels.lung_lesion.ignore.train.pos|n}} | {{labels.lung_lesion.ignore.train.neg|n}} | {{labels.lung_lesion.ignore.train.unc|n}} | {{labels.lung_lesion.ignore.test.pos|n}} | {{labels.lung_lesion.ignore.test.neg|n}} | {{labels.lung_lesion.ignore.test.unc|n}} | {{labels.lung_lesion.ignore.val.pos|n}} | {{labels.lung_lesion.ignore.val.neg|n}} | {{labels.lung_lesion.ignore.val.unc|n}} |
| Lung Opacity | {{labels.lung_opacity.ignore.train.pos|n}} | {{labels.lung_opacity.ignore.train.neg|n}} | {{labels.lung_opacity.ignore.train.unc|n}} | {{labels.lung_opacity.ignore.test.pos|n}} | {{labels.lung_opacity.ignore.test.neg|n}} | {{labels.lung_opacity.ignore.test.unc|n}} | {{labels.lung_opacity.ignore.val.pos|n}} | {{labels.lung_opacity.ignore.val.neg|n}} | {{labels.lung_opacity.ignore.val.unc|n}} |
| No Finding | {{labels.no_finding.ignore.train.pos|n}} | {{labels.no_finding.ignore.train.neg|n}} | {{labels.no_finding.ignore.train.unc|n}} | {{labels.no_finding.ignore.test.pos|n}} | {{labels.no_finding.ignore.test.neg|n}} | {{labels.no_finding.ignore.test.unc|n}} | {{labels.no_finding.ignore.val.pos|n}} | {{labels.no_finding.ignore.val.neg|n}} | {{labels.no_finding.ignore.val.unc|n}} |
| Pleural Effusion | {{labels.pleural_effusion.ignore.train.pos|n}} | {{labels.pleural_effusion.ignore.train.neg|n}} | {{labels.pleural_effusion.ignore.train.unc|n}} | {{labels.pleural_effusion.ignore.test.pos|n}} | {{labels.pleural_effusion.ignore.test.neg|n}} | {{labels.pleural_effusion.ignore.test.unc|n}} | {{labels.pleural_effusion.ignore.val.pos|n}} | {{labels.pleural_effusion.ignore.val.neg|n}} | {{labels.pleural_effusion.ignore.val.unc|n}} |
| Pleural Other | {{labels.pleural_other.ignore.train.pos|n}} | {{labels.pleural_other.ignore.train.neg|n}} | {{labels.pleural_other.ignore.train.unc|n}} | {{labels.pleural_other.ignore.test.pos|n}} | {{labels.pleural_other.ignore.test.neg|n}} | {{labels.pleural_other.ignore.test.unc|n}} | {{labels.pleural_other.ignore.val.pos|n}} | {{labels.pleural_other.ignore.val.neg|n}} | {{labels.pleural_other.ignore.val.unc|n}} |
| Pneumonia | {{labels.pneumonia.ignore.train.pos|n}} | {{labels.pneumonia.ignore.train.neg|n}} | {{labels.pneumonia.ignore.train.unc|n}} | {{labels.pneumonia.ignore.test.pos|n}} | {{labels.pneumonia.ignore.test.neg|n}} | {{labels.pneumonia.ignore.test.unc|n}} | {{labels.pneumonia.ignore.val.pos|n}} | {{labels.pneumonia.ignore.val.neg|n}} | {{labels.pneumonia.ignore.val.unc|n}} |
| Pneumothorax | {{labels.pneumothorax.ignore.train.pos|n}} | {{labels.pneumothorax.ignore.train.neg|n}} | {{labels.pneumothorax.ignore.train.unc|n}} | {{labels.pneumothorax.ignore.test.pos|n}} | {{labels.pneumothorax.ignore.test.neg|n}} | {{labels.pneumothorax.ignore.test.unc|n}} | {{labels.pneumothorax.ignore.val.pos|n}} | {{labels.pneumothorax.ignore.val.neg|n}} | {{labels.pneumothorax.ignore.val.unc|n}} |
| Support Devices | {{labels.support_devices.ignore.train.pos|n}} | {{labels.support_devices.ignore.train.neg|n}} | {{labels.support_devices.ignore.train.unc|n}} | {{labels.support_devices.ignore.test.pos|n}} | {{labels.support_devices.ignore.test.neg|n}} | {{labels.support_devices.ignore.test.unc|n}} | {{labels.support_devices.ignore.val.pos|n}} | {{labels.support_devices.ignore.val.neg|n}} | {{labels.support_devices.ignore.val.unc|n}} |
::note P: Positive, N: Negative, U: Uncertain. Đếm ở mức nghiên cứu.
::check all(D["labels"]["no_finding"]["ignore"][s]["neg"] == 0 and D["labels"]["no_finding"]["ignore"][s]["unc"] == 0 for s in ("train", "val", "test")) | No Finding có Negative hoặc Uncertain theo nhãn gốc

::pagebreak
## Khi ô trống được coi là Negative
Bảng 2.2 đếm theo cách mô hình hiện tại được huấn luyện, tương ứng với thiết lập
blank_label_policy bằng negative, giống cách bài báo gốc xử lý giá trị thiếu. Các cột P và U giữ
nguyên như Bảng 2.1; chỉ cột N tăng lên vì mọi ô trống của nghiên cứu có thông tin CheXpert được
tính là Negative. Khi đó Negative là lớp đông nhất ở mọi bệnh, và No Finding có
{{labels.no_finding.negative.train.neg|n}} nghiên cứu Negative trên tập train.

::table Số nghiên cứu khi ô trống được coi là Negative
::widths 3.3, 1.6, 1.6, 1.6, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25
::font 9
| Lớp | Train P | Train N | Train U | Test P | Test N | Test U | Val P | Val N | Val U |
|---|---|---|---|---|---|---|---|---|---|
| Atelectasis | {{labels.atelectasis.negative.train.pos|n}} | {{labels.atelectasis.negative.train.neg|n}} | {{labels.atelectasis.negative.train.unc|n}} | {{labels.atelectasis.negative.test.pos|n}} | {{labels.atelectasis.negative.test.neg|n}} | {{labels.atelectasis.negative.test.unc|n}} | {{labels.atelectasis.negative.val.pos|n}} | {{labels.atelectasis.negative.val.neg|n}} | {{labels.atelectasis.negative.val.unc|n}} |
| Cardiomegaly | {{labels.cardiomegaly.negative.train.pos|n}} | {{labels.cardiomegaly.negative.train.neg|n}} | {{labels.cardiomegaly.negative.train.unc|n}} | {{labels.cardiomegaly.negative.test.pos|n}} | {{labels.cardiomegaly.negative.test.neg|n}} | {{labels.cardiomegaly.negative.test.unc|n}} | {{labels.cardiomegaly.negative.val.pos|n}} | {{labels.cardiomegaly.negative.val.neg|n}} | {{labels.cardiomegaly.negative.val.unc|n}} |
| Consolidation | {{labels.consolidation.negative.train.pos|n}} | {{labels.consolidation.negative.train.neg|n}} | {{labels.consolidation.negative.train.unc|n}} | {{labels.consolidation.negative.test.pos|n}} | {{labels.consolidation.negative.test.neg|n}} | {{labels.consolidation.negative.test.unc|n}} | {{labels.consolidation.negative.val.pos|n}} | {{labels.consolidation.negative.val.neg|n}} | {{labels.consolidation.negative.val.unc|n}} |
| Edema | {{labels.edema.negative.train.pos|n}} | {{labels.edema.negative.train.neg|n}} | {{labels.edema.negative.train.unc|n}} | {{labels.edema.negative.test.pos|n}} | {{labels.edema.negative.test.neg|n}} | {{labels.edema.negative.test.unc|n}} | {{labels.edema.negative.val.pos|n}} | {{labels.edema.negative.val.neg|n}} | {{labels.edema.negative.val.unc|n}} |
| Enlarged Cardio | {{labels.enlarged_cardio.negative.train.pos|n}} | {{labels.enlarged_cardio.negative.train.neg|n}} | {{labels.enlarged_cardio.negative.train.unc|n}} | {{labels.enlarged_cardio.negative.test.pos|n}} | {{labels.enlarged_cardio.negative.test.neg|n}} | {{labels.enlarged_cardio.negative.test.unc|n}} | {{labels.enlarged_cardio.negative.val.pos|n}} | {{labels.enlarged_cardio.negative.val.neg|n}} | {{labels.enlarged_cardio.negative.val.unc|n}} |
| Fracture | {{labels.fracture.negative.train.pos|n}} | {{labels.fracture.negative.train.neg|n}} | {{labels.fracture.negative.train.unc|n}} | {{labels.fracture.negative.test.pos|n}} | {{labels.fracture.negative.test.neg|n}} | {{labels.fracture.negative.test.unc|n}} | {{labels.fracture.negative.val.pos|n}} | {{labels.fracture.negative.val.neg|n}} | {{labels.fracture.negative.val.unc|n}} |
| Lung Lesion | {{labels.lung_lesion.negative.train.pos|n}} | {{labels.lung_lesion.negative.train.neg|n}} | {{labels.lung_lesion.negative.train.unc|n}} | {{labels.lung_lesion.negative.test.pos|n}} | {{labels.lung_lesion.negative.test.neg|n}} | {{labels.lung_lesion.negative.test.unc|n}} | {{labels.lung_lesion.negative.val.pos|n}} | {{labels.lung_lesion.negative.val.neg|n}} | {{labels.lung_lesion.negative.val.unc|n}} |
| Lung Opacity | {{labels.lung_opacity.negative.train.pos|n}} | {{labels.lung_opacity.negative.train.neg|n}} | {{labels.lung_opacity.negative.train.unc|n}} | {{labels.lung_opacity.negative.test.pos|n}} | {{labels.lung_opacity.negative.test.neg|n}} | {{labels.lung_opacity.negative.test.unc|n}} | {{labels.lung_opacity.negative.val.pos|n}} | {{labels.lung_opacity.negative.val.neg|n}} | {{labels.lung_opacity.negative.val.unc|n}} |
| No Finding | {{labels.no_finding.negative.train.pos|n}} | {{labels.no_finding.negative.train.neg|n}} | {{labels.no_finding.negative.train.unc|n}} | {{labels.no_finding.negative.test.pos|n}} | {{labels.no_finding.negative.test.neg|n}} | {{labels.no_finding.negative.test.unc|n}} | {{labels.no_finding.negative.val.pos|n}} | {{labels.no_finding.negative.val.neg|n}} | {{labels.no_finding.negative.val.unc|n}} |
| Pleural Effusion | {{labels.pleural_effusion.negative.train.pos|n}} | {{labels.pleural_effusion.negative.train.neg|n}} | {{labels.pleural_effusion.negative.train.unc|n}} | {{labels.pleural_effusion.negative.test.pos|n}} | {{labels.pleural_effusion.negative.test.neg|n}} | {{labels.pleural_effusion.negative.test.unc|n}} | {{labels.pleural_effusion.negative.val.pos|n}} | {{labels.pleural_effusion.negative.val.neg|n}} | {{labels.pleural_effusion.negative.val.unc|n}} |
| Pleural Other | {{labels.pleural_other.negative.train.pos|n}} | {{labels.pleural_other.negative.train.neg|n}} | {{labels.pleural_other.negative.train.unc|n}} | {{labels.pleural_other.negative.test.pos|n}} | {{labels.pleural_other.negative.test.neg|n}} | {{labels.pleural_other.negative.test.unc|n}} | {{labels.pleural_other.negative.val.pos|n}} | {{labels.pleural_other.negative.val.neg|n}} | {{labels.pleural_other.negative.val.unc|n}} |
| Pneumonia | {{labels.pneumonia.negative.train.pos|n}} | {{labels.pneumonia.negative.train.neg|n}} | {{labels.pneumonia.negative.train.unc|n}} | {{labels.pneumonia.negative.test.pos|n}} | {{labels.pneumonia.negative.test.neg|n}} | {{labels.pneumonia.negative.test.unc|n}} | {{labels.pneumonia.negative.val.pos|n}} | {{labels.pneumonia.negative.val.neg|n}} | {{labels.pneumonia.negative.val.unc|n}} |
| Pneumothorax | {{labels.pneumothorax.negative.train.pos|n}} | {{labels.pneumothorax.negative.train.neg|n}} | {{labels.pneumothorax.negative.train.unc|n}} | {{labels.pneumothorax.negative.test.pos|n}} | {{labels.pneumothorax.negative.test.neg|n}} | {{labels.pneumothorax.negative.test.unc|n}} | {{labels.pneumothorax.negative.val.pos|n}} | {{labels.pneumothorax.negative.val.neg|n}} | {{labels.pneumothorax.negative.val.unc|n}} |
| Support Devices | {{labels.support_devices.negative.train.pos|n}} | {{labels.support_devices.negative.train.neg|n}} | {{labels.support_devices.negative.train.unc|n}} | {{labels.support_devices.negative.test.pos|n}} | {{labels.support_devices.negative.test.neg|n}} | {{labels.support_devices.negative.test.unc|n}} | {{labels.support_devices.negative.val.pos|n}} | {{labels.support_devices.negative.val.neg|n}} | {{labels.support_devices.negative.val.unc|n}} |
::note P: Positive, N: Negative, U: Uncertain. Đây là phân bố nhãn mô hình thực sự học.
::check all(D["labels"][k]["ignore"][s][c] == D["labels"][k]["negative"][s][c] for k in D["labels"] for s in ("train", "val", "test") for c in ("pos", "unc")) | Cột P hoặc U khác nhau giữa hai bảng
::check all(D["labels"][k]["negative"][s]["neg"] + D["labels"][k]["negative"][s]["pos"] + D["labels"][k]["negative"][s]["unc"] == D["splits"][s]["labelled"] for k in D["labels"] for s in ("train", "val", "test")) | Tổng P, N, U của Bảng 2.2 không bằng số nghiên cứu có nhãn

# So sánh với bảng của bài báo
Số liệu của chúng tôi không trùng với bảng thống kê nhãn trong bài báo META-CXR, và ở nhiều dòng
hai cột N và U của bài báo dường như bị đổi chỗ. Chẳng hạn với Atelectasis trên tập train, bài
báo ghi N bằng {{paper.atelectasis.train.neg|n}} và U bằng {{paper.atelectasis.train.unc|n}},
còn chúng tôi đếm được N bằng {{labels.atelectasis.ignore.train.neg|n}} và U bằng
{{labels.atelectasis.ignore.train.unc|n}}. Với Pneumothorax trên tập train, bài báo ghi U bằng
{{paper.pneumothorax.train.unc|n}}, gần với số Negative chúng tôi đếm được là
{{labels.pneumothorax.ignore.train.neg|n}}, trong khi số Uncertain của chúng tôi chỉ là
{{labels.pneumothorax.ignore.train.unc|n}}. Đây là nhận xét rút ra từ việc so sánh hai bảng,
chưa được tác giả xác nhận; số của bài báo được chép lại từ ảnh bảng và chưa đối chiếu với bản
PDF. Phần chênh lệch còn lại có thể do bài báo dùng phiên bản dữ liệu hoặc cách lọc khác.
::check D["paper"]["atelectasis"]["train"]["neg"] > D["paper"]["atelectasis"]["train"]["unc"] and D["labels"]["atelectasis"]["ignore"]["train"]["unc"] > D["labels"]["atelectasis"]["ignore"]["train"]["neg"] | Atelectasis không cho thấy N và U đổi chỗ
::check D["paper"]["pneumothorax"]["train"]["unc"] > D["paper"]["pneumothorax"]["train"]["neg"] and D["labels"]["pneumothorax"]["ignore"]["train"]["neg"] > D["labels"]["pneumothorax"]["ignore"]["train"]["unc"] | Pneumothorax không cho thấy N và U đổi chỗ
