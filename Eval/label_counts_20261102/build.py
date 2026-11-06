"""Build so_lieu.json and noi_dung.md for the CheXpert label-distribution document.

Input: label_counts_raw.json, written on the training host by
scripts/count_chexpert_blank_policy.py (counts only, no identifiers).
Run, then render:
    python3 build.py
    ~/.venvs/thesis-docs/bin/python ../../.claude/skills/thesis-docs/scripts/render_doc.py \
        noi_dung.md phan_bo_nhan_chexpert.docx --pdf
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
raw = json.loads((HERE / "label_counts_raw.json").read_text())

# Same row order as the paper's table.
LABELS = [
    ("Atelectasis", "atelectasis", "Atelectasis"),
    ("Cardiomegaly", "cardiomegaly", "Cardiomegaly"),
    ("Consolidation", "consolidation", "Consolidation"),
    ("Edema", "edema", "Edema"),
    ("Enlarged Cardiomediastinum", "enlarged_cardio", "Enlarged Cardio"),
    ("Fracture", "fracture", "Fracture"),
    ("Lung Lesion", "lung_lesion", "Lung Lesion"),
    ("Lung Opacity", "lung_opacity", "Lung Opacity"),
    ("No Finding", "no_finding", "No Finding"),
    ("Pleural Effusion", "pleural_effusion", "Pleural Effusion"),
    ("Pleural Other", "pleural_other", "Pleural Other"),
    ("Pneumonia", "pneumonia", "Pneumonia"),
    ("Pneumothorax", "pneumothorax", "Pneumothorax"),
    ("Support Devices", "support_devices", "Support Devices"),
]
SPLITS = ["train", "test", "val"]
POLICIES = ["ignore", "negative"]

# Transcribed by hand from the image of the paper's table that the user sent
# (2026-11-02), rows used in section 4 only. Not checked against the PDF.
PAPER = {
    "atelectasis": {"train": {"pos": 48181, "neg": 10759, "unc": 1579}},
    "pneumothorax": {"train": {"pos": 10988, "neg": 1193, "unc": 44292}},
}

data = {"labels": {}, "splits": {}, "paper": PAPER}
for split in SPLITS:
    prov = raw["splits"][split]["negative"]["ignore_provenance"]
    data["splits"][split] = {
        "studies": prov["studies"],
        "labelled": raw["splits"][split]["negative"]["classification_valid"],
        "no_record": prov["no_chexpert_record"],
        "all_blank": prov["record_all_blank"],
    }
    data["splits"][split]["unlabelled"] = prov["no_chexpert_record"] + prov["record_all_blank"]
for name, key, _ in LABELS:
    data["labels"][key] = {
        pol: {
            split: {k: raw["splits"][split][pol]["per_label"][name][k] for k in ("pos", "neg", "unc")}
            for split in SPLITS
        }
        for pol in POLICIES
    }
(HERE / "so_lieu.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))


def table(policy, caption, note):
    lines = [
        f"::table {caption}",
        "::widths 3.3, 1.6, 1.6, 1.6, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25",
        "::font 9",
        "| Lớp | Train P | Train N | Train U | Test P | Test N | Test U | Val P | Val N | Val U |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, key, short in LABELS:
        cells = [
            f"{{{{labels.{key}.{policy}.{split}.{c}|n}}}}"
            for split in SPLITS for c in ("pos", "neg", "unc")
        ]
        lines.append(f"| {short} | " + " | ".join(cells) + " |")
    lines.append(f"::note {note}")
    return "\n".join(lines)


md = f"""---
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
có {{{{splits.train.studies|n}}}} nghiên cứu, tập val có {{{{splits.val.studies|n}}}} và tập
test có {{{{splits.test.studies|n}}}}. Trong đó lần lượt {{{{splits.train.unlabelled|n}}}},
{{{{splits.val.unlabelled|n}}}} và {{{{splits.test.unlabelled|n}}}} nghiên cứu không có thông
tin CheXpert, do không có bản ghi nhãn hoặc có bản ghi nhưng cả 14 ô đều trống. Các nghiên cứu
này không được tính vào bất kỳ ô nào của hai bảng dưới đây, nên số nghiên cứu có nhãn là
{{{{splits.train.labelled|n}}}} ở train, {{{{splits.val.labelled|n}}}} ở val và
{{{{splits.test.labelled|n}}}} ở test.
::check all(D["splits"][s]["studies"] - D["splits"][s]["unlabelled"] == D["splits"][s]["labelled"] for s in ("train", "val", "test")) | Số nghiên cứu có nhãn không khớp

# Phân bố nhãn

## Theo nhãn gốc
Bảng 2.1 đếm nhãn đúng như file CheXpert ghi, ô trống không được tính. Đây là cùng cách đếm với
bảng thống kê nhãn trong bài báo META-CXR: lớp No Finding chỉ có Positive vì công cụ gán nhãn
không bao giờ ghi Negative cho lớp này.

{table("ignore", "Số nghiên cứu theo nhãn gốc, ô trống không tính",
       "P: Positive, N: Negative, U: Uncertain. Đếm ở mức nghiên cứu.")}
::check all(D["labels"]["no_finding"]["ignore"][s]["neg"] == 0 and D["labels"]["no_finding"]["ignore"][s]["unc"] == 0 for s in ("train", "val", "test")) | No Finding có Negative hoặc Uncertain theo nhãn gốc

::pagebreak
## Khi ô trống được coi là Negative
Bảng 2.2 đếm theo cách mô hình hiện tại được huấn luyện, tương ứng với thiết lập
blank_label_policy bằng negative, giống cách bài báo gốc xử lý giá trị thiếu. Các cột P và U giữ
nguyên như Bảng 2.1; chỉ cột N tăng lên vì mọi ô trống của nghiên cứu có thông tin CheXpert được
tính là Negative. Khi đó Negative là lớp đông nhất ở mọi bệnh, và No Finding có
{{{{labels.no_finding.negative.train.neg|n}}}} nghiên cứu Negative trên tập train.

{table("negative", "Số nghiên cứu khi ô trống được coi là Negative",
       "P: Positive, N: Negative, U: Uncertain. Đây là phân bố nhãn mô hình thực sự học.")}
::check all(D["labels"][k]["ignore"][s][c] == D["labels"][k]["negative"][s][c] for k in D["labels"] for s in ("train", "val", "test") for c in ("pos", "unc")) | Cột P hoặc U khác nhau giữa hai bảng
::check all(D["labels"][k]["negative"][s]["neg"] + D["labels"][k]["negative"][s]["pos"] + D["labels"][k]["negative"][s]["unc"] == D["splits"][s]["labelled"] for k in D["labels"] for s in ("train", "val", "test")) | Tổng P, N, U của Bảng 2.2 không bằng số nghiên cứu có nhãn

# So sánh với bảng của bài báo
Số liệu của chúng tôi không trùng với bảng thống kê nhãn trong bài báo META-CXR, và ở nhiều dòng
hai cột N và U của bài báo dường như bị đổi chỗ. Chẳng hạn với Atelectasis trên tập train, bài
báo ghi N bằng {{{{paper.atelectasis.train.neg|n}}}} và U bằng {{{{paper.atelectasis.train.unc|n}}}},
còn chúng tôi đếm được N bằng {{{{labels.atelectasis.ignore.train.neg|n}}}} và U bằng
{{{{labels.atelectasis.ignore.train.unc|n}}}}. Với Pneumothorax trên tập train, bài báo ghi U bằng
{{{{paper.pneumothorax.train.unc|n}}}}, gần với số Negative chúng tôi đếm được là
{{{{labels.pneumothorax.ignore.train.neg|n}}}}, trong khi số Uncertain của chúng tôi chỉ là
{{{{labels.pneumothorax.ignore.train.unc|n}}}}. Đây là nhận xét rút ra từ việc so sánh hai bảng,
chưa được tác giả xác nhận; số của bài báo được chép lại từ ảnh bảng và chưa đối chiếu với bản
PDF. Phần chênh lệch còn lại có thể do bài báo dùng phiên bản dữ liệu hoặc cách lọc khác.
::check D["paper"]["atelectasis"]["train"]["neg"] > D["paper"]["atelectasis"]["train"]["unc"] and D["labels"]["atelectasis"]["ignore"]["train"]["unc"] > D["labels"]["atelectasis"]["ignore"]["train"]["neg"] | Atelectasis không cho thấy N và U đổi chỗ
::check D["paper"]["pneumothorax"]["train"]["unc"] > D["paper"]["pneumothorax"]["train"]["neg"] and D["labels"]["pneumothorax"]["ignore"]["train"]["neg"] > D["labels"]["pneumothorax"]["ignore"]["train"]["unc"] | Pneumothorax không cho thấy N và U đổi chỗ
"""
(HERE / "noi_dung.md").write_text(md)
print("wrote so_lieu.json, noi_dung.md")
