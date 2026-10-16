"""Chapter 1 data statistics and preprocessing sections. Called from build.py."""
from fmt import f, pct, n

LABEL_VI = {
    "No Finding": "Không có bất thường (No Finding)", "Enlarged Cardiomediastinum": "Giãn trung thất (Enlarged Cardiomediastinum)",
    "Cardiomegaly": "Tim to (Cardiomegaly)", "Lung Opacity": "Mờ phổi (Lung Opacity)", "Lung Lesion": "Tổn thương phổi (Lung Lesion)",
    "Edema": "Phù phổi (Edema)", "Consolidation": "Đông đặc (Consolidation)", "Pneumonia": "Viêm phổi (Pneumonia)",
    "Atelectasis": "Xẹp phổi (Atelectasis)", "Pneumothorax": "Tràn khí màng phổi (Pneumothorax)",
    "Pleural Effusion": "Tràn dịch màng phổi (Pleural Effusion)", "Pleural Other": "Bất thường màng phổi khác (Pleural Other)",
    "Fracture": "Gãy xương (Fracture)", "Support Devices": "Thiết bị hỗ trợ (Support Devices)",
}
SPLIT_VI = {"train": "Huấn luyện", "val": "Kiểm định", "test": "Kiểm tra"}


def views(d, key):
    v = d[key]
    lat = v.get("LATERAL", 0) + v.get("LL", 0)
    known = v.get("PA", 0) + v.get("AP", 0) + lat
    return v.get("PA", 0), v.get("AP", 0), lat, sum(v.values()) - known


def write(T, DS, CFG, c):
    raw, sp, lk, L = DS["raw"], DS["splits"], DS["leakage"], DS["labels"]
    tr, va, te = sp["train"], sp["val"], sp["test"]
    tot_studies = sum(sp[s]["studies"] for s in sp)
    tot_images = sum(sp[s]["images"] for s in sp)

    T.sub_heading("Thống kê dữ liệu đầu vào")
    T.para(
        "Dữ liệu đầu vào gồm ba tệp siêu dữ liệu đi kèm MIMIC-CXR-JPG và cây thư mục báo cáo văn bản của MIMIC-CXR. Tệp "
        "metadata mô tả từng ảnh (tư thế chụp, kích thước), tệp split gán mỗi ảnh vào một tập theo phân chia chính thức, "
        "và tệp chexpert chứa nhãn 14 bệnh lý ở mức nghiên cứu. Bảng dưới đây thống kê các nguồn này trước mọi bước lọc.")
    rows_img, cols_img = raw["image_rows"], raw["image_columns"]
    T.table("Thống kê dữ liệu thô của MIMIC-CXR-JPG trước tiền xử lý",
            ["Đại lượng", "Giá trị"],
            [["Số ảnh trong tệp metadata", n(raw["metadata_images"])],
             ["Số nghiên cứu (lần chụp)", n(raw["metadata_studies"])],
             ["Số bệnh nhân", n(raw["metadata_subjects"])],
             ["Số nghiên cứu có bản ghi nhãn CheXpert", n(raw["chexpert_studies"])],
             ["Số bản ghi nhãn để trống toàn bộ 14 bệnh lý", n(raw["chexpert_all_blank"])],
             ["Tỉ lệ ô nhãn để trống trong ma trận CheXpert", pct(raw["chexpert_blank_fraction"])],
             ["Số ảnh có cạnh nhỏ hơn 100 điểm ảnh", n(raw["images_below_100px"])],
             ["Chiều cao ảnh, trung vị (khoảng 5% đến 95%)",
              f"{n(rows_img['median'])} ({n(rows_img['p5'])} đến {n(rows_img['p95'])}) điểm ảnh"],
             ["Chiều rộng ảnh, trung vị (khoảng 5% đến 95%)",
              f"{n(cols_img['median'])} ({n(cols_img['p5'])} đến {n(cols_img['p95'])}) điểm ảnh"]],
            widths=[10.4, 6.0])
    vr = dict(raw["view_counts_all_images"])
    rare = {k: v for k, v in vr.items() if k not in ("PA", "AP", "LATERAL", "LL", "UNKNOWN")}
    main = [(k, vr[k]) for k in ("AP", "PA", "LATERAL", "LL", "UNKNOWN") if k in vr]
    vrows = [[k, n(v), pct(v / raw["metadata_images"])] for k, v in main]
    vrows.append([f"Tư thế hiếm khác ({len(rare)} loại)", n(sum(rare.values())), pct(sum(rare.values()) / raw["metadata_images"])])
    T.table("Phân bố tư thế chụp trên toàn bộ ảnh",
            ["Tư thế chụp", "Số ảnh", "Tỉ lệ"], vrows, widths=[6.4, 5.0, 5.0],
            note="PA: sau trước; AP: trước sau; LATERAL và LL: nghiêng; UNKNOWN: không ghi tư thế trong siêu dữ liệu. "
                 "Tư thế hiếm được mô hình xếp chung nhóm không rõ tư thế.")

    T.sub_heading("Quy trình tiền xử lý")
    T.para(
        "Tiền xử lý được thực hiện bằng tập lệnh preporcessing/preprocess_mimic_cxr.py, chạy trên CPU và không đọc ảnh, "
        "tạo ra bộ manifest full_allviews_v2 gồm ba tệp train, val và test. Quy trình gồm bảy bước.")
    T.para(
        "Bước 1, làm sạch nhãn: kiểm tra mỗi cặp bệnh nhân và nghiên cứu chỉ có một bản ghi nhãn, giữ lại cả nghiên cứu "
        "không có nhãn nào và đánh dấu bằng cờ has_chexpert_label. Bước 2, làm sạch siêu dữ liệu: loại dòng trùng lặp hoàn "
        "toàn, kiểm tra tính duy nhất theo mã ảnh, loại ảnh có cạnh nhỏ hơn 100 điểm ảnh, và gán tư thế UNKNOWN cho ảnh "
        "không ghi tư thế thay vì loại bỏ. Bước 3, làm sạch phân chia: chuẩn hóa tên tập và kiểm tra rằng không bệnh nhân "
        "nào và không nghiên cứu nào xuất hiện ở hai tập. Bước 4, ghép ba nguồn theo khóa bệnh nhân, nghiên cứu và ảnh với "
        "ràng buộc một đối một, để phép ghép không thể nhân bản dòng một cách âm thầm.")
    T.para(
        "Bước 5, trích xuất văn bản: đọc báo cáo của từng nghiên cứu, làm sạch và tách phần FINDINGS và IMPRESSION bằng "
        "bộ phân tích mimic_report_parser.py. Khi báo cáo không có tiêu đề FINDINGS, bộ phân tích khôi phục phần tường "
        "thuật thay vì lấy cả báo cáo, và không bao giờ dùng IMPRESSION để thay cho FINDINGS. Bước 6, lọc độ dài: đếm số "
        "token từ vựng của FINDINGS, đánh dấu không hợp lệ các báo cáo dưới 3 token hoặc vượt phân vị 99% độ dài của tập "
        "huấn luyện; ngưỡng trên chỉ được tính từ tập huấn luyện để tập kiểm tra không ảnh hưởng tới quy trình. IMPRESSION "
        "được lọc riêng theo phân phối độ dài của chính nó. Báo cáo không hợp lệ bị xóa trắng văn bản đích nhưng nghiên cứu "
        "vẫn được giữ lại cho bài toán phân loại. Bước 7, sinh đường dẫn ảnh tương đối và ghi ba tệp manifest.")
    T.table("Các tham số và quy tắc lọc của quy trình tiền xử lý",
            ["Quy tắc", "Giá trị", "Hệ quả"],
            [["Tư thế chụp được giữ", "Tất cả", "Phục vụ hợp nhất nhiều góc nhìn"],
             ["Kích thước ảnh tối thiểu", "100 điểm ảnh mỗi cạnh", f"Loại {n(raw['images_below_100px'])} ảnh"],
             ["Số token FINDINGS tối thiểu", "3", "Nhãn lọc TOO_SHORT"],
             ["Số token FINDINGS tối đa", "Phân vị 99% của tập huấn luyện", "Nhãn lọc TOO_LONG"],
             ["Không trích được FINDINGS", "Không áp dụng", "Nhãn lọc NO_FINDINGS"],
             ["Nghiên cứu không có nhãn", "Giữ lại", "Bị che khỏi hàm mất mát phân loại"],
             ["Ô nhãn để trống", "Giữ giá trị bỏ qua (-100)", "Không bị coi là âm tính"],
             ["Phân chia dữ liệu", "Tệp split chính thức", "Không tự chia lại"]],
            widths=[5.2, 5.4, 5.8])

    T.sub_heading("Phân chia tập huấn luyện, kiểm định và kiểm tra")
    T.para(
        "Đề tài không tự chia dữ liệu mà dùng nguyên phân chia chính thức của MIMIC-CXR, trong đó mỗi bệnh nhân chỉ thuộc "
        "một tập. Phép kiểm tra trên manifest sau tiền xử lý cho kết quả "
        f"{lk['subject_train_val'] + lk['subject_train_test'] + lk['subject_val_test']} bệnh nhân và "
        f"{lk['study_train_val'] + lk['study_train_test'] + lk['study_val_test']} nghiên cứu xuất hiện ở hơn một tập, "
        f"và {lk['duplicate_dicom']} mã ảnh bị trùng. Tổng cộng manifest có {n(tot_images)} ảnh thuộc {n(tot_studies)} "
        "nghiên cứu. Tỉ lệ các tập rất lệch về phía huấn luyện, là đặc điểm của phân chia chính thức: tập kiểm định và "
        "kiểm tra nhỏ, nên các khoảng tin cậy trong tài liệu được ước lượng bằng bootstrap thay vì dựa vào một điểm số.")
    rows = []
    for s in ("train", "val", "test"):
        d = sp[s]
        rows.append([SPLIT_VI[s], n(d["subjects"]), n(d["studies"]), pct(d["studies"] / tot_studies), n(d["images"]),
                     f(d["images"] / d["studies"], 2), n(d["has_chexpert_record"])])
    rows.append([("Tổng", True), (n(sum(sp[s]["subjects"] for s in sp)), True), (n(tot_studies), True), ("100,0%", True),
                 (n(tot_images), True), (f(tot_images / tot_studies, 2), True),
                 (n(sum(sp[s]["has_chexpert_record"] for s in sp)), True)])
    T.table("Số lượng bệnh nhân, nghiên cứu và ảnh theo từng tập",
            ["Tập", "Bệnh nhân", "Nghiên cứu", "Tỉ lệ nghiên cứu", "Ảnh", "Ảnh trên nghiên cứu", "Nghiên cứu có nhãn"],
            rows, widths=[2.6, 2.2, 2.4, 2.2, 2.3, 2.2, 2.5], font_size=10)
    rows = []
    for s in ("train", "val", "test"):
        d = sp[s]
        pa, ap, lat, unk = views(d, "view_counts_images")
        rows.append([SPLIT_VI[s], n(pa), n(ap), n(lat), n(unk)])
    T.table("Phân bố tư thế chụp theo ảnh trong từng tập",
            ["Tập", "PA", "AP", "Nghiêng", "Không rõ hoặc hiếm"], rows, widths=[3.4, 3.2, 3.2, 3.3, 3.3])
    T.para(
        "Mô hình Giai đoạn 1 lấy mẫu theo nghiên cứu: mỗi nghiên cứu chọn một ảnh chính theo thứ tự ưu tiên PA, AP, nghiêng, "
        "và tối đa một ảnh phụ có tư thế khác ảnh chính. Ảnh chụp lặp lại cùng tư thế không được dùng làm ảnh phụ vì không "
        "bổ sung góc nhìn mới. Bảng sau cho thấy số nghiên cứu có ảnh phụ thực sự, tức là số nghiên cứu mà mô-đun hợp nhất "
        "góc nhìn có dữ liệu để hoạt động.")
    rows = []
    for s in ("train", "val", "test"):
        d = sp[s]
        a = d["anchor_view_counts"]
        ips = d["images_per_study"]
        rows.append([SPLIT_VI[s], n(a.get("PA", 0)), n(a.get("AP", 0)), n(a.get("LATERAL", 0)), n(a.get("UNKNOWN", 0)),
                     n(d["studies_multi_image"]), f"{n(d['studies_with_aux_view'])} ({pct(d['studies_with_aux_view'] / d['studies'])})",
                     f"{n(ips['median'])} (tối đa {d['images_per_study_max']})"])
    T.table("Ảnh chính, ảnh phụ và số ảnh trên mỗi nghiên cứu",
            ["Tập", "Ảnh chính PA", "Ảnh chính AP", "Ảnh chính nghiêng", "Ảnh chính không rõ", "Nghiên cứu nhiều ảnh",
             "Có ảnh phụ khác tư thế", "Số ảnh, trung vị"],
            rows, widths=[2.2, 1.9, 1.9, 1.9, 1.9, 2.0, 2.5, 2.1], font_size=9)

    T.sub_heading("Thống kê văn bản báo cáo")
    T.para(
        "Mục tiêu sinh của Giai đoạn 2 là phần FINDINGS. Bảng dưới đây thống kê kết quả trích xuất và lọc độ dài ở mức "
        "nghiên cứu. Nhóm hợp lệ là các nghiên cứu có báo cáo tham chiếu dùng để huấn luyện và chấm điểm sinh báo cáo.")
    rows = []
    for s in ("train", "val", "test"):
        d = sp[s]
        r = d["filter_reason"]
        rows.append([SPLIT_VI[s], n(d["studies"]), f"{n(r.get('VALID', 0))} ({pct(r.get('VALID', 0) / d['studies'])})",
                     n(r.get("NO_FINDINGS", 0)), n(r.get("TOO_SHORT", 0)), n(r.get("TOO_LONG", 0)), n(d["impression_valid"]),
                     n(d["findings_and_impression_valid"])])
    T.table("Kết quả lọc phần FINDINGS và IMPRESSION theo nghiên cứu",
            ["Tập", "Nghiên cứu", "FINDINGS hợp lệ", "Không có FINDINGS", "Quá ngắn", "Quá dài", "IMPRESSION hợp lệ",
             "Có cả hai"],
            rows, widths=[2.2, 1.9, 2.6, 2.0, 1.5, 1.5, 2.5, 2.2], font_size=9)
    rows = []
    for s in ("train", "val", "test"):
        for sec, key in (("FINDINGS", "findings_tokens_valid"), ("IMPRESSION", "impression_tokens_valid")):
            a = sp[s][key]
            rows.append([SPLIT_VI[s], sec, n(a["n"]), f(a["mean"], 1), n(a["median"]), f"{n(a['p5'])} đến {n(a['p95'])}",
                         n(a["max"])])
    T.table("Độ dài phần FINDINGS và IMPRESSION hợp lệ, tính bằng số token từ vựng",
            ["Tập", "Phần báo cáo", "Số nghiên cứu", "Trung bình", "Trung vị", "Khoảng 5% đến 95%", "Tối đa"],
            rows, widths=[2.4, 2.6, 2.3, 2.1, 1.9, 2.9, 2.2], font_size=10)
    em = tr["extraction_method"]
    em_rows = sorted(em.items(), key=lambda kv: -kv[1])
    EM_VI = {"FINDINGS_TAG": "Có tiêu đề FINDINGS, lấy trực tiếp",
             "IMPRESSION_ONLY": "Chỉ có IMPRESSION, không tạo đích FINDINGS",
             "NARRATIVE_BODY": "Không có tiêu đề, khôi phục phần tường thuật",
             "FINDINGS_IMPRESSION_COMBINED": "FINDINGS và IMPRESSION viết gộp",
             "NARRATIVE_AFTER_COMPARISON": "Tường thuật nằm sau mục so sánh",
             "NO_FINDINGS": "Không trích được nội dung"}
    rows = []
    for k, _ in em_rows:
        rows.append([k, EM_VI.get(k, "")] + [n(sp[s]["extraction_method"].get(k, 0)) for s in ("train", "val", "test")])
    T.table("Cách trích xuất văn bản đích theo từng tập (số nghiên cứu)",
            ["Phương thức", "Ý nghĩa", "Huấn luyện", "Kiểm định", "Kiểm tra"],
            rows, widths=[6.0, 4.6, 2.0, 1.9, 1.9], font_size=8.5,
            note="Tên phương thức là giá trị cột extraction_method do bộ phân tích báo cáo ghi lại.")
    T.para(
        f"Trên tập huấn luyện, {n(tr['findings_valid_le15_tokens'])} nghiên cứu có FINDINGS hợp lệ nhưng không quá 15 token, "
        f"chiếm {pct(tr['findings_valid_le15_tokens'] / tr['filter_reason'].get('VALID', 1))} số báo cáo đích hợp lệ. Đây là "
        "các báo cáo một dòng, phần lớn mô tả kỹ thuật chụp; chúng vượt qua ngưỡng tối thiểu 3 token nên vẫn được dùng làm "
        "mục tiêu sinh, và được ghi nhận như một hạn chế của dữ liệu.")

    def ap_share(d):
        a = d["anchor_view_counts"]; return a.get("AP", 0) / d["studies"]
    T.para(
        "Ba tập không đồng nhất về phân phối, và điều này cần được lưu ý khi so sánh kết quả kiểm định với kiểm tra. Tập "
        f"kiểm tra chỉ gồm {n(te['subjects'])} bệnh nhân cho {n(te['studies'])} nghiên cứu, trung bình "
        f"{f(te['studies'] / te['subjects'], 1)} nghiên cứu mỗi bệnh nhân, so với {f(tr['studies'] / tr['subjects'], 1)} ở tập "
        f"huấn luyện và {f(va['studies'] / va['subjects'], 1)} ở tập kiểm định; các nghiên cứu của cùng một bệnh nhân tương "
        "quan với nhau, nên bootstrap theo nghiên cứu có thể cho khoảng tin cậy hẹp hơn thực tế. Tỉ lệ ảnh chính chụp tư thế "
        f"AP, thường gặp ở bệnh nhân nằm tại giường, là {pct(ap_share(te))} ở tập kiểm tra so với {pct(ap_share(tr))} ở tập "
        f"huấn luyện. Trung vị độ dài FINDINGS hợp lệ của tập kiểm tra là {n(te['findings_tokens_valid']['median'])} token, "
        f"dài hơn {n(tr['findings_tokens_valid']['median'])} token của tập huấn luyện, và tỉ lệ ô nhãn để trống thấp hơn "
        f"({pct(te['blank_fraction'])} so với {pct(tr['blank_fraction'])}). Các khác biệt này gợi ý tập kiểm tra chứa nhiều ca "
        "có nhiều phát hiện được mô tả hơn, nhưng đề tài không có dữ liệu lâm sàng để kiểm chứng mức độ nặng.")

    T.sub_heading("Thống kê nhãn theo từng bệnh lý")
    T.para(
        "Nhãn CheXpert có bốn trạng thái: dương tính (P), âm tính (N), không chắc chắn (U) và để trống khi báo cáo không nhắc "
        "tới bệnh lý. Ba bảng dưới đây đếm số nghiên cứu theo từng trạng thái cho mỗi nhãn và mỗi tập. Cột tỉ lệ hiện diện là "
        "số nghiên cứu dương tính chia cho tổng số nghiên cứu của tập, tương ứng với cách diễn giải study_presence dùng để "
        "báo cáo F1; cột tỉ lệ dương tính khi được nhắc là P chia cho tổng P, N và U, tương ứng với cách diễn giải "
        "masked_polarity mà hàm mất mát phân loại sử dụng.")
    for s in ("train", "val", "test"):
        d = sp[s]
        rows = []
        for lab in L:
            x = d["label_counts"][lab]
            mentioned = x["P"] + x["N"] + x["U"]
            rows.append([LABEL_VI[lab], n(x["P"]), n(x["N"]), n(x["U"]), n(x["blank"]), pct(x["P"] / d["studies"]),
                         pct(x["P"] / mentioned) if mentioned else "không xác định"])
        T.table(f"Phân bố nhãn P, N, U và để trống trên tập {SPLIT_VI[s].lower()} ({n(d['studies'])} nghiên cứu)",
                ["Bệnh lý", "P", "N", "U", "Để trống", "Tỉ lệ hiện diện", "Tỉ lệ P khi được nhắc"],
                rows, widths=[5.6, 1.7, 1.6, 1.6, 1.8, 2.0, 2.1], font_size=9,
                note=(f"{n(d['studies_missing_chexpert_row'])} nghiên cứu của tập này không có bản ghi nhãn và được tính vào "
                      "cột để trống." if d["studies_missing_chexpert_row"] else None))
    # derived summary
    def lab_sum(d, lab): return d["label_counts"][lab]
    rows = []
    for s in ("train", "val", "test"):
        d = sp[s]
        rows.append([SPLIT_VI[s], pct(d["blank_fraction"]), f(d["labels_per_study_nonblank"]["mean"], 2),
                     f(d["positives_per_study"]["mean"], 2), n(d["studies_only_no_finding"])])
    T.table("Mức độ thưa của ma trận nhãn",
            ["Tập", "Tỉ lệ ô để trống", "Số nhãn được nhắc trên nghiên cứu", "Số nhãn dương tính trên nghiên cứu",
             "Nghiên cứu chỉ có nhãn No Finding"],
            rows, widths=[2.8, 2.9, 3.6, 3.6, 3.5], font_size=10)
    nf = te["label_counts"]["No Finding"]; po = te["label_counts"]["Pleural Other"]; fr = te["label_counts"]["Fracture"]
    rare = min(((lab, te["label_counts"][lab]["P"]) for lab in L), key=lambda kv: kv[1])
    at_tr = tr["label_counts"]["Atelectasis"]
    n_pmaj = sum(1 for lab in L if tr["label_counts"][lab]["P"] > tr["label_counts"][lab]["N"])
    T.para(
        "Ba đặc điểm của phân bố nhãn quyết định cách thiết kế hàm mất mát và độ đo. Thứ nhất, ma trận rất thưa: "
        f"{pct(tr['blank_fraction'])} ô trên tập huấn luyện để trống, mỗi nghiên cứu trung bình chỉ có "
        f"{f(tr['labels_per_study_nonblank']['mean'], 2)} trên 14 nhãn được nhắc tới. Thứ hai, khi chỉ xét các ô được nhắc, "
        f"mất cân bằng bị đảo ngược so với trực giác: {n_pmaj} trên 14 bệnh lý có số ô dương tính nhiều hơn âm tính trên tập "
        f"huấn luyện, ví dụ Atelectasis có {n(at_tr['P'])} ô dương tính so với {n(at_tr['N'])} ô âm tính, phù hợp với việc "
        "báo cáo thường chỉ nhắc tới một bệnh lý khi có dấu hiệu của nó. Đây là lý do F1 dưới cách diễn giải masked_polarity bị thổi phồng và đề tài chỉ báo cáo F1 "
        f"dưới study_presence. Thứ ba, một số nhãn suy biến: No Finding không bao giờ có giá trị âm tính ({n(nf['N'])} ô N "
        f"trên tập kiểm tra), Pleural Other có {n(po['P'])} P và {n(po['N'])} N, Fracture có {n(fr['P'])} P và {n(fr['N'])} N "
        f"trên tập kiểm tra, và nhãn hiếm nhất trên tập kiểm tra là {rare[0]} với {n(rare[1])} nghiên cứu dương tính. Các nhãn "
        "này được diễn giải thận trọng trong bảng kết quả từng nhãn.")

    T.sub_heading("Xử lý ảnh và nhãn khi huấn luyện")
    T.para(
        "Ảnh JPG được đọc dưới dạng thang xám, thu nhỏ cạnh ngắn về 512 điểm ảnh và giữ tỉ lệ, sau đó cắt vùng trung tâm "
        "448 nhân 448 và nhân bản thành ba kênh. BioViL-T nhận ảnh 448 và tạo lưới 14 nhân 14 token; PubMedCLIP nhận bản thu "
        "nhỏ 224 của cùng vùng cắt, nên hai bộ mã hóa dùng chung một hệ tọa độ. Tăng cường dữ liệu chỉ áp dụng cho tập huấn "
        "luyện; ảnh chính và ảnh phụ của một nghiên cứu được biến đổi ngẫu nhiên độc lập với nhau. Tập kiểm định và kiểm tra "
        "chỉ qua thu nhỏ và cắt trung tâm. Giai đoạn 2 dùng bộ xử lý ảnh gốc của MedGemma, đưa ảnh về 896 nhân 896 không giữ "
        "tỉ lệ và không cắt.")
    T.table("Các phép biến đổi ảnh",
            ["Phép biến đổi", "Tham số", "Áp dụng cho"],
            [["Thu nhỏ cạnh ngắn", "512 điểm ảnh, giữ tỉ lệ", "Mọi tập, Giai đoạn 1"],
             ["Cắt trung tâm", "448 nhân 448", "Mọi tập, Giai đoạn 1"],
             ["Biến đổi affine ngẫu nhiên", "Xoay 5 độ, tịnh tiến 2%, co giãn 5%, xác suất 0,5", "Tập huấn luyện"],
             ["Thay đổi độ sáng và tương phản", "Biên độ 0,1, xác suất 0,5", "Tập huấn luyện"],
             ["Bộ xử lý ảnh MedGemma", "896 nhân 896, không giữ tỉ lệ", "Mọi tập, Giai đoạn 2"]],
            widths=[5.0, 6.8, 4.6])
    T.para(
        "Nhãn được ánh xạ thành ba mục tiêu khác nhau tùy mục đích. Bảng sau tóm tắt cách một ô nhãn gốc được chuyển thành "
        "mục tiêu của đầu phân loại, mục tiêu của cổng nhắc tới, và nhãn dùng khi đánh giá.")
    T.table("Ánh xạ nhãn CheXpert thành mục tiêu huấn luyện và đánh giá",
            ["Giá trị gốc", "Đầu phân loại P/N/U", "Cổng nhắc tới", "Đánh giá study_presence"],
            [["1 (dương tính)", "Lớp dương tính", "1", "Có bệnh"],
             ["0 (âm tính)", "Lớp âm tính", "1", "Không có bệnh"],
             ["-1 (không chắc chắn)", "Lớp không chắc chắn", "1", "Không có bệnh"],
             ["Để trống", "Bỏ qua (-100), không tính mất mát", "0", "Không có bệnh"],
             ["Nghiên cứu không có bản ghi nhãn", "Bỏ qua toàn bộ", "Bỏ qua toàn bộ", "Không có bệnh"]],
            widths=[4.2, 4.6, 3.0, 4.6])

    # values the rest of the build asserts on
    return {"leak_total": lk["subject_train_val"] + lk["subject_train_test"] + lk["subject_val_test"]
            + lk["study_train_val"] + lk["study_train_test"] + lk["study_val_test"] + lk["duplicate_dicom"],
            "val_studies": va["studies"], "test_studies": te["studies"], "train_studies": tr["studies"]}
