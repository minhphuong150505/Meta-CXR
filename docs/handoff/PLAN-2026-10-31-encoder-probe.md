# PLAN 2026-10-31: đo encoder trên cùng dữ liệu — encoder nào cho phân loại tốt hơn?

**Trạng thái:** ĐÃ CHẠY 2026-10-31 — **KHÔNG CÓ ỨNG VIÊN** theo quy tắc đã chốt (xem Execution report).
User duyệt 2026-10-31 ("ok", đã chấp nhận điều khoản MedSigLIP). Chỉ đo, không train Stage 1,
không sửa YAML, không chấm test.

## 1. Câu hỏi và bằng chứng sẵn có

Năm hướng tăng recall (ASL, mở khóa encoder ở pilot và đủ budget, gộp epoch, TTA) đều không qua ngưỡng. Phép
đo 2026-10-11 (`~/stage1_meas_20261011/`, probe tuyến tính pooled, test 2.780 study): MHCAC 0,745 AUROC so
với một token pooled của MedCLIP-Swin 0,734; RAD-DINO 0,721; PubMedCLIP 0,638; BioViL-T qua projector ngẫu
nhiên 0,490. Tức gần như toàn bộ năng lực phân loại đến từ biểu diễn của Swin, phần MHCAC thêm chỉ ~0,01.

**Kiểm tra mới 2026-10-31 (CPU, chỉ đọc):**

- Backbone BioViL-T nạp đúng trọng số pretrained (362/364 tensor đổi khi nạp; 0 key lạ). Không có lỗi nạp hỏng.
- Đặc trưng thật của BioViL-T là **512 chiều**, lưới 14×14 (`patch_embeddings`); pipeline chiếu nó qua projector
  1408 chiều khởi tạo ngẫu nhiên. Projector 128 chiều đã pretrain nạp trọn vẹn khi dựng `joint_feature_size=128`.
- Chưa từng đo BioViL-T đọc đúng cách (512 chiều, pretrained).
- Tài liệu: MedSigLIP 0,846 và RAD-DINO 0,836 AUC trung bình trên embedding đóng băng (arXiv 2606.11606), nhưng
  pooling CLS làm mất tín hiệu tổn thương nhỏ; pooling cục bộ theo patch lấy lại được.

## 2. Encoder và cách đọc

| Encoder | Tiền xử lý | Vector toàn cục | Token lưới |
|---|---|---|---|
| MedCLIP-Swin (đang dùng) | `swin_image` của dataset | token pooled (768) | 7×7 (49 × 768) |
| PubMedCLIP (đang dùng) | `pubmedclip_image` của dataset | CLS sau LN (768) | 7×7 patch (49 × 768) |
| BioViL-T đọc đúng | `image` (448) của dataset | `img_embedding` (512) | 14×14 → gộp 7×7 (49 × 512) |
| BioViL-T 128 (mô tả) | như trên | trung bình projector pretrained (128) | — |
| RAD-DINO | gốc của nó: cạnh ngắn 518, cắt giữa 518 | [CLS ; trung bình patch] (1.536) | 37×37 → gộp 8×8 (64 × 768) |
| MedSigLIP-448 | gốc của nó: 448×448, mean/std 0,5 | [pooler ; trung bình patch] (2.304) | 32×32 → gộp 8×8 (64 × 1.152) |

Chỉ ảnh anchor của mỗi study (`max_aux_views=0`), ảnh sạch (augmentation tắt), encoder pretrained
(`--checkpoint none`), bf16.

**Tập:** 40.000 study train ngẫu nhiên (seed 16; chỉ study có nhãn dùng để fit, 10% làm holdout chọn siêu tham
số) và toàn bộ val 1.808 study. Test không đụng tới. Đặc trưng là dữ liệu dẫn xuất, chỉ nằm trên máy train
(`~/encprobe_20261031/`).

## 3. Probe (giống nhau cho mọi encoder)

- **Probe tuyến tính:** softmax 3 lớp cho từng finding trên vector toàn cục đã chuẩn hóa; L2 chọn trên holdout.
- **Probe attention (chính):** LayerNorm → chiếu 256 chiều + mã hóa vị trí học được; vector toàn cục thêm như
  một token; 14 query học được (một mỗi finding), attention một đầu, rồi tuyến tính ra 3 lớp. Chọn epoch trên
  holdout; 3 seed, lấy trung bình xác suất. Đây gần với cách MHCAC đọc token.
- **Tổ hợp (mô tả):** probe attention trên token nối của nhiều encoder (mỗi encoder chiếu riêng + embedding
  encoder): bộ ba đang dùng (Swin + BioViL-T + PubMedCLIP), Swin + từng ứng viên, và tất cả.

## 4. Quy tắc đọc, chốt TRƯỚC khi chạy

- **Chỉ số:** macro AP "có bệnh" (P hoặc U so với N, trên `s = p_pos/(p_pos+p_neg)`), trung bình 14 finding, val
  n = 1.808, cùng hàm với các pilot trước. Bootstrap ghép cặp theo study, 1.000 lần, seed 16.
- **Cổng G0:** hàm chỉ số tính lại MHCAC từ file npz đã lưu phải ra đúng 0,4329 (C42) và 0,4271 (R1C).
- **Một encoder X là ỨNG VIÊN cho một lần thay đổi Stage 1** khi đạt một trong hai:
  1. probe attention của X − probe attention của Swin ≥ +0,02, cận dưới CI > 0;
  2. probe attention tổ hợp Swin + X − probe attention Swin ≥ +0,02, cận dưới CI > 0 (X bổ sung thông tin).
- Ngưỡng 0,02 vì phần MHCAC thêm trên probe chỉ ~0,01: chênh nhỏ hơn khó còn lại sau khi train lại.
- **Không encoder nào đạt:** ghi kết quả âm; encoder không phải điểm nghẽn có thể sửa bằng cách đổi encoder.
- **Chỉ mô tả:** probe tuyến tính, AUROC "có bệnh", AUROC ba lớp trung bình, AP từng finding (nhất là finding
  hiếm), MHCAC (C42, R1C) trên cùng study.
- Mọi lần train lại Stage 1 với encoder mới là quyết định riêng của user.

## 5. Code

- `scripts/encoder_probe_features.py` (mới): trích đặc trưng theo bảng mục 2.
- `scripts/encoder_probe_fit.py` (mới): probe + chỉ số + bootstrap + kết luận.
- `tests/test_encoder_probe.py`: chỉ số AP/AUROC, tiền xử lý gốc của RAD-DINO/MedSigLIP, hình dạng probe.
- Cập nhật `struct/`, CLAUDE.md, README cùng commit.

**Chi phí:** trích đặc trưng khoảng 1–2 giờ GPU (bị giới hạn bởi giải mã ảnh), probe khoảng 30–60 phút GPU.
Ổ đĩa khoảng 25 GB.

## 6. Giới hạn

- MedSigLIP, RAD-DINO và MedCLIP đều được pretrain có MIMIC-CXR; split họ dùng chưa xác minh được, nên có thể
  có lợi thế rò dữ liệu.
- Probe fit trên 40.000 study, MHCAC train trên 220.216: so với MHCAC là thận trọng cho probe.
- Chỉ ảnh anchor; MHCAC có thêm view phụ.
- Một bộ siêu tham số probe cho mọi encoder; không dò riêng.

## Execution report

**Kết luận theo quy tắc đã chốt: KHÔNG CÓ ỨNG VIÊN.** Không encoder nào, đứng riêng hay ghép với Swin, hơn probe
attention của Swin ≥ +0,02.

**Chạy:**

- Chuỗi `~/encprobe_chain.sh` chạy đúng một lần từ `~/Meta-CXR-encprobe` (`e5c9d4d` + hai script và test của plan
  này). Đặc trưng val 00:46:57 → 00:48:43 (94 s, 1.786/1.808 study có nhãn); train 40.000 study 00:48:43 →
  01:22:02 (1.979 s, 39.615 có nhãn). Khoảng 8 GB GPU. MedSigLIP nạp đủ trọng số (không cảnh báo thiếu).
- Bước probe lần 1 dừng ở cổng G0 (01:22:04): macro AP khớp nhưng phép so nhãn sai, vì npz của eval hook ghi "không
  nhãn" là `-1`, còn probe ghi `-100` (đúng 22 study không có nhãn CheXpert; mọi ô có nhãn trùng khớp). Sửa phép so
  sánh để quy mọi giá trị âm về một ký hiệu, quy tắc đọc không đổi; chạy lại bước probe 01:23:40 → 01:40:22.
- **G0 qua:** macro AP tính lại 0,4329 (C42) và 0,4271 (R1C), đúng số đã ghi; nhãn khớp sau khi ghép theo key.
- Probe: fit 35.654 study, holdout 3.961, val 1.808.

**Macro AP "có bệnh" trên val (n = 1.808) / AUROC "có bệnh":**

| Đặc trưng | Tuyến tính | Attention |
|---|---|---|
| MHCAC C42 (pha 1b, mốc) | 0,4329 / 0,8053 | — |
| MHCAC R1C (mô hình đang báo cáo) | 0,4271 / 0,8017 | — |
| **MedCLIP-Swin** | **0,4264** / 0,7863 | **0,4263** / 0,7945 |
| BioViL-T 512 (đọc đúng) | 0,3984 / 0,7881 | 0,4039 / 0,7823 |
| MedSigLIP-448 | 0,3997 / 0,7720 | 0,3939 / 0,7851 |
| RAD-DINO | 0,3910 / 0,7707 | 0,3972 / 0,7861 |
| PubMedCLIP | 0,3181 / 0,7092 | 0,3155 / 0,7149 |
| Swin + BioViL-T | — | 0,4315 / 0,8004 |
| Swin + MedSigLIP | — | 0,4264 / 0,8019 |
| Swin + RAD-DINO | — | 0,4232 / 0,7950 |
| Swin + PubMedCLIP | — | 0,4240 / 0,7894 |
| Bộ ba đang dùng (Swin + BioViL-T + PubMedCLIP) | — | 0,4294 / 0,7974 |
| Cả năm encoder | — | 0,4274 / 0,8027 |

**Quy tắc (probe attention, Δ so với Swin, bootstrap ghép cặp 1.000 lần, seed 16):**

| So sánh | Δ macro AP [CI] |
|---|---|
| BioViL-T − Swin | −0,0224 [−0,0320; −0,0122] |
| MedSigLIP − Swin | −0,0323 [−0,0417; −0,0220] |
| RAD-DINO − Swin | −0,0291 [−0,0398; −0,0178] |
| PubMedCLIP − Swin | −0,1108 [−0,1237; −0,0970] |
| (Swin + BioViL-T) − Swin | +0,0052 [+0,0010; +0,0098] |
| (Swin + MedSigLIP) − Swin | +0,0001 [−0,0050; +0,0053] |
| (Swin + RAD-DINO) − Swin | −0,0031 [−0,0097; +0,0032] |
| (Swin + PubMedCLIP) − Swin | −0,0023 [−0,0068; +0,0017] |
| Swin − MHCAC R1C (mô tả) | −0,0008 [−0,0087; +0,0068] |
| Swin − MHCAC C42 (mô tả) | −0,0066 [−0,0175; +0,0031] |

**AP theo finding hiếm (mô tả; R1C / attn Swin / attn tốt nhất):** Pneumothorax 0,362 / 0,336 / 0,392 (Swin +
BioViL-T + PubMedCLIP và cả năm); Enlarged Cardiomediastinum 0,136 / 0,157 / 0,178 (BioViL-T); Lung Lesion 0,163 /
0,116 / 0,117 (tuyến tính MedSigLIP 0,160); Pleural Other 0,041 / 0,022 / 0,028; Fracture 0,030 / 0,020 / 0,024
(tuyến tính MedSigLIP 0,043). Không encoder nào làm các finding hiếm "đọc được": Pleural Other và Fracture ở dưới
0,05 AP với mọi đặc trưng.

**Đọc kết quả:**

- **MedCLIP-Swin là encoder mạnh nhất trên dữ liệu này**, đứng riêng hơn MedSigLIP, RAD-DINO và BioViL-T đọc đúng
  0,022–0,032 macro AP. Thứ hạng trong tài liệu (MedSigLIP, RAD-DINO đứng đầu) không giữ ở đây.
- **Một probe attention nhỏ trên Swin đóng băng, fit trên 40.000 study, đạt ngang mô hình đang báo cáo**
  (−0,0008 so với R1C). Toàn bộ phần MHCAC + hai encoder còn lại + 220.216 study train gần như không thêm gì trên
  biểu diễn của Swin.
- **Thêm encoder chỉ thêm rất ít:** tốt nhất là BioViL-T (+0,0052, CI trên 0 nhưng dưới ngưỡng 0,02). MedSigLIP
  không thêm gì (+0,0001).
- **Kết luận cho câu hỏi của user:** điểm nghẽn không nằm ở việc chọn encoder đóng băng. Đổi sang MedSigLIP hay
  RAD-DINO, hay đọc BioViL-T đúng cách, đều không vượt được trần của Swin ở chế độ đóng băng.

**Giới hạn:**

- MedCLIP (Swin) được pretrain trên MIMIC-CXR + CheXpert với split không xác minh được; MedSigLIP và RAD-DINO cũng có
  MIMIC-CXR. Nếu MedCLIP đã thấy ảnh val lúc pretrain, lợi thế của Swin có thể một phần là rò dữ liệu.
- Chỉ đặc trưng đóng băng; không đo encoder được tinh chỉnh.
- Chỉ ảnh anchor; MedSigLIP và RAD-DINO lấy lưới gộp 8×8, Swin/BioViL-T/PubMedCLIP 7×7.
- Một bộ siêu tham số probe chung; holdout là một phần của train, không phải val.
