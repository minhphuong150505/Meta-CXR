# Tổng hợp toàn bộ đợt điều tra 2026-10-10/11 (một file)

File này gộp năm tài liệu của đợt điều tra thành một bản đọc liền mạch:

| Nguồn | Nội dung | Nằm ở mục |
|---|---|---|
| [`TONG_HOP_2026-10-11.md`](TONG_HOP_2026-10-11.md) | chẩn đoán Stage 1 / Stage 2, tài liệu, đề xuất | 2–6, 10, 12–14 |
| [`CHI_MUC_TINH_MOI.md`](CHI_MUC_TINH_MOI.md) | chỉ mục khác biệt so với bài báo (TM-01…TM-29) | 9 |
| [`handoff/PLAN-2026-10-11-soft-token-probe.md`](handoff/PLAN-2026-10-11-soft-token-probe.md) | linear probe soft token | 3.4 |
| [`handoff/PLAN-2026-10-11-stage1-improvement.md`](handoff/PLAN-2026-10-11-stage1-improvement.md) | nghiên cứu cải thiện Stage 1, đo M1–M5, pilot BioViL | 4–5 |
| [`handoff/PLAN-2026-10-11-stage2-context-pilot.md`](handoff/PLAN-2026-10-11-stage2-context-pilot.md) | pilot Stage 2 P0/P1/P2 và lần train toàn bộ | 7–8 |

Năm file gốc vẫn giữ nguyên và là nơi ghi chi tiết; khi hai nơi lệch nhau, file gốc
và artifact trên máy train là chuẩn. Chỉ có số tổng hợp — không văn bản báo cáo, không
ID bệnh nhân, không đường dẫn ảnh. Khoảng tin cậy là bootstrap ghép cặp theo study.

Bài báo gốc: D. Edirisinghe et al., "Chest X-Ray Report Generation Using Abnormality
Guided Vision Language Model", IEEE Access, vol. 13, 2025 (META-CXR).

## 0. Trạng thái tại thời điểm viết (2026-10-11, 16:31)

- **Stage 1:** giữ checkpoint `run_20261005_paper`. Không train lại.
- **Stage 2:** đang train toàn bộ, cấu hình P2, 2 epoch. Khởi chạy 15:50 ngày 11/10;
  lúc 16:31 đang dựng record Stage 1 cho tập train (70.132 / 222.758 study), chưa có
  lỗi, chưa có val loss. Dự kiến xong trưa đến chiều 13/10.
- **Chấm điểm:** script chờ `~/stage2_p2_score.sh` đã đặt trên máy train; tự chấm test
  sau khi log có `P2FULL_ALL_DONE` (mục 8.3).
- **Chưa có:** số test của mô hình mới. Mọi số Stage 2 mới trong file này là số pilot
  trên 500 study val.
- Mã: bản mới nhất `9b2963d`, nhánh `feat/stage2-finding-tokens`.

## 1. Tóm tắt

**Chẩn đoán**

1. Soft token của META-Former sụp thành **một** vector ngay ở pha 1a: cả 32 query cùng
   đọc token CLS toàn cục của PubMedCLIP, trạng thái trùng nhau và không tách lại được.
2. Soft token giữ thông tin bệnh phổ biến, mất bệnh khu trú/hiếm (Fracture ở mức ngẫu
   nhiên).
3. Mô hình Stage 2 cũ (v3) **bỏ qua cue**: đưa cue từ nhãn thật vào cũng không đổi báo cáo.
4. Báo cáo yếu chủ yếu vì thiếu bối cảnh: chỉ chép FINDINGS của lần chụp trước (không
   học gì) đã ngang bài báo ở BLEU-1/BLEU-4 và hơn mô hình 0,12 CE lexicon.
5. Luồng BioViL có hai lỗi kế thừa từ code gốc của tác giả: token bị trộn do `reshape`
   sai trục, và lớp chiếu 1408 chiều là ngẫu nhiên, đóng băng ở 1a và 1b.
6. Nhãn CheXpert tự động khá nhiễu so với nhãn bác sĩ (kappa 0,49–0,93 tùy bệnh).

**Stage 1**

- Số chính dùng quy tắc hai ngưỡng fit trên val: weighted F1 **0,820**, F1 5 bệnh
  **0,749** (bài báo 0,73 và 0,701). **Không phải giao thức argmax của bài báo.**
- Pha 1b rút còn 3 epoch, pha 1c bỏ khỏi mặc định — tăng epoch không giúp.
- Đã đo và không có lợi: RAD-DINO, sửa luồng BioViL (−0,020 AUROC trong pilot), thêm
  ảnh phụ, cờ EVOKE.

**Stage 2**

- Đã thêm: prompt có bối cảnh, cue lấy từ nhãn thật lúc train, giải mã beam, chấm CE
  bằng CheXbert.
- Pilot 500 study val: bối cảnh tăng BLEU-4 0,121 → 0,165 và CIDEr 0,284 → 0,487;
  cấu hình P2 đạt 0,178 và 0,518.
- Mô hình mới có thêm indication và báo cáo lần chụp trước, nên **không cùng điều kiện
  đầu vào với bài báo**.

## 2. Cách đo

| Thí nghiệm | Script | n | Artifact trên máy train |
|---|---|---|---|
| Sụp token theo phase | `scripts/diagnose_qformer_collapse.py` | 200 study val / checkpoint | `~/diag_qformer_20261011b/`, `...c/` |
| Probe soft token (cache của v3) | `scripts/probe_soft_tokens_cached.py` | 50.000 train / 2.780 test | `~/probe_softtok_20261011/` |
| Probe cùng giao thức (soft token, ảnh MedGemma, MHCAC) | `scripts/probe_image_features.py` | 53.300 fit / 5.922 giữ lại / 2.780 test | `~/stage2_analysis_20261011/probe_features.*` |
| Đặc trưng từng encoder (M1–M4) | `scripts/embed_cxr_encoders.py` | 53.300 fit / 2.780 test | `~/stage1_meas_20261011/` |
| Nhãn bác sĩ (M5) | `scripts/analyze_radiologist_labels.py` | 685 study test | `~/stage1_meas_20261011/` |
| Mốc truy hồi | `scripts/retrieval_report_baselines.py` | 60.000 train / 2.800 test | `~/stage2_analysis_20261011/retrieval*` |
| Embedding ảnh MedGemma | `scripts/embed_medgemma_images.py` | 62.800 ảnh | `~/stage2_analysis_20261011/emb_*` |
| Mốc chép lần chụp trước | `scripts/prior_report_baseline.py` | 2.508 / 2.800 test | `~/stage2_analysis_20261011/prior/` |
| Phân tích lỗi + CE lexicon | `scripts/analyze_stage2_generations.py` | 2.800 test | `~/stage2_analysis_20261011/analysis*.json` |
| Cue theo ngưỡng / cue oracle | probe trên máy train | 2.800 / 1.000 test | `~/stage2_v3_test_cutcue_20261010/`, `.../oracle2/` |
| 1b tốt nhất trên test | `pretraining.train` evaluate-only | 3.269 test | `~/eval_1b_test_20261011b/` |
| Pilot 1b sửa BioViL | `~/s1pilot.sh` | 60.000 train / 1.808 val | `~/s1pilot_20261011/` (đã xóa `.pth`) |
| Pilot Stage 2 P0/P1/P2 | `~/pilot_run.sh` | 10.000 train / 500 val | `~/pilot_20261011/` |
| Chấm Stage 2 (giao thức bài báo, ghép cặp) | `scripts/score_stage2_paper_protocol.py` | — | — |
| CE bằng CheXbert | `scripts/score_ce_chexbert.py` (venv `~/.venvs/chexbert`) | — | — |

**CE lexicon** = `safety.claims.LexiconClaimParser` gán nhãn cả báo cáo sinh lẫn tham
chiếu (14 bệnh, có cực tính); là thước đo **tương đối**, không phải CE CheXbert.

## 3. META-Former: soft token sụp

### 3.1 Sụp ở phase nào

| checkpoint | cosine giữa 32 token / study | số hướng hiệu dụng | entropy cross-attn tầng 3 | độ giống bản đồ attention giữa query |
|---|---:|---:|---:|---:|
| khởi tạo BLIP-2 | 0,445 | 2,94 | 0,61 | 0,27 |
| sau **1a** | **0,9989** | **1,002** | **0,11** | **1,000** |
| sau 1b | 0,9989 | 1,002 | 0,15 | 0,978 |
| sau 1c (v3 dùng) | 0,9997 | 1,001 | 0,09 | 0,906 |
| 1c run cũ 0925b | 0,9999 | 1,000 | 0,05 | 0,999 |

Đo độc lập trên cache Stage 2 của v3 (cùng checkpoint 1c): cosine trong một study
**+0,9995**, giữa các study +0,4112. Bộ đọc BLIP-2 chưa train đo ngày 2026-09-03 trên
`run_20260820_ft` là +0,80 — pha 1a làm nặng thêm chứ không cải thiện. Run cũ cũng sụp
(0,9999), tức đây là lỗi có tính hệ thống của thiết kế/huấn luyện, không phải của một run.

### 3.2 Cơ chế

Sau 1a, ở hai tầng cross-attention đầu, token nhận nhiều attention nhất là **chỉ số 196
ở 100% study — token CLS toàn cục của PubMedCLIP** (0,23 và 0,33 tổng attention). 32
query đọc cùng một token toàn cục nên trạng thái trùng nhau (cosine 0,915 sau tầng 1,
1,000 từ tầng 3). Các tầng sau dùng chung trọng số cho mọi query (self-attention giữa
các query giống nhau, FFN dùng chung) nên **không còn gì phá được đối xứng**. Mục tiêu
1a cũng không đòi query khác nhau: ITC lấy max theo query, LM dùng query làm prefix,
ITM lấy trung bình — và **ITM ở mức ngẫu nhiên suốt 1a** (val 0,644–0,670, ngẫu nhiên
0,6365).

### 3.3 Pha 1b và 1c

- **1b:** Q-Former đóng băng sau 10% đầu, trong khi StreamAdapter (không có LayerNorm
  phía sau ở PubMedCLIP và Swin) đẩy norm token 19,6 → 1.713 (PubMedCLIP) và 12 → 413
  (Swin); cosine giữa các token trong một ảnh 0,01 → 0,45 (BioViL), 0,71, 0,65 — token
  ảnh mất phần lớn tính không gian.
- **1c (1 epoch, batch 8):** ITC R@5 0,65 → **0,17**. Thay mọi token ảnh bằng token
  trung bình của ảnh, đầu ra vẫn cosine 0,81 với đầu ra thật — Q-Former gần như chỉ đọc
  "ảnh trung bình". Gradient căn chỉnh lớn gấp 7–35 lần gradient phân loại trên
  projector chung, cosine ≈ 0.

### 3.4 Thông tin bệnh còn lại trong soft token

Hai phép đo, khác giao thức nên số tuyệt đối lệch nhau; kết luận theo bệnh thì giống.

**(a) Probe trên cache của v3** (fit 50.000 study train, L2 chọn trên val, 2.780 test):

| | AUROC TB test | Neg | Pos | Unc |
|---|---:|---:|---:|---:|
| MHCAC (cùng 2.780 study) | 0,7454 | 0,7800 | 0,7847 | 0,6162 |
| probe, trung bình 32 token | 0,7127 | 0,7262 | 0,7464 | 0,6301 |
| probe, cả 32 token (24.576 chiều) | 0,6530 | 0,6779 | 0,6748 | 0,5715 |
| probe, trung bình, nhãn train xáo trộn | 0,5607 | | | |

- Trung bình 32 token giữ (0,713 − 0,5)/(0,745 − 0,5) = 87% phần AUROC trên mức ngẫu
  nhiên của MHCAC. Thiếu thông tin nhãn **không** phải lý do chính khiến báo cáo kém.
- Probe 32 token kém hơn trung bình vì token đã sụp: 24k chiều thêm vào chỉ là nhiễu.
- Mốc xáo trộn nhãn là 0,56 chứ không phải 0,50 (chọn mô hình tốt nhất trên val), nên
  chênh dưới ~0,05 giữa probe và MHCAC chỉ nên coi là xấp xỉ.

**(b) Probe cùng giao thức** (53.300 fit, 2.780 test):

| | AUROC TB | Pos | Neg | Unc |
|---|---:|---:|---:|---:|
| MHCAC (mốc) | **0,745** | 0,785 | 0,780 | 0,616 |
| soft token (trung bình 32) | 0,725 | 0,752 | 0,743 | 0,644 |
| ảnh MedGemma (trung bình 256) | 0,690 | 0,731 | 0,709 | 0,584 |

AUROC lớp Positive theo bệnh (MHCAC / soft token / ảnh MedGemma): Edema 0,843 / 0,846 /
0,822; Pleural Effusion 0,899 / 0,892 / 0,860; Support Devices 0,905 / 0,900 / 0,900;
Pneumothorax 0,865 / 0,831 / 0,782; Pleural Other 0,833 / 0,747 / 0,602; **Fracture
0,684 / 0,492 / 0,623**; Lung Lesion 0,748 / 0,718 / 0,704.

Kết luận chung: soft token ngang MHCAC ở bệnh phổ biến, mất ở bệnh khu trú/hiếm, và mất
mọi thứ một vector không chứa nổi (vị trí, mức độ, nhiều bệnh cùng lúc, chi tiết thiết
bị). Nút thắt là **sụp** (một token hiệu dụng), không phải thiếu tín hiệu 14 nhãn. Ảnh
MedGemma lấy trung bình thấp hơn soft token ở 12/14 bệnh.

## 4. Stage 1: độ dài và cấu hình các phase

| phase | quan sát | kết luận |
|---|---|---|
| 1a (4 epoch, batch 128) | gate R@5 0,61 / 0,63 / 0,66 / 0,67; val ITC loss thấp nhất ở epoch 1 rồi tăng (2,95 → 3,38); ITM = ngẫu nhiên | 2 epoch đủ; ITM không đóng góp; vấn đề chính là sụp token |
| 1b (5 epoch) | val AUROC 0,757 / 0,764 / **0,778** / 0,775 / 0,773; macro recall đỉnh ở epoch chỉ số 2 | 3 epoch đủ (tiết kiệm ~40% của ~7 giờ) |
| 1c (1 epoch, batch 8) | val AUROC 0,778 → 0,771; macro recall 0,492 → 0,486; R@5 0,65 → 0,17 | không giúp phân loại trên val, phá căn chỉnh |
| StreamAdapter | norm token tăng 90× / 34× trong 1b | thiếu chuẩn hóa sau adapter |

**1b tốt nhất so với 1c trên test** (3.269 study, argmax, bootstrap ghép cặp 1.000 lần):
AUROC trung bình 0,7473 so với 0,7436, chênh +0,0037 [−0,0012; +0,0095] — không có ý
nghĩa; macro recall +0,0050 [+0,0021; +0,0076] nghiêng về 1b; weighted F1 −0,0074
[−0,0091; −0,0056] và F1 5 bệnh −0,0144 [−0,0179; −0,0107] nghiêng về 1c. Khả năng phân
biệt như nhau; chênh F1 là dịch điểm làm việc của argmax, quy tắc hai ngưỡng fit trên
val hiệu chỉnh lại điểm này.

**Vì sao tăng epoch không giúp.** Sau 10% đầu của 1b chỉ còn MHCAC, StreamAdapter, view
fusion và đầu MPC được học; cả ba encoder đóng băng, `shared_visual_projector` và
`ln_vision` cũng đóng băng (nằm trong `fade_out`). Đường phân loại là một bộ đọc trên
đặc trưng cố định — gần với linear probe mở rộng — nên hội tụ trong 2–3 epoch rồi quá
khớp. Bằng chứng thêm: `run_20260930_3class` (loss chọn epoch 3, F1 chọn epoch 2); run
cũ kéo 10 → 15 epoch (cách chấm nhị phân cũ): test AUROC −0,0005 [−0,0043; +0,0031].
Ở 1a, R@5 chỉ tăng ~0,01/epoch, val ITC loss xấu đi từ epoch 1, và soft token đã sụp ở
cuối pha.

**Quyết định của user (2026-10-11):** 1b còn 3 epoch và chấm test; 1c không chạy mặc
định (`scripts/run_stage1_phases.sh` chạy `"phase1a phase1b"`, 1c bật bằng
`PHASES="phase1a phase1b phase1c"`). Lịch cosine 3 epoch là đường LR mới, chưa chạy.

## 5. Stage 1: nghiên cứu cải thiện

### 5.1 Số chính hiện tại

Quy tắc hai ngưỡng theo từng bệnh trên `p_pos/(p_pos+p_neg)`, fit trên val (1.808
study), vẫn ba lớp. File ngưỡng `configs/stage1_cutpoints/run_20261005_paper.json`.
Test 3.269 study:

| | weighted P / R / F1 | F1 5 bệnh | macro recall |
|---|---|---:|---:|
| hai ngưỡng (số chính) | 0,827 / 0,834 / **0,820** | **0,749** [0,741; 0,757] | 0,446 |
| argmax (giao thức bài báo) | 0,839 / 0,781 / 0,780 | 0,669 | 0,461 |
| bài báo | 0,87 / 0,78 / 0,73 | 0,701 | không nêu |

AUROC không đổi theo quy tắc quyết định (0,784 / 0,778 / 0,663). Phải ghi "không phải
giao thức argmax của bài báo" cạnh mọi số chính.

### 5.2 Nhiễu nhãn

685 study test có nhãn bác sĩ trong MIMIC-CXR-JPG 2.1.0. Mô hình vừa học vừa được chấm
bằng nhãn tự động, nên các bệnh nhãn kém có trần thấp bất kể kiến trúc.

| Bệnh | kappa | precision dương tính nhãn tự động | recall |
|---|---:|---:|---:|
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

Chấm cùng mô hình theo nhãn bác sĩ (676 study): AUROC lớp Positive cao hơn +0,022
[+0,008; +0,036], macro recall +0,049 [+0,022; +0,064] so với chấm bằng nhãn tự động.

### 5.3 Đòn bẩy trong tài liệu: cho encoder thích nghi

**Trong repo** (cách chấm nhị phân cũ, đã gỡ): `run_20260820_ft` mở khóa encoder nông
(ResNet50 `layer4` + projector, CLIP block 10–11 + `post_layernorm`, LR encoder 1e-5):
macro AUROC 0,7441 → 0,7643, +0,0201 [+0,0136; +0,0263], tăng 14/14 bệnh. Mở khóa sâu
hơn không thêm: +0,0049 [−0,0003; +0,0100]. Kappa đổi cùng lúc, nên quy cho việc mở
khóa là suy luận, chưa có ablation.

**Bằng chứng ngược:** pha 1c của `run_20261005_paper` cũng mở khóa nông mà không tăng
AUROC test. Ở 1c gradient căn chỉnh lớn gấp 7–35 lần gradient phân loại, ITC ở batch 8,
pha chỉ dài 1 epoch — encoder bị kéo bởi mục tiêu căn chỉnh chứ không phải phân loại.

Tài liệu: xem bảng 10.2.

### 5.4 Phương án đã đề xuất (KHÔNG thực hiện)

- **A:** 1b mở khóa encoder nông ngay từ đầu, tái dùng checkpoint 1a, không 1c (~5,7 giờ).
- **B (LP-FT):** giữ 1b đóng băng, thêm pha phân loại thuần 1–2 epoch mở khóa nông.

Cả hai bị thay thế sau khi đo M1–M5 phát hiện hai lỗi BioViL (5.6); pilot 5.7 gồm cả
biến thể mở khóa nông (arm C) và không arm nào thắng.

### 5.5 Kết quả đo M1–M5 (chỉ đọc checkpoint)

Probe tuyến tính ba lớp, 53.300 study fit / 2.780 study test, ảnh sạch.

| Nguồn đặc trưng | AUROC TB | Pos |
|---|---:|---:|
| MHCAC (dự đoán của mô hình, mốc) | 0,745 | 0,785 |
| MedCLIP-Swin, token pooled (pretrained) | **0,734** | 0,768 |
| RAD-DINO [CLS; trung bình patch] | 0,721 | 0,764 |
| PubMedCLIP CLS | 0,638 | 0,661 |
| BioViL sau `run_20260820_ft` (10 epoch mở khóa) | 0,726 | 0,764 |
| BioViL sau 1c (1 epoch mở khóa) | 0,707 | 0,739 |
| BioViL "pretrained" (projector 1408 chưa train) | 0,490 | không đọc được |
| soft token sau 1a / 1b / 1c | 0,721 / **0,689** / 0,725 | 0,749 / 0,718 / 0,753 |

- **M1** không đọc được theo cách đã định (không có mốc BioViL pretrained có nghĩa);
  nhưng BioViL sau khi projector và `layer4` được train đạt 0,707–0,726 — thông tin có ở đó.
- **M2 — RAD-DINO: không tích hợp.** Một mình nó thấp hơn MedCLIP-Swin (0,721 so với 0,734).
- **M3 — pha 1b làm soft token kém đi 0,031** (0,721 → 0,689); 1c kéo lại (0,725).
- **MHCAC chỉ hơn probe tuyến tính trên MỘT token của MedCLIP-Swin 0,012 AUROC** — hai
  luồng còn lại gần như không đóng góp.
- **M5:** xem 5.2.
- Đặc trưng PubMedCLIP CLS giống hệt nhau giữa pretrained, 1c và `run_20260820_ft`
  (0,6378 cả ba) — chưa xác minh được việc mở khóa block 10–11 có đổi trọng số hay không.

### 5.6 Hai lỗi ở luồng BioViL (có từ code gốc của tác giả, mặc định chưa sửa)

1. **Token bị trộn.** BioViL trả `[B, 1408, 14, 14]`; `blip2_qformer.py` (hai chỗ) và
   `precompute_features.py` gọi `.reshape(B, -1, 1408)` không hoán vị trục. 196 "token"
   vì thế là các nhóm ~7 bản đồ kênh duỗi phẳng, không phải 196 ô ảnh; một ô ảnh thật bị
   rải ra cả 196 token. Mã hóa vị trí, attention theo vùng và bản đồ giải thích trên
   luồng này đều vô nghĩa. Code gốc `DasithEdirisinghe/META-CXR` có đúng dòng này.
2. **Projector 1408 chiều là khởi tạo ngẫu nhiên và đóng băng trong 1a và 1b.**
   `ImageModel(joint_feature_size=1408)` bỏ projector 128 chiều của BioViL-T khi nạp
   trọng số; projector mới chỉ nằm trong danh sách train của pha 1c.

Hệ quả: phần lớn tín hiệu phân loại hiện đến từ MedCLIP-Swin; BioViL-T — encoder có độ
phân giải không gian cao nhất — gần như chưa được dùng đúng.

Đã thêm cờ tùy chọn `model.biovil.token_layout: patch` (mã `ef964ab`); vắng cờ =
`legacy_reshape`, giống hệt trước. YAML đang ship không bật.

### 5.7 Pilot sửa luồng BioViL — không có lợi

Chỉ pha 1b, 60.000 study, 2 epoch, cùng checkpoint 1a, val 1.808 study, mã `ef964ab`.
13:10 → 15:33, ba arm `rc=0`, mỗi arm ~48 phút, VRAM đỉnh 6,4 / 6,4 / 6,8 GB.

| arm | khác gì | epoch 0 (AUROC / macro recall / F1-5) | epoch 1 | `loss_cls` train cuối |
|---|---|---|---|---:|
| A | như cấu hình đang ship | 0,702 / 0,476 / 0,689 | **0,740** / 0,483 / 0,714 | 0,570 |
| B | bố cục patch; projector BioViL được train; `shared_visual_projector` + `ln_vision` không đóng băng | 0,688 / 0,469 / 0,701 | 0,721 / 0,477 / 0,706 | 0,566 |
| C | B + mở khóa encoder nông | 0,704 / 0,485 / 0,716 | 0,726 / 0,482 / 0,727 | 0,560 |

| | AUROC TB | AUROC Pos | macro recall | F1-5 |
|---|---|---|---|---|
| B − A | **−0,0197 [−0,0288; −0,0103]** | −0,0238 [−0,0359; −0,0132] | −0,0052 [−0,0093; −0,0014] | −0,0077 [−0,0126; −0,0027] |
| C − A | **−0,0358 [−0,0470; −0,0247]** | −0,0366 [−0,0500; −0,0219] | +0,0025 [−0,0032; +0,0081] | +0,0017 [−0,0040; +0,0080] |

**Theo quy tắc chốt trước: không arm nào thắng — giữ `run_20261005_paper`, không train
lại Stage 1.** B và C có loss train thấp hơn nhưng AUROC val thấp hơn A có ý nghĩa.

Pilot này **không** chứng minh bố cục patch tệ hơn về nguyên tắc: B và C khởi từ
checkpoint 1a đã khớp với bố cục cũ, chỉ 2 epoch trên 60.000 study, và cả ba arm còn
đang tăng ở epoch 1. Nó chỉ cho thấy trong ngân sách kiểm chứng rẻ, sửa hai lỗi không
đem lại lợi ích đo được, nên không đủ căn cứ để chi 15–20 giờ train lại 1a + 1b.

### 5.8 Phát hiện phụ khi đọc code (chưa sửa)

- Augmentation train chỉ đi vào luồng BioViL; PubMedCLIP và Swin có tiền xử lý riêng,
  không augmentation.
- Record **train** của Stage 2 được tính từ ảnh đã augmentation ngẫu nhiên; val/test từ
  ảnh sạch.

## 6. Vì sao báo cáo Stage 2 (v3) yếu

### 6.1 Mốc so sánh (test 2.800 study, giao thức bài báo)

| | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore | CE lexicon | F1-5 lexicon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bài báo META-CXR | 0,390 | 0,102 | 0,173 | 0,280 | 0,291 | 0,426 | không có | không có |
| **v3 beam (sinh)** | 0,357 | 0,098 | 0,137 | 0,265 | 0,128 | 0,369 | 0,239 | 0,379 |
| v3 greedy | 0,285 | 0,073 | 0,120 | 0,250 | 0,084 | 0,361 | 0,230 | 0,346 |
| một báo cáo cố định | 0,256 | 0,061 | 0,101 | 0,238 | 0,062 | 0,340 | 0,060 | 0,128 |
| báo cáo train ngẫu nhiên | 0,281 | 0,046 | 0,103 | 0,202 | 0,035 | 0,290 | 0,164 | 0,235 |
| chép theo soft token | 0,330 | 0,062 | 0,125 | 0,220 | 0,066 | 0,315 | 0,259 | 0,362 |
| chép theo ảnh MedGemma | 0,338 | 0,068 | 0,130 | 0,226 | 0,077 | 0,328 | 0,240 | 0,336 |
| chép theo nhãn MHCAC | 0,334 | 0,057 | 0,123 | 0,213 | 0,052 | 0,304 | 0,294 | 0,455 |
| chép theo nhãn THẬT | 0,333 | 0,072 | 0,128 | 0,230 | 0,085 | 0,334 | 0,416 | 0,559 |
| **chép lần trước, không có thì v3 beam** | **0,391** | **0,106** | 0,155 | 0,259 | 0,169 | 0,365 | — | — |

Chỉ trên 2.508 study có lần trước: chép lần trước so với v3 beam — CE lexicon **0,356 so
với 0,236** (F1-5 0,424 / 0,382), CIDEr +0,0459 [+0,0210; +0,0699], ROUGE-L −0,0067
[−0,0116; −0,0017], BERTScore −0,0049 [−0,0107; +0,0009].

### 6.2 LLM không dùng cue

| thay cue (giữ nguyên mọi thứ khác) | n | BERTScore | ROUGE-L | CIDEr | CE lexicon |
|---|---:|---|---|---|---|
| Eq. 22 → ngưỡng hai mức (beam) | 2.800 | +0,0010 [−0,0015; +0,0035] | +0,0004 | +0,0040 [−0,0073; +0,0157] | 0,239 → 0,215 |
| Eq. 22 → **nhãn THẬT** (greedy) | 1.000 | −0,0024 [−0,0079; +0,0032] | −0,0021 [−0,0059; +0,0018] | +0,0050 [−0,0104; +0,0196] | 0,269 → 0,249 |
| hoán đổi cue giữa các study (v3, val) | 300 | −0,014 | | | |
| hoán đổi soft token giữa các study (v3, val) | 300 | −0,072 | | | |

Cue oracle đổi ở 977/1.000 study mà 153 báo cáo vẫn giống hệt từng chữ và không chỉ số
nào thay đổi có ý nghĩa: **v3 bỏ qua kênh cue**, nên cải thiện phân loại Stage 1 không
thể cải thiện báo cáo của nó. Nguyên nhân khả dĩ: lúc train, cue Eq. 22 nhiễu (trung
bình 4,37 bệnh "dương tính" mỗi study) nên mô hình học cách không tin; trong khi soft
token đã mang phần lớn thông tin bệnh phổ biến.

### 6.3 Viết thiếu, viết theo khuôn

- 37 từ (greedy) / 47 từ (beam) so với 56 của bác sĩ; BERTScore precision 0,424, recall
  0,299; chỉ nhắc 32–34% số bệnh mà bác sĩ có bàn tới (oracle truy hồi 43,5%).
- Không bao giờ nhắc Fracture, Lung Lesion, Pleural Other, Enlarged Cardiomediastinum
  (mention recall 0,00); rất ít Pneumonia (0,17), Lung Opacity (0,24); luôn viết câu âm
  tính quen thuộc (pneumothorax 0,95, effusion 0,87).
- 1.473 báo cáo khác nhau trên 2.800 (greedy); báo cáo phổ biến nhất chiếm 7,5–11,4%.

### 6.4 Thiếu bối cảnh lâm sàng

- 90,0% FINDINGS test có ngôn ngữ so sánh thời gian; mô hình tự viết câu so sánh trong
  51% báo cáo mà bản gốc không so sánh.
- Lần chụp trước có FINDINGS: test 89,6%, val 65,4%, train 64,1%; khoảng cách trung vị
  7–9 ngày, khoảng một nửa ≤ 7 ngày. Tập test MIMIC nghiêng về bệnh nhân chụp nhiều lần.
- INDICATION/HISTORY: 91,7% study test (trung vị 8 từ); COMPARISON 67,4%; TECHNIQUE
  24,6%. Lateral: 38,2% study test.

### 6.5 Metric NLG thưởng văn phong

Một báo cáo cố định có ROUGE-L 0,238 và BERTScore 0,340 — cao hơn chép một báo cáo có
nhãn đúng hoàn toàn (0,230; 0,334). Tài liệu cùng chiều: PromptMRG, RaDialog, ReRef
(bảng 10.1).

### 6.6 Stage 2 train chưa tới

v3: 1 epoch, 170.370 mẫu, LoRA r8, LR 1e-4. Loss train theo thập phân vị 1,301 → 1,016,
vẫn giảm ở cuối (1,028 → 1,017 → 1,016); val 1,061 ≈ train 1,080 — thiếu train chứ không
quá khớp. RaDialog: 4 epoch, LR 3e-4. Ảnh hưởng tới chất lượng sinh chưa đo được (loss
có teacher forcing không phản ánh trực tiếp lúc sinh).

### 6.7 Xếp hạng nguyên nhân theo bằng chứng

1. Thiếu bối cảnh lần chụp trước (bằng chứng tại chỗ mạnh nhất, 6.1).
2. LLM bỏ qua cue (đo trực tiếp bằng oracle, 6.2).
3. Metric thưởng văn phong — giới hạn của thước đo, không phải lỗi mô hình (6.5).
4. Kênh ảnh chỉ còn một vector (mục 3; hoán đổi soft token −0,072 BERTScore).
5. Thiếu indication (tài liệu mạnh; đã gộp vào pilot cùng lần chụp trước, chưa tách riêng).
6. Train Stage 2 ngắn (bằng chứng yếu).

## 7. Pilot Stage 2 (P0 / P1 / P2)

### 7.1 Vì sao pilot này

- Đòn bẩy NLG lớn nhất trong tài liệu là bối cảnh đầu vào (RaDialog, MAIRA-2), và dữ
  liệu có sẵn (6.4).
- Cue oracle không đổi báo cáo: muốn phân loại tốt hơn có ích thì phải train để mô hình
  dựa vào cue.
- Kênh ảnh gốc MedGemma không được probe/truy hồi tại chỗ ủng hộ → chỉ là arm tùy chọn
  (P3), không chạy.
- Sửa Q-Former đòi train lại 1a với cache ~148 GB trong khi `/home` khi đó còn 59 GB; và
  RaDialog-align (Q-Former 32 query hoạt động bình thường) chỉ đạt BLEU-4 9,5 / ROUGE-L
  27,1, ngang mức v3. Vì vậy không nằm trong pilot.

### 7.2 Thiết kế

Cùng seed, cùng 10.000 study train (7.604 record có FINDINGS hợp lệ), 1 epoch, LoRA
r8/α16. Chấm 500 study val cố định (cùng `sample_key`, cùng thứ tự), beam 4 + length
penalty 2 và greedy; bootstrap ghép cặp 2.000 lần, seed 16.

| arm | kênh ảnh | cue MHCAC | bối cảnh |
|---|---|---|---|
| P0 | 32 soft token (như v3) | Eq. 22 | không |
| P1 | 32 soft token | Eq. 22 | indication + FINDINGS lần trước |
| P2 | 32 soft token | train: nhãn thật, 20% giữ cue MHCAC; suy luận: MHCAC ngưỡng hai mức | indication + FINDINGS lần trước |

**Quy tắc chốt trước:** (1) nhận bối cảnh nếu P1 − P0 có ROUGE-L **và** BERTScore với
cận dưới > 0; (2) nhận cách train cue nếu P2 − P1 có CE lexicon ≥ +0,02 với cận dưới > 0
và không chỉ số NLG nào giảm có ý nghĩa; (3) không arm nào thỏa: giữ v3, chỉ thêm beam.

### 7.3 Code đã viết cho pilot

- `--prompt-style paper_context`: prompt bài báo + dòng `Indication:` + dòng FINDINGS
  lần chụp trước (kèm số ngày), lấy từ `training/dataio/study_context.py` — lần chụp
  **trước hẳn** gần nhất của cùng bệnh nhân. Gắn sau khi dựng record Stage 1 nên không
  đụng cache.
- `--train-cues oracle_mix --train-cue-model-share 0.2`: cue của record train lấy từ lớp
  CheXpert thật, trừ 20% cố định giữ cue MHCAC (`training/dataio/oracle_cues.py`);
  val/test luôn dùng cue MHCAC.
- Beam trong `generate_stage2_reports.py` (`--num-beams 4 --length-penalty 2.0`); soft
  token được lặp theo beam. Mặc định vẫn greedy.
- Chấm điểm đưa vào git: `scripts/score_stage2_paper_protocol.py`,
  `scripts/score_ce_chexbert.py`.

### 7.4 Kết quả (01:17 → 07:15, `PILOT_ALL_DONE`)

| beam 4, lp 2 | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore | CE lexicon | CheXbert micro F1-14 | micro F1-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 (như v3) | 0,370 | 0,121 | 0,142 | 0,301 | 0,284 | 0,410 | 0,212 | 0,438 | 0,438 |
| P1 (+ bối cảnh) | 0,408 | 0,165 | 0,168 | 0,336 | 0,487 | 0,437 | 0,289 | 0,441 | 0,441 |
| P2 (+ cue nhãn thật lúc train) | **0,423** | **0,178** | **0,172** | **0,345** | **0,518** | **0,448** | **0,308** | **0,463** | **0,494** |

| greedy | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore | CE lexicon | CheXbert micro F1-14 | micro F1-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0,296 | 0,094 | 0,127 | 0,288 | 0,193 | 0,410 | 0,213 | 0,452 | 0,469 |
| P1 | 0,349 | 0,146 | 0,156 | 0,338 | 0,407 | 0,455 | 0,287 | 0,457 | 0,453 |
| P2 | 0,345 | 0,144 | 0,156 | 0,337 | 0,387 | 0,460 | 0,290 | 0,470 | 0,480 |

Ghép cặp (n = 500):

| | BERTScore | ROUGE-L | CIDEr | BLEU-4 | CE lexicon | CheXbert micro F1-5 |
|---|---|---|---|---|---|---|
| P1 − P0, beam | +0,0275 [+0,0143; +0,0406] | +0,0359 [+0,0229; +0,0490] | +0,2027 [+0,1118; +0,3002] | +0,0386 [+0,0249; +0,0535] | +0,0763 [+0,0370; +0,1131] | +0,0022 [−0,0525; +0,0576] |
| P1 − P0, greedy | +0,0450 [+0,0319; +0,0574] | +0,0496 [+0,0372; +0,0617] | +0,2132 [+0,1338; +0,2965] | +0,0537 [+0,0410; +0,0660] | +0,0744 [+0,0318; +0,1122] | −0,0157 [−0,0653; +0,0379] |
| P2 − P1, beam | +0,0110 [+0,0010; +0,0209] | +0,0081 [−0,0000; +0,0164] | +0,0312 [−0,0145; +0,0740] | +0,0103 [+0,0026; +0,0180] | +0,0190 [−0,0075; +0,0453] | +0,0532 [+0,0158; +0,0889] |
| P2 − P1, greedy | +0,0050 [−0,0025; +0,0124] | −0,0004 [−0,0071; +0,0064] | −0,0195 [−0,0624; +0,0219] | −0,0040 [−0,0108; +0,0028] | +0,0029 [−0,0403; +0,0499] | +0,0266 [−0,0095; +0,0655] |

### 7.5 Đọc theo quy tắc

1. **Bối cảnh — NHẬN.** P1 − P0 có ROUGE-L và BERTScore với cận dưới > 0 ở cả beam và
   greedy; BLEU-4 +0,04–0,05, CIDEr +0,20, CE lexicon +0,075. CheXbert micro không đổi;
   macro F1-14 tăng (+0,047 beam, +0,068 [+0,022; +0,110] greedy).
2. **Train cue bằng nhãn thật — KHÔNG đạt ngưỡng đã chốt.** CE lexicon beam +0,019
   [−0,008; +0,045], greedy +0,003. Ngoài quy tắc: CheXbert micro F1-5 beam +0,053
   [+0,016; +0,089] có ý nghĩa, BERTScore và BLEU-4 beam tăng có ý nghĩa, không chỉ số
   NLG nào giảm có ý nghĩa. Có tín hiệu tốt nhưng chưa xác lập ở n = 500, một seed.

**User chọn cấu hình P2** cho lần train toàn bộ, biết rằng điều kiện (2) chưa đạt.

P0 của pilot (BLEU-4 0,121 trên 500 study val) không so trực tiếp được với v3 trên
2.800 study test; chỉ các hiệu ghép cặp trong pilot là có giá trị.

## 8. Train Stage 2 toàn bộ (đang chạy)

### 8.1 Cấu hình

User duyệt 2026-10-11: P2, 2 epoch. Stage 1 giữ `run_20261005_paper` (pha 1c tốt nhất).
`~/stage2_p2_full.sh` → `~/stage2_p2_20261011/`, log `~/stage2_p2_20261011.log`, checkout
`~/Meta-CXR-v3` (manifest `full_allviews_v3`), mã `1e0e181` (cùng mã với pilot).

```
run_medgemma_qlora.py --pipeline-mode meta_cxr_qformer_with_mhcac_prompt --section-mode findings_only
  --prompt-style paper_context --cue-rule cutpoints --threshold-path configs/stage1_cutpoints/run_20261005_paper.json
  --train-cues oracle_mix --train-cue-model-share 0.2 --train-epochs 2 --batch-size 2 --grad-accum 8
  --num-workers 4 --save-every-updates 32 --no-upload --skip-test
```

Sau khi train: sinh trên test với beam 4 + length penalty 2 và greedy, `--max-new-tokens
256` (`gen_test_beam`, `gen_test_greedy`); kết thúc bằng `P2FULL_ALL_DONE`. Ước tính:
dựng record Stage 1 ~1,8 giờ, train ~40 giờ, sinh ~3 giờ. Để có chỗ, đã xóa các `.pth`
của pilot Stage 1 (`~/s1pilot_20261011`, 21 GB); `.npz` và log còn giữ.

### 8.2 Rà soát trước khi để run chạy 2 ngày (16:15)

- **Độ dài chuỗi:** trên 7.604 record train và 1.360 record val của pilot P2 (cùng
  prompt): prompt p50 / p99 / max = 375 / 480 / 547 token, cả chuỗi max 676 — dưới
  `--max-length 768`; 0 dòng bị cắt, 0 token target bị mất.
- **Chọn checkpoint:** val loss chấm cuối mỗi epoch, adapter tốt nhất lưu riêng,
  `--patience 1`; LR cosine trên toàn bộ số update, warm-up 3%.
- **Giữ nguyên như pilot (chưa có phép đo ủng hộ thay đổi):** LoRA r8/α16, LR 1e-4, batch
  2 × 8; record train tính từ ảnh có augmentation; soft token đã sụp.
- Kết luận: không có tối ưu nào đã được kiểm chứng mà run này còn thiếu; không dừng run.

### 8.3 Kiểm tra lúc 16:31 và việc còn lại

- Tiến trình còn sống, 41 phút; dựng record train 70.132 / 222.758 (~25–35 study/giây).
- Log không có Traceback, OOM, Xid; ổ dữ liệu `ntfs3`; máy không reboot từ 04/10;
  `/home` còn 44 GB.
- Đã đặt `~/stage2_p2_score.sh` (log `~/stage2_p2_score.log`): chờ `P2FULL_ALL_DONE`,
  rồi chấm `gen_test_beam` và `gen_test_greedy` bằng hai script chấm, ghép cặp với v3
  beam (`~/stage2_v3_test_beam_20261009/gen/beam4_lp2.jsonl`, 2.800 study) trên các
  study chung; ghi vào `~/stage2_p2_20261011/scores/`, kết thúc bằng `P2SCORE_ALL_DONE`.
  Thấy `ABORT` thì không chấm gì. Script này không train và không đổi pipeline.

Còn phải làm khi run xong:

1. Đọc val loss epoch 1 và epoch 2 — epoch thứ hai có giúp không.
2. Đọc điểm test, so ghép cặp với v3 và so với bài báo. Bản greedy mới được ghép với v3
   **beam** (không cùng cách giải mã) — ghi rõ khi báo số.
3. Ghi rõ mô hình mới có indication và báo cáo lần chụp trước, không cùng điều kiện với
   bài báo.
4. Cập nhật `TONG_HOP_2026-10-11.md`, `CHI_MUC_TINH_MOI.md`, `README.md`, `CLAUDE.md`
   và file này, rồi commit.

Nếu run lỗi: báo nguyên nhân và hỏi user trước khi chạy lại.

## 9. So với bài báo: chỉ mục tính mới

"Tính mới" là **so với bài báo META-CXR**, không phải so với toàn bộ tài liệu; kỹ thuật
lấy từ công trình khác được ghi nguồn. Chi tiết "bài báo" đã đối chiếu với PDF (mục
III–V, Eq. 22, Bảng 2–7, Hình 10–11).

### 9.1 Cách phân loại

| Loại | Ý nghĩa | Được viết trong luận văn là |
|---|---|---|
| A | Khác bài báo, có số đo cho thấy tốt hơn, đã đưa vào pipeline | đóng góp có bằng chứng |
| B | Khác bài báo, đã đưa vào pipeline, chưa tách riêng đóng góp | bổ sung/khác biệt, chưa kết luận tốt hơn |
| C | Phát hiện mới từ phân tích, không phải thay đổi mô hình | đóng góp phân tích |
| D | Đã thử, không cải thiện | kết quả âm, báo cáo trung thực |
| E | Đề xuất | hướng phát triển |
| K | Khác bài báo nhưng không phải tính mới | ghi ở phần cài đặt/giới hạn |

### 9.2 Bảng chỉ mục

Cột "Trạng thái 11/10" là cập nhật của file này; `CHI_MUC_TINH_MOI.md` còn ghi một số
mục theo trạng thái trước pilot (xem 9.9).

| ID | Hạng mục | Loại | Trạng thái 11/10 |
|---|---|---|---|
| TM-01 | Số chính Stage 1 dùng hai ngưỡng theo bệnh fit trên val | A | đang dùng |
| TM-02 | Giải mã beam 4 + length penalty 2 | A | đã có trong `generate_stage2_reports.py` |
| TM-03 | Làm sạch target FINDINGS v3 | A | đang dùng |
| TM-04 | GradCache + SigLIP cho pha 1a | A | đang dùng |
| TM-05 | Đa góc chụp: ViewFusion, MPC, StreamAdapter, view consistency | B | đang dùng |
| TM-06 | Luồng PubMedCLIP: post_layernorm, trừ trung bình, tiền xử lý riêng | B | đang dùng |
| TM-07 | Mở khóa một phần encoder ở pha 1c | B | 1c không còn chạy mặc định |
| TM-08 | Chọn checkpoint theo macro recall; loại study không có nhãn | B | đang dùng |
| TM-09 | Giải thích Stage 2 (attention rollout + hai cổng kiểm định) | B | đo trên adapter đã xóa |
| TM-10 | Bootstrap ghép cặp, mốc so sánh, nhóm con theo số ảnh | B | đang dùng |
| TM-11 | Soft token sụp thành một vector ở pha 1a | C | mục 3 |
| TM-12 | 1b làm lệch đầu vào Q-Former; 1c phá căn chỉnh | C | mục 3.3, 4 |
| TM-13 | Soft token giữ bệnh phổ biến, mất bệnh khu trú/hiếm | C | mục 3.4 |
| TM-14 | Metric NLG thưởng văn phong; LLM không dùng thông tin phân loại | C | mục 6.2, 6.5 |
| TM-15 | Thiếu bối cảnh lâm sàng | C | mục 6.1, 6.4 |
| TM-16 | Ngưỡng Eq. 22 không tái lập được và làm giảm F1 | C | — |
| TM-17 | Cờ EVOKE | D | không dùng |
| TM-18 | Tối đa 3 ảnh phụ | D | không dùng |
| TM-19 | Cue Stage 2 theo ngưỡng hai mức (trên adapter v3) | D | xem ghi chú 9.6 |
| TM-20 | Finding token học được | D | không dùng |
| TM-21 | Logit-adjusted loss | D | không dùng |
| TM-22 | ITC ở batch nhỏ; khung nhị phân/mention gate | D | đã gỡ |
| TM-23 | Indication + FINDINGS lần chụp trước trong prompt | E | **đã làm**; pilot dương tính; số test chờ run |
| TM-24 | Kênh ảnh gốc MedGemma thay 32 soft token | E | chưa làm, không ưu tiên |
| TM-25 | Sửa sụp META-Former; bỏ 1c; 1b 3 epoch | E | **một phần**: 1b 3 epoch và 1c tùy chọn đã làm; sửa sụp chưa |
| TM-26 | Đo CE bằng CheXbert | E | **đã làm** (`score_ce_chexbert.py`) |
| TM-27 | MedGemma 1.5 4B QLoRA thay Vicuna-7B | K | — |
| TM-28 | Sửa stop token Stage 2 | K | — |
| TM-29 | Độ dài pha theo epoch; gradient checkpointing; bf16 | K | — |

Phát sinh trong đợt này, chưa có ID trong chỉ mục: hai lỗi BioViL kế thừa từ code gốc
(5.6), nhiễu nhãn so với nhãn bác sĩ (5.2), pilot sửa BioViL và RAD-DINO âm tính (5.5,
5.7), cue nhãn thật lúc train (7).

### 9.3 A — Có bằng chứng tốt hơn bài báo

**TM-01 — Quy tắc hai ngưỡng theo bệnh.** Bài báo báo số phân loại bằng argmax (Hình 10,
Bảng 5/7), chỉ dùng ngưỡng Eq. 22 cho prompt, không nêu fit trên tập nào. Số: mục 5.1.
So với argmax: F1 5 bệnh +0,0805 [+0,0743; +0,0866], macro recall −0,0149 (1.000 lần,
seed 16). Nguồn: `handoff/PLAN-2026-10-10-cutpoint-headline.md`, commit `be1acc0`.

**TM-02 — Beam 4, length penalty 2.** Bài báo không nêu cách giải mã. Test 2.800 study,
cùng adapter v3: beam − greedy BERTScore +0,0083 [+0,0037; +0,0129], ROUGE-L +0,0150
[+0,0112; +0,0188], CIDEr +0,0439 [+0,0230; +0,0653]. Chọn trên val 300 trước, chạy test
một lần. Nguồn: `handoff/PLAN-2026-10-07-stage2-header-fix.md`.

**TM-03 — Làm sạch target FINDINGS (manifest v3).** 9,8% target train bắt đầu bằng dòng
tiêu đề phiếu chụp và 3,4% chỉ có dòng đó; mô hình học lại và 57% báo cáo test chỉ là
dòng tiêu đề (BLEU-1 0,056). Parser v3 bỏ dòng này: 0 báo cáo chỉ-tiêu-đề, greedy BLEU-1
0,285. Hai run khác cohort test (2.984 và 2.800 study).

**TM-04 — GradCache + SigLIP cho pha 1a.** Bài báo dùng ITC/ITM/ITG của BLIP-2, không
nêu batch. Trên GPU 16 GB, ITC ở batch 8 cho kết quả ngẫu nhiên cả bốn lần đo. Với
GradCache batch 128 + loss sigmoid + cache đặc trưng: lần đầu ITC vượt ngẫu nhiên — R@5
0,664 / 0,641, `delta_nats` +2,93 (val 256 cặp). Lưu ý TM-11: căn chỉnh vượt ngẫu nhiên
nhưng token vẫn sụp. Nguồn: D-021.

### 9.4 B — Khác/bổ sung, chưa tách riêng đóng góp

**TM-05 — Đa góc chụp.** Bài báo: "focused exclusively on frontal views". Dự án: một ảnh
neo (PA > AP > lateral) + tối đa 1 ảnh phụ, `ViewFusionModule`, MPC và `StreamAdapter`
(ý tưởng từ EVOKE), `view_consistency_loss` có lề và cổng độ tự tin. Chưa có ablation đa
góc chụp so với chỉ frontal; các biến thể đã thử đều không tốt hơn (TM-17, TM-18).

**TM-06 — Luồng PubMedCLIP.** Đọc patch qua `post_layernorm` rồi trừ trung bình theo ảnh
(cosine trung bình giữa patch 0,674 → −0,014), giữ CLS; tiền xử lý CLIP riêng (D-022).
196 + 50 + 50 = 296 token, khớp số token bài báo mô tả.

**TM-07 — Mở khóa encoder ở pha 1c.** Bài báo chỉ mở "projection heads of the visual
encoders". Dự án mở thêm ResNet50 `layer4` và khối 10–11 + `post_layernorm` của CLIP.
Bằng chứng duy nhất (+0,020 AUROC) đo trong khung nhị phân đã gỡ (D-023).

**TM-08 — Chọn checkpoint và lọc nhãn.** Chọn theo macro recall ba lớp trên val (D-024).
Ô trống = âm tính như bài báo (D-018), nhưng study không có thông tin CheXpert nào bị
loại thay vì biến thành 14 ô âm tính.

**TM-09 — Giải thích Stage 2.** Attention rollout có trọng số gradient trên MedGemma,
hai cổng kiểm định (ảnh lệch study: +0,2160 [+0,1860; +0,2468] NLL, n = 100; ngẫu nhiên
hóa trọng số: rho 0,986 → 0,295), bộ gán nhãn câu `lexicon_v2`. Đo trên adapter đã xóa
2026-10-05.

**TM-10 — Độ chặt của đánh giá.** Bootstrap ghép cặp theo study cho mọi so sánh, mốc so
sánh, nhóm con theo số ảnh, kiểm tra không fit ngưỡng trên test.

### 9.5 C — Phát hiện mới từ phân tích

TM-11 → mục 3.1–3.2; TM-12 → mục 3.3 và 4; TM-13 → mục 3.4; TM-14 → mục 6.2, 6.5 (thêm:
chép báo cáo train có nhãn khớp dự đoán MHCAC đạt CE lexicon 0,294, cao hơn báo cáo LLM
sinh 0,239); TM-15 → mục 6.1, 6.4.

**TM-16 — Ngưỡng Eq. 22.** Bài báo không nêu ngưỡng fit trên tập nào (đường ROC duy
nhất, Hình 5, vẽ trên test) và không công bố file ngưỡng. Eq. 22 fit trên val của dự án,
chấm test: weighted F1 0,702, F1 5 bệnh 0,637 — thấp hơn argmax (0,780; 0,669).

### 9.6 D — Đã thử, không cải thiện

| ID | Thử nghiệm | Kết quả (test, bootstrap ghép cặp) | Nguồn |
|---|---|---|---|
| TM-17 | Cắt gradient K/V ảnh phụ + τ MPC 0,5 (như EVOKE) | F1 5 bệnh −0,0160 [−0,0193; −0,0122]; weighted F1 −0,0033 | `PLAN-2026-10-09-evoke-flags.md` |
| TM-18 | Tối đa 3 ảnh phụ | AUROC −0,0085 [−0,0132; −0,0028]; F1 5 bệnh −0,0114 | `PLAN-2026-10-06-multi-aux-views.md` |
| TM-19 | Cue Stage 2 theo ngưỡng hai mức | beam: mọi chỉ số ngang; greedy ROUGE-L −0,0030 [−0,0051; −0,0009] | `PLAN-2026-10-10-cutpoint-headline.md` |
| TM-20 | 13 finding token học được | không cải thiện; BERTScore −0,0075 so với không cue | `PLAN-2026-09-10-mention-finding-tokens.md` |
| TM-21 | Logit-adjusted loss (τ 1,0 / 0,5) | mô phỏng: τ 1,0 sụp về Uncertain; không áp dụng | `PLAN-2026-10-01-logit-adjusted-loss.md` |
| TM-22 | ITC batch 8 (4 lần); khung nhị phân + mention gate | ITC = ngẫu nhiên; khung nhị phân đã gỡ (D-023) | `PLAN-2026-08-19-itc-temp-probe.md` |
| — | RAD-DINO (probe) | 0,721 so với MedCLIP-Swin 0,734 | mục 5.5 |
| — | Sửa luồng BioViL (pilot 1b) | AUROC −0,0197 [−0,0288; −0,0103] | mục 5.7 |

Ghi chú TM-19: kết quả âm đó đo trên adapter v3, vốn train với cue Eq. 22 và bỏ qua cue.
Lần train P2 dùng cue ngưỡng hai mức lúc suy luận với một adapter được train để đọc cue
— là phép thử công bằng mà TM-19 chưa có; kết quả chờ run.

### 9.7 E — Đề xuất

**TM-23 — Bối cảnh trong prompt Stage 2.** Đã thực hiện (mục 7); pilot: ROUGE-L +0,036,
BERTScore +0,028, BLEU-4 +0,039, CIDEr +0,20 (beam, 500 val). Số test chờ run.

**TM-24 — Kênh ảnh gốc của MedGemma** (256 token) thay 32 soft token. Bằng chứng yếu:
tài liệu ủng hộ (MedGemma 4B RadGraph F1 29,5; MAIRA-2/LLaVA-Rad dùng toàn bộ patch
token), nhưng tại chỗ truy hồi bằng embedding ảnh MedGemma không hơn soft token (CE
lexicon 0,240 so với 0,259; ROUGE-L 0,226 so với 0,220), probe 0,690 so với 0,725, và
MedGemma zero-shot thua v3 beam trên mọi chỉ số trừ METEOR. Train ảnh gốc đắt gấp ~4
lần. Không ưu tiên.

**TM-25 — Sửa Stage 1 nếu giữ META-Former.** Đã làm: 1b 3 epoch, 1c tùy chọn. Chưa làm:
bỏ CLS/token toàn cục khỏi K/V của Q-Former hoặc cộng lại embedding vị trí của query ở
mỗi tầng (kiểu DETR), phạt độ giống nhau giữa query, LayerNorm sau StreamAdapter, bỏ
ITM, 1a 2 epoch. Chắc chắn giảm sụp, chưa chắc tăng NLG; cần train lại 1a (cache ~148 GB,
`/home` còn 44 GB).

**TM-26 — CE bằng CheXbert.** Đã thực hiện: `scripts/score_ce_chexbert.py`, venv riêng
`~/.venvs/chexbert` (`f1chexbert`, chạy offline).

### 9.8 K — Khác bài báo nhưng không phải tính mới

**TM-27 — LLM.** Bài báo: Vicuna-7B LoRA khởi tạo từ RaDialog. Dự án: MedGemma 1.5 4B
IT, QLoRA NF4, LoRA r8/α16 như bài báo (D-027). Đổi vì phần cứng 16 GB và giấy phép.

**TM-28 — Stop token.** Sửa lỗi của chính dự án (Stage 2 bỏ qua `<end_of_turn>`, 78%
token sinh sau điểm kết thúc); CIDEr val n = 100: 0,0107 → 0,2369.

**TM-29 — Cài đặt huấn luyện.** Bài báo 20.000 bước biểu diễn + 5.000 bước sinh, RTX
4070 16 GB. Dự án: độ dài theo epoch (quyết định của user), RTX 5060 Ti 16 GB, bf16,
gradient checkpointing Q-Former, label smoothing ITC 0,1.

### 9.9 Chỗ file gốc đã cũ so với hiện trạng

Ghi lại để khi cập nhật kết quả cuối thì sửa cùng lúc:

- `CHI_MUC_TINH_MOI.md`: TM-02 còn ghi beam "chưa có trong `generate_stage2_reports.py`";
  TM-23 và TM-26 còn xếp loại E "chưa làm"; TM-25 chưa ghi phần đã làm.
- `TONG_HOP_2026-10-11.md` mục 0, 7, 8 còn ghi "chờ duyệt"; kết quả nằm ở mục 11.7–11.8.
- `PLAN-2026-10-11-stage2-context-pilot.md` và `PLAN-2026-10-11-stage1-improvement.md`:
  tiêu đề còn ghi "chờ user duyệt"; phần Execution report là trạng thái đúng.
- TM-13 ghi probe 0,713 (giao thức a), mục 2.4 của `TONG_HOP` ghi 0,725 (giao thức b) —
  cả hai đúng, khác giao thức (mục 3.4).

### 9.10 Phần giống bài báo

Ba encoder đóng băng cùng họ (BioViL-T ResNet50, PubMedCLIP ViT, MedCLIP Swin); chiếu về
1408 chiều rồi nối theo trục token; che ngẫu nhiên 10% đặc trưng mỗi encoder;
META-Former khởi tạo từ Q-Former BLIP-2 với ITC/ITM/ITG; MHCAC 6 tầng, 8 common expert
token, 14 expert token chuyên biệt, dropout 0,2, loss CE có trọng số + contrastive +
orthogonality + sparsity, không label smoothing; ba lớp P/N/U, ô trống = âm tính; lịch
ba pha (1a căn chỉnh, 1b chuyển sang MHCAC với LR 5e-5 → 2e-4 → 1e-5, 1c huấn luyện
chung — nay là tùy chọn); Stage 2 chiếu đầu ra META-Former vào LLM; LoRA r8/α16; tập
chia chính thức (val 1.808, test 3.269 study); bộ chỉ số BLEU/METEOR/ROUGE-L/CIDEr/
BERTScore và F1 5 bệnh.

Khác ở lần train P2: prompt có thêm indication và FINDINGS lần chụp trước; cue lúc suy
luận theo ngưỡng hai mức thay vì Eq. 22; cue lúc train lấy từ nhãn thật.

## 10. Tài liệu

### 10.1 Sinh báo cáo

| Nguồn | Thay đổi | Hiệu quả trên MIMIC-CXR |
|---|---|---|
| MAIRA-2 (arXiv 2406.04449) | ảnh + báo cáo lần trước, mục comparison | bỏ lúc suy luận ROUGE-L −28,9%; train không có −11,7% |
| MAIRA-2 | lateral + technique | bỏ lúc suy luận ROUGE-L −22% (tập có lateral) |
| MAIRA-2 7B | đầy đủ bối cảnh, MLP 4 lớp trên 1.369 token RAD-DINO | ROUGE-L 38,4, BLEU-4 23,4, RadGraph 34,6 |
| RaDialog (arXiv 2311.18681) | thêm indication | BLEU-4 9,5 → 14,8; ROUGE-L 27,1 → 31,6 |
| RaDialog | danh sách bệnh có cấu trúc | CE 26,1 → 39,4; BLEU-4 9,0 → 9,5 |
| RaDialog | LLM 7B / 13B / 33B | BLEU-4 9,5 / 9,5 / 9,5 — cỡ LLM không quan trọng |
| PromptMRG (AAAI 2024) | prompt chẩn đoán | CE F1 0,370 → 0,444; BLEU-4 0,116 → 0,106 |
| MedGemma (arXiv 2507.05201) | 4B PT không fine-tune; fine-tune ảnh + indication | RadGraph F1 29,5; 30,3 |
| MedGemma 1.5 (arXiv 2604.05081) | 4B IT; so sánh hai thời điểm | RadGraph F1 27,2; MS-CXR-T 65,7 |
| DeCo (arXiv 2405.20985) | Q-Former so với pooling | nén bằng query gây thiếu hụt ngữ nghĩa thị giác |
| ViT Registers (arXiv 2309.16588) | token hút attention | cần token đệm để giữ attention đúng chỗ |
| METransformer (CVPR 2023) | loss trực giao giữa token học được | giữ các token khác nhau |
| TRRG (arXiv 2408.12141) | số query 4 / 8 / 16 / 32 | tốt nhất 16 |
| MLRG (CVPR 2025) | đa góc + lần trước; CE bằng `f1chexbert` | SOTA BLEU-4, RadGraph |
| ReRef (arXiv 2609.19093) | đổi văn phong tham chiếu | đổi thứ hạng mô hình |

Mức v3 (beam BLEU-4 0,098, ROUGE-L 0,265) ngang RaDialog-align (9,5 / 27,1) — một hệ
Q-Former 32 query gần giống META-CXR mà Q-Former không sụp. Tức là chỉ sửa Q-Former khó
kéo NLG vượt mức này; đòn bẩy lớn nằm ở bối cảnh đầu vào.

### 10.2 Phân loại / encoder

| Nguồn | Nội dung | Hàm ý |
|---|---|---|
| Tayebi Arasteh et al., arXiv 2510.07191 | trên MIMIC-CXR, LoRA thua fine-tune toàn phần 2,1–4,1 điểm AUROC; đặc trưng đóng băng DINOv3-7B thua backbone 86–89M được fine-tune; 512 px > 224 px | đóng băng là trần thấp |
| arXiv 2406.10050 | linear probe ~80–82 so với fine-tune ~87–88 AUROC; LP-FT có lợi với ResNet-50 | cùng chiều |
| Kumar et al., ICLR 2022 (LP-FT), arXiv 2202.10054 | fine-tune hơn linear probe ~2% trong phân phối; LP rồi FT hơn FT ~1% | "đọc trước, tinh chỉnh sau" là hợp lý |
| R2GenGPT, arXiv 2309.09812 | Swin đóng băng → fine-tune: BLEU-4 0,117 → 0,134, CE F1 0,325 → 0,389 | thích nghi encoder giúp cả sinh báo cáo |
| arXiv 2609.09185 | đặc trưng đóng băng, MIMIC 14 nhãn: RAD-DINO 0,827, BioViL-T 0,805, nối hai 0,837 AUROC | RAD-DINO đáng đo — đã đo, không hơn (5.5) |
| MedGemma (arXiv 2507.05201) | MedSigLIP mạnh ở phân loại CXR zero-shot | probe trung bình 256 token (0,690) không phản ánh đủ kênh ảnh nhiều token |
| VisualCheXbert (arXiv 2102.11467), Jain et al. (arXiv 2104.00793) | nhãn từ báo cáo lệch nhãn từ ảnh | nhiễu nhãn là giới hạn thật (5.2) |
| Pham et al. (arXiv 1911.06475) | nhãn Uncertain làm mềm (LSR) cải thiện AUC trên CheXpert | lớp Uncertain yếu nhất (AUROC 0,663) |
| Kiểm định 2025 (arXiv 2509.15107) | nhãn công khai MIMIC/CheXpert khớp chuyên gia < 60% (mẫu nhỏ) | cùng chiều 5.2 |

## 11. Bảy đề xuất ban đầu và kết cục

| # | Đề xuất | Độ chắc chắn lúc đề xuất | Kết cục 11/10 |
|---|---|---|---|
| 1 | FINDINGS lần chụp trước + INDICATION trong prompt Stage 2 | cao | đã làm; pilot đạt quy tắc; đang train toàn bộ |
| 2 | Train Stage 2 để dùng cue (nhãn thật lúc train, MHCAC lúc suy luận) | trung bình | đã làm; pilot chưa đạt ngưỡng CE lexicon nhưng CheXbert F1-5 tăng; user chọn dùng |
| 3 | Beam 4 + length penalty 2 trong bộ sinh chính thức | cao | đã làm |
| 4 | CE bằng CheXbert | cao (thước đo) | đã làm |
| 5 | 1b còn 3 epoch; bỏ 1c hoặc thay bằng pha căn chỉnh riêng | trung bình–cao | đã làm (1c thành tùy chọn); lịch 3 epoch chưa chạy |
| 6 | Sửa sụp META-Former | trung bình | chưa làm |
| 7 | Kênh ảnh gốc MedGemma thay soft token | thấp | chưa làm, không ưu tiên |

**Không đề xuất** (đã đo, không có lợi): thêm epoch 1a/1b; tăng số ảnh phụ; cờ EVOKE;
finding token; logit-adjusted loss (đổi F1 lấy recall, user đã từ chối); 1c như cấu hình
cũ; mở khóa sâu; RAD-DINO; sửa luồng BioViL trong ngân sách pilot; đổi LLM lớn hơn
(RaDialog: không đổi).

## 12. Việc còn mở

- Kết quả test của lần train P2 (mục 8.3).
- Epoch thứ hai của Stage 2 có giúp không — chỉ biết khi có hai val loss.
- Tách đóng góp của indication và của báo cáo lần chụp trước (pilot gộp hai thứ).
- 1a train lại trên bố cục patch của BioViL — pilot 5.7 không thử.
- Sửa sụp META-Former (đề xuất 6).
- Việc mở khóa CLIP block 10–11 có thực sự đổi trọng số không (5.5).
- Stage 2 lấy soft token từ checkpoint 1a hay 1c: 1b làm soft token kém đi 0,031, 1c kéo
  lại; lần train P2 dùng checkpoint 1c.
- Ảnh hưởng của việc record train Stage 2 tính từ ảnh có augmentation (5.8).

## 13. Giới hạn

- CE lexicon là xấp xỉ bằng luật; số tuyệt đối không so được với CE CheXbert.
- Mốc truy hồi và probe dùng 50.000–60.000 study train, không phải toàn bộ.
- Mỗi phase một checkpoint, một seed; mỗi arm pilot một run, một seed.
- Cue oracle đo trên 1.000 study, greedy.
- Pilot Stage 2: 10.000 study train, 500 study val — hiệu ghép cặp có giá trị, số tuyệt
  đối không so được với test.
- Embedding MedGemma lấy trung bình 256 token: không nói gì về việc LLM đọc đủ 256 token
  không gian thì sao.
- Số chính Stage 1 không phải giao thức argmax của bài báo; mô hình Stage 2 mới không
  cùng điều kiện đầu vào với bài báo.
- MedCLIP được pretrain trên MIMIC-CXR + CheXpert với cách chia không kiểm chứng được.
- Nhãn CheXpert tự động vừa là nhãn train vừa là nhãn chấm (5.2).

## 14. Nguồn

- META-CXR: D. Edirisinghe et al., IEEE Access 13, 2025.
- MAIRA-2: https://arxiv.org/abs/2406.04449
- RaDialog: https://arxiv.org/abs/2311.18681
- PromptMRG: https://arxiv.org/abs/2308.12604
- MedGemma: https://arxiv.org/abs/2507.05201 ; MedGemma 1.5: https://arxiv.org/abs/2604.05081
- DeCo: https://arxiv.org/abs/2405.20985
- Vision Transformers Need Registers: https://arxiv.org/abs/2309.16588
- METransformer: https://mlanthology.org/cvpr/2023/wang2023cvpr-metransformer
- TRRG: https://arxiv.org/abs/2408.12141
- MLRG: https://arxiv.org/abs/2502.20056
- ReRef (Reporting Practice Matters): https://arxiv.org/abs/2609.19093
- EVOKE: https://arxiv.org/abs/2411.10224
- LP-FT (Kumar et al., ICLR 2022): https://arxiv.org/abs/2202.10054
- Resolution scaling / DINOv3 trên CXR (Tayebi Arasteh et al.): https://arxiv.org/abs/2510.07191
- Comparison of fine-tuning strategies: https://arxiv.org/abs/2406.10050
- R2GenGPT: https://arxiv.org/abs/2309.09812
- RAD-DINO + BioViL-T: https://arxiv.org/abs/2609.09185
- RAD-DINO: https://arxiv.org/abs/2401.10815
- VisualCheXbert: https://arxiv.org/abs/2102.11467 ; Jain et al.: https://arxiv.org/abs/2104.00793
- Pham et al. (LSR): https://arxiv.org/abs/1911.06475
- Kiểm định nhãn 2025: https://arxiv.org/abs/2509.15107
