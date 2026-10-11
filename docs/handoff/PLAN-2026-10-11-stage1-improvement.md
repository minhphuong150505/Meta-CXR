# PLAN 2026-10-11 — Stage 1: tăng epoch hay đổi gì để chỉ train một lần (ĐỀ XUẤT, CHỜ ĐO + USER DUYỆT)

Trạng thái: **đề xuất. Chưa sửa config, chưa train.** Các phép đo ở mục 4 chỉ đọc
checkpoint có sẵn, không train gì; chạy sau pilot Stage 2 vì máy chỉ có một GPU.
Câu hỏi của user (2026-10-11): sau khi rút gọn Stage 1, tăng epoch 1a/1b có tốt hơn
không, hay có cách nào khác — nghiên cứu như đã làm cho Stage 2, để không phải train
lại nhiều lần.

## 1. Tăng epoch: không — đã có bằng chứng từ các run cũ

| phase | bằng chứng | kết luận |
|---|---|---|
| 1a (4 epoch, batch 128) | gate R@5 theo epoch 0,61 / 0,63 / 0,66 / 0,67; val ITC loss thấp nhất ở epoch 1 rồi tăng (2,95 → 3,38); ITM ở mức ngẫu nhiên cả pha; 32 soft token đã sụp thành một vector ở cuối 1a (`TONG_HOP_2026-10-11.md` mục 2) | thêm epoch chỉ tăng R@5 ~0,01/epoch, val loss xấu đi, không sửa được sụp |
| 1b (5 epoch) | val AUROC 0,757 / 0,764 / **0,778** / 0,775 / 0,773; macro recall đỉnh ở epoch chỉ số 2; `run_20260930_3class`: loss chọn epoch 3, F1 chọn epoch 2; run cũ (cách chấm nhị phân cũ) kéo 10 → 15 epoch: test AUROC −0,0005 [−0,0043; +0,0031] | 1b bão hòa ở epoch 2–3 |

**Vì sao bão hòa sớm:** sau 10% đầu của 1b chỉ còn MHCAC, StreamAdapter, view fusion
và đầu MPC được học; cả ba encoder đóng băng, và `shared_visual_projector` + `ln_vision`
cũng bị đóng băng (nằm trong `fade_out`). Đường phân loại là một bộ đọc trên đặc trưng
cố định — gần với linear probe mở rộng — nên hội tụ trong 2–3 epoch, sau đó chỉ quá khớp.
Muốn vượt trần này phải cho phần biểu diễn ảnh thích nghi với bài toán, không phải train
lâu hơn.

## 2. Đòn bẩy có bằng chứng: để encoder thích nghi với bài toán phân loại

**Trong repo** (run đủ dài, toàn bộ dữ liệu, cách chấm cũ — nhị phân `study_presence`):
`run_20260820_ft` mở khóa encoder nông (ResNet50 `layer4` + projector, CLIP block 10–11 +
`post_layernorm`, LR encoder 1e-5) so với encoder đóng băng: macro AUROC 0,7441 → 0,7643,
**+0,0201 [+0,0136; +0,0263], tăng ở 14/14 bệnh**; F1, precision, recall cùng tăng có ý
nghĩa. Lưu ý: kappa đổi cùng lúc (kappa chỉ dịch điểm làm việc, AUROC không phụ thuộc
kappa). Mở khóa sâu hơn (`layer3`, CLIP 8–9) không thêm gì: +0,0049 [−0,0003; +0,0100].

**Bằng chứng ngược trong repo, và vì sao nó không bác bỏ điều trên:** pha 1c của
`run_20261005_paper` cũng mở khóa encoder nông nhưng không tăng AUROC test (1b tốt nhất
so với 1c: +0,0037 [−0,0012; +0,0095]). Ở 1c, gradient căn chỉnh (ITC/ITM/LM, w = 1)
lớn gấp 7–35 lần gradient phân loại trên tham số chung (đã đo), ITC chạy ở batch 8, và
pha chỉ dài 1 epoch — tức encoder được kéo bởi mục tiêu căn chỉnh chứ không phải phân loại.

**Tài liệu:**

| Nguồn | Kết quả | Hàm ý |
|---|---|---|
| Tayebi Arasteh et al., arXiv 2510.07191 (7 bộ CXR, gồm MIMIC-CXR) | LoRA thấp hơn fine-tune toàn phần 2,1–4,1 điểm AUROC trên MIMIC-CXR; đặc trưng đóng băng của DINOv3-7B thua backbone 86–89M được fine-tune; 512 px tốt hơn 224 px | thích nghi encoder > đóng băng, kể cả với encoder rất lớn |
| Comparison of fine-tuning strategies, arXiv 2406.10050 | linear probe ~80–82 AUROC so với fine-tune toàn phần ~87–88; LP-FT có lợi với ResNet-50 / DenseNet-121 | cùng chiều |
| Kumar et al., ICLR 2022 (LP-FT), arXiv 2202.10054 | trong phân phối: fine-tune hơn linear probe ~2%; LP rồi FT hơn FT ~1% | thứ tự "đọc trước, tinh chỉnh sau" là an toàn |
| R2GenGPT, arXiv 2309.09812 | Swin đóng băng → fine-tune: BLEU-4 0,117 → 0,134, CE F1 0,325 → 0,389 trên MIMIC-CXR | thích nghi encoder cũng giúp sinh báo cáo |
| arXiv 2609.09185 (đặc trưng đóng băng, 14 nhãn MIMIC, U-Ones) | RAD-DINO 0,827 AUROC, BioViL-T 0,805; nối hai nguồn 0,837 | RAD-DINO đáng đo (mục 4, M2) |

## 3. Đề xuất cấu hình (CHƯA áp dụng — cần user duyệt)

**Phương án A (khuyến nghị): 1b mở khóa encoder nông ngay từ đầu, tái dùng 1a, không 1c.**
Thay đổi chỉ ở config, khối `run.phases.phase1b` của `mimic_cxr_full.yaml`:

```yaml
    phase1b:
      epochs: 3
      unfreeze_encoder_blocks: true          # MỚI
      model:
        encoder_finetune:                    # MỚI — đúng bộ của run_20260820_ft
          enabled: true
          patterns:
            - visual_encoder.encoder.encoder.layer4
            - visual_encoder.projector
            - pubmedclip.model.vision_model.encoder.layers.10
            - pubmedclip.model.vision_model.encoder.layers.11
            - pubmedclip.model.vision_model.post_layernorm
          keep_batchnorm_eval: true
        # phần loss/multi_view giữ nguyên
```

- Chạy: `PHASES="phase1b"`, khởi tạo từ `checkpoint_phase1a.pth` đã có (bản đã bỏ
  `mhcac.expert_tokens`, như `run_20261005_paper`). Không cần train lại 1a, không cần
  cache 148 GB.
- Chi phí ước tính: 1b hiện 0,36 s/it, 13.922 it/epoch ≈ 1,4 h/epoch; mở khóa nông đã đo
  +37% thời gian/epoch → ~1,9 h/epoch, **~5,7 h cho 3 epoch** + chấm val/test. VRAM 1b
  ~7,0 GB, mở khóa nông thêm ~5–10%.
- **Quy tắc chấp nhận (chốt trước):** so với 1b tốt nhất của `run_20261005_paper` trên
  cùng 3.269 study test, bootstrap ghép cặp: chấp nhận nếu AUROC trung bình (ba lớp) có
  cận dưới > 0; nếu không, giữ checkpoint hiện có — **không train lại**. Cả hai đều dùng
  quy tắc hai ngưỡng fit trên val cho số chính.
- Rủi ro: encoder đổi làm đầu vào Q-Former (đã đóng băng) lệch → soft token cho Stage 2
  có thể kém đi. Đo ở M3; nếu lệch đáng kể, Stage 2 lấy soft token từ checkpoint 1a (cần
  sửa code Stage 2: một checkpoint cho soft token, một cho cue).

**Phương án B (LP-FT):** giữ 1b như hiện tại (3 epoch, đóng băng), thêm một pha phân loại
thuần 1–2 epoch mở khóa encoder nông với ITC/ITM/LM = 0, LR thấp. Có cơ sở tài liệu
(LP-FT) nhưng chưa có bằng chứng trong repo; đắt hơn A ~1,9–3,8 h.

## 4. Đo trước khi train (chỉ đọc checkpoint, ~1,5 h GPU sau pilot Stage 2)

Script: `~/stage1_meas_20261011.sh` trên máy train (mã `scripts/embed_cxr_encoders.py`,
`scripts/probe_image_features.py`, `scripts/analyze_radiologist_labels.py`). Cùng tập
con 60.000 study train / 2.800 test như các probe trước, ảnh sạch (không augmentation),
probe tuyến tính ba lớp, AUROC trung bình trên test.

| # | So sánh | Quy tắc đọc (chốt trước) |
|---|---|---|
| M1 | BioViL / PubMedCLIP pretrained so với cùng encoder sau 1c (1 epoch mở khóa) và sau `run_20260820_ft` (10 epoch mở khóa) | hơn ≥ 0,01 → fine-tune làm biểu diễn tốt lên, ủng hộ phương án A |
| M2 | RAD-DINO so với từng encoder hiện tại; nối ba encoder hiện tại so với nối thêm RAD-DINO | RAD-DINO ≥ encoder tốt nhất + 0,01 **và** nối thêm ≥ +0,005 → đề xuất tích hợp RAD-DINO (cần code: bố cục token gốc, gộp patch) |
| M3 | soft token sau 1a / 1b / 1c | 1a hơn 1b ≥ 0,01 → độ lệch đầu vào có hại; Stage 2 nên lấy soft token từ 1a |
| M4 | BioViL trung bình toàn ảnh so với 2×2 vùng | vùng hơn ≥ 0,01 (nhất là bệnh khu trú) → mất vị trí là cái giá thật của việc soft token sụp; ủng hộ kênh ảnh nhiều token cho Stage 2 |
| M5 | MHCAC 1b và 1c chấm theo nhãn bác sĩ (685 study test có cả hai loại nhãn) so với nhãn tự động | ước lượng phần sai số do nhãn |

## 5. Không đề xuất (đã đo, không có lợi)

Thêm epoch 1a/1b (mục 1); 3 ảnh phụ (`run_20261007_aux3`, AUROC −0,0085); cờ EVOKE
(`run_20261009_evoke_d`, F1-5 −0,016); loss logit-adjusted (mô phỏng: đổi F1 lấy recall,
user đã từ chối); 1c như cấu hình hiện tại (không tăng AUROC, phá R@5 0,65 → 0,17); mở
khóa sâu (không hơn mở khóa nông).

## 6. Thứ tự train một lần

1. Pilot Stage 2 (đang chạy) → chọn bối cảnh / cách train cue.
2. Đo M1–M5 (mục 4) → chốt phương án Stage 1 và nguồn soft token.
3. User duyệt → Stage 1 phương án A (~6 h), một lần, có checkpoint hiện tại làm dự phòng.
4. Stage 2 toàn bộ một lần với cấu hình thắng pilot, cue từ checkpoint Stage 1 mới.

## 7. Cập nhật 2026-10-11 chiều: kết quả đo và pilot được duyệt

Kết quả M1–M5: `docs/TONG_HOP_2026-10-11.md` mục 11.5–11.6. Hai điểm đổi kế hoạch:
RAD-DINO không hơn MedCLIP-Swin (không tích hợp); và luồng BioViL có hai lỗi kế thừa từ
code gốc — token bị trộn (`reshape` không hoán vị trục) và projector 1408 chiều là ngẫu
nhiên, đóng băng trong 1a/1b. "Mở khóa encoder nông" (mục 3) vì thế chưa đủ.

**Pilot (user duyệt 2026-10-11), `~/s1pilot.sh` → `~/s1pilot_20261011/`, mã `9b79822`:**
chỉ pha 1b, 60.000 study train đầu tiên, 2 epoch, warm-up 150 update, cùng checkpoint 1a
của `run_20261005_paper`, không chấm test.

| arm | khác gì |
|---|---|
| A | như cấu hình đang ship |
| B | `model.biovil.token_layout=patch`; `visual_encoder.projector` được train; `shared_visual_projector` + `ln_vision` không bị đóng băng sau 10% |
| C | B + mở khóa encoder nông (5 pattern của `run_20260820_ft`) |

**Quy tắc chọn (chốt trước):** trên 1.808 study val, epoch tốt nhất của mỗi arm theo
`macro_recall`; bootstrap ghép cặp AUROC trung bình ba lớp. Nhận B (hoặc C) nếu hiệu so
với A có cận dưới > 0; nhận C thay B nếu C − B có cận dưới > 0. Không arm nào thắng: giữ
`run_20261005_paper`, không train lại Stage 1. Lưu ý giới hạn: B và C khởi từ checkpoint 1a
đã học trên bố cục token cũ, nên pilot đo phần lợi cho MHCAC, không đo Q-Former.
