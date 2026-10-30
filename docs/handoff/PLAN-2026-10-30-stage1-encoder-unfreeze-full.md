# PLAN 2026-10-30: mở khóa encoder nông trong pha 1b ở budget đầy đủ (5 epoch, 220.216 study)

**Trạng thái:** ĐÃ CHẠY 2026-10-30 — **L42 TỆ HƠN** theo quy tắc đã chốt (xem Execution report). Không
chạy giai đoạn B. User chọn hướng này ("3") sau khi ba pilot tăng recall
(ASL, U5 ở budget pilot, cue Eq. 22) đều thất bại. Không sửa code, không sửa YAML: chỉ override cấu hình,
giống hệt U5 (`PLAN-2026-10-29-stage1-encoder-unfreeze-1b.md`).

## 1. Câu hỏi

Pilot U5 (60.000 study, 2 epoch) cho kết quả TỆ HƠN: macro AP "có bệnh" −0,0093 [−0,0143; −0,0045].
Pilot đó không trả lời được một câu hỏi: encoder có cần nhiều bước train hơn mới thích nghi có lợi hay
không. `run_20260820_ft` (10 epoch, kiến trúc cũ) từng thấy AUROC tăng ở 14/14 finding, mạnh nhất ở các
finding hiếm. Plan này kiểm tra câu hỏi đó ở **đúng budget pha 1b của mô hình đang báo cáo**: 5 epoch,
toàn bộ tập train, khoảng 9,2 lần budget pilot.

Vì sao chọn 5 epoch chứ không phải 10: `run_20261005_paper/phase1b` đã có sẵn ở 5 epoch, seed 42, kèm
file dự đoán val của từng epoch. Nó là control miễn phí. Chạy 10 epoch thì phải chạy thêm một control
10 epoch (khoảng 14 giờ nữa).

## 2. Arm

| Arm | Công thức | Seed | Trạng thái |
|---|---|---|---|
| C42 (control) | `run_20261005_paper/phase1b`: 5 epoch, warm-up 500, chọn checkpoint theo `macro_recall` | 42 | đã có |
| L42 | giống hệt C42, cộng mở khóa 5 pattern nông như U5 | 42 | chạy mới |

**Override cho L42.** Từ `run_20261005_paper/checkpoint_phase1a.pth` (bản copy), qua
`scripts/run_stage1_phases.sh` với `PHASES="phase1b"`:

```
P=run.phases.phase1b
PAT=[visual_encoder.encoder.encoder.layer4,visual_encoder.projector,pubmedclip.model.vision_model.encoder.layers.10,pubmedclip.model.vision_model.encoder.layers.11,pubmedclip.model.vision_model.post_layernorm]
$P.epochs=5 $P.run.test_splits=[] run.seed=42
$P.unfreeze_encoder_blocks=true $P.model.encoder_finetune.enabled=true
$P.model.encoder_finetune.patterns=$PAT $P.model.encoder_finetune.keep_batchnorm_eval=true
```

`$P.epochs=5` cần thiết vì YAML hiện để 1b là 3 epoch (thay đổi ngày 2026-10-11), còn C42 chạy 5.
Không có `truncate_train`, warm-up giữ 500 như C42.

## 3. Các cổng kiểm tra (dừng ngay nếu một cổng không qua)

- **G0.** Host pull lên commit của plan này (chỉ đổi tài liệu so với `6a438ac`), checkout sạch. GPU rảnh,
  không job khác, ổ `ntfs3`, `/home` còn ít nhất 15 GB, thư mục đích chưa có.
- **G1: code hiện tại có tái tạo đúng C42 không.** C42 chạy bằng code ngày 2026-10-05. Nếu code đã trôi
  thì so L42 với C42 sẽ lẫn hai thay đổi. Chạy công thức C42 (không mở khóa, seed 42, 5 epoch) bằng code
  hiện tại trong khoảng 700 iteration rồi dừng. So các dòng log ở iteration 0, 50, …, 650: `lr` và mọi
  giá trị loss in ra phải **giống hệt** dòng tương ứng trong `~/run_20261005_paper.log`. Seed 42 chạy
  lại đã từng cho kết quả giống từng bit (2026-10-09). Khoảng 10 phút.
  - Nếu không khớp: dừng, báo user. Khi đó cần một control mới chạy bằng code hiện tại (khoảng 7 giờ),
    và đó là quyết định riêng.
- **G2.** Smoke L42 (2.000 study, 1 epoch) phải in `unfroze 69 parameters (31.85M) across 5 patterns;
  batchnorm_eval=True`, cùng `visual_encoder 17.67M` và `pubmedclip 14.18M` ở dòng trainable. Không có
  Traceback, OOM hay NaN. Khoảng 10 phút.
- **Arm thật:** rc=0, seed 42, đủ 5 file `val_predictions_epoch_{0..4}.npz` cộng `..._best.npz`, không có
  file test nào.

**Chi phí:** C42 mất 7h08m. Ở pilot, U5 và P15 nhanh ngang nhau (U5 44–54 phút, P15 46–51 phút). Ước
tính L42 **khoảng 7–8 giờ**, cộng G1 và G2 khoảng 20 phút. **Tổng khoảng 8 giờ GPU.** Đỉnh VRAM khoảng 6,8 GB (như pilot). Ổ đĩa
khoảng 6 GB cho checkpoint.

## 4. Quy tắc đọc kết quả, chốt TRƯỚC khi chạy

**Chỉ một seed mỗi arm, và chưa có ngưỡng nhiễu (NB) đo ở budget đầy đủ.** Plan dùng NB đo ở budget pilot
làm ngưỡng tạm (macro AP 0,0042; wF1 0,0027; F1-5 0,0060). Vì một seed không tách được nhiễu seed, giai
đoạn này **chỉ quyết định có đi tiếp hay không, không bao giờ kết luận "tốt hơn"**.

- **Chỉ số quyết định:** macro AP "có bệnh" (P hoặc U so với N, trên `s = p_pos/(p_pos+p_neg)`),
  trung bình 14 finding, trên val (1.808 study). So file `val_predictions_epoch_best.npz` của L42 với của
  C42. Cả hai đều chọn checkpoint theo cùng quy tắc `macro_recall`.
- Δ = L42 − C42. Khoảng tin cậy bằng bootstrap ghép cặp theo study, 1.000 lần, seed 16.
- Cutpoint cho rào chắn: cross-fit trên val, giống các pilot trước.

**ĐI TIẾP sang giai đoạn B** khi đạt đủ ba điều kiện:

1. Δ macro AP > +0,0042.
2. Cận dưới khoảng tin cậy > 0.
3. Rào chắn: Δ wF1 ≥ −0,0027 và Δ F1-5 ≥ −0,0060.

**TỆ HƠN** khi Δ macro AP < −0,0042, hoặc khi rào chắn không đạt.

**Mọi trường hợp còn lại:** KHÔNG XÁC LẬP. Dừng hướng này.

**Chỉ để mô tả:**

- Đường macro AP qua từng epoch (0–4) của cả hai arm. Đây là thứ trả lời trực tiếp câu hỏi "khoảng cách
  có lớn dần theo số bước train không".
- AUROC "có bệnh", recall ở precision mốc, recall và precision P/U dưới cutpoint và dưới argmax.
- AP theo từng finding, nhất là các finding hiếm: Enlarged Cardiomediastinum, Fracture, Pleural Other,
  Lung Lesion, Consolidation, Pneumothorax.

**Giai đoạn B (chỉ khi ĐI TIẾP, và cần user duyệt riêng):** chạy cả hai arm ở seed 17, gồm control 1b
khoảng 7 giờ và L khoảng 7–8 giờ. Đọc bằng quy tắc hai seed của pilot (Δ > 0 ở cả hai seed, trung bình
> NB/√2, cận dưới CI > 0, rào chắn). Chỉ khi B đạt mới tính đến pha 1c, cutpoint mới, chấm test một lần,
và kiểm tra ảnh hưởng lên soft token của Stage 2.

Test không được chấm trong plan này.

## 5. Giới hạn đã biết trước

- Một seed mỗi arm, chỉ val. NB ở budget đầy đủ chưa đo; NB pilot chỉ là ngưỡng tạm.
- So ở cuối pha 1b. Pha 1c của mô hình đang báo cáo cũng mở khóa 5 pattern này trong 1 epoch, nên mô
  hình cuối cùng (1b+1c) sẽ khác ít hơn.
- Encoder thay đổi thì đầu vào của Q-Former (đóng băng sau 10% đầu 1b) bị lệch, nên soft token cho
  Stage 2 có thể kém đi.
- LR encoder dùng chung `init_lr_enc` 1e-5 cho mọi pattern; không thử LR khác.
- Swin (MedCLIP) không nằm trong 5 pattern.

## Execution report

**Kết luận theo quy tắc đã chốt: TỆ HƠN** (Δ macro AP −0,0079 < −0,0042). Rào chắn đạt. Không chạy giai
đoạn B; dừng hướng này.

**Chạy:**

- User duyệt ("ok") ngày 2026-10-30. Host ở `4338f16` (code giống `6a438ac`, chỉ khác tài liệu).
- **G1** lần đầu dừng (09:36): log hiện tại có thêm cột `loss_query_div`, luôn bằng 0 (knob mặc định tắt từ
  2026-10-15). User duyệt so sánh lại sau khi bỏ cột đó: 14/14 dòng (iteration 0–650) giống hệt log C42 về
  `lr` và mọi loss. **G1 qua.**
- **G2** qua: smoke 0:00:52, `max mem` 6.811 MiB, log in `unfroze 69 parameters (31.85M) across 5 patterns;
  batchnorm_eval=True`.
- **L42:** chuỗi `~/unf_full2.sh` chạy đúng một lần, L42 bắt đầu 09:39:33, xong 16:51:13; `Training time
  7:09:50` (C42 7h08m), `max mem` khoảng 6,8 GB, không có Traceback/NaN. Đủ `val_predictions_epoch_{0..4}`
  cộng `_best`, không có file test. Probe `~/unf_full_probe.py` chạy tự động, kết thúc `UNF_FULL_DONE`
  16:51:40. Checkpoint chọn theo `macro_recall`: L42 epoch 4, C42 epoch 2.

**L42 − C42, val n = 1.808, file `_best`:**

| Chỉ số | C42 | L42 | Δ |
|---|---|---|---|
| **macro AP "có bệnh"** (quyết định) | 0,4329 | 0,4249 | **−0,0079** [−0,0166; −0,0001] |
| AUROC "có bệnh" | 0,8053 | 0,8000 | −0,0052 |
| Recall ở precision mốc (RP12) | 0,4640 | 0,4314 | −0,0327 |
| wF1 (cutpoint, cross-fit) — rào chắn | 0,8578 | 0,8577 | −0,0002 (≥ −0,0027, đạt) |
| F1-5 (cutpoint, cross-fit) — rào chắn | 0,8067 | 0,8029 | −0,0038 (≥ −0,0060, đạt) |
| Recall P/U dưới cutpoint | 0,5598 | 0,5576 | −0,0022 |
| Precision P/U dưới cutpoint | 0,6336 | 0,6330 | −0,0006 |
| Recall P/U dưới argmax | 0,7590 | 0,7563 | −0,0027 |
| Precision P/U dưới argmax | 0,4829 | 0,4716 | −0,0113 |

Bootstrap ghép cặp theo study, 1.000 lần, seed 16.

**Đường macro AP qua từng epoch (chỉ mô tả):**

| Epoch | C42 | L42 | Δ |
|---|---|---|---|
| 0 | 0,4116 | 0,4051 | −0,0064 |
| 1 | 0,4172 | 0,4148 | −0,0024 |
| 2 | 0,4329 | 0,4216 | −0,0113 |
| 3 | 0,4320 | 0,4250 | −0,0071 |
| 4 | 0,4326 | 0,4249 | −0,0077 |

L42 thấp hơn C42 ở mọi epoch, và khoảng cách không thu hẹp dần theo số bước train.

**AP theo finding (Δ L42 − C42, chỉ mô tả):** giảm ở 9/14. Finding hiếm: Pleural Other −0,056, Pneumothorax
−0,016, Enlarged Cardiomediastinum −0,013, Fracture −0,002, Consolidation −0,001; chỉ Lung Lesion tăng
(+0,030). Các finding khác: Cardiomegaly −0,027, Support Devices −0,023, Edema −0,017; tăng ở Lung Opacity
+0,010, Pneumonia +0,008, Pleural Effusion +0,001.

**Đọc kết quả:** ở đúng budget pha 1b của mô hình đang báo cáo (5 epoch, toàn bộ train), mở khóa 5 pattern
nông vẫn làm xếp hạng kém đi, cùng hướng với pilot U5 (−0,0093) và arm C ngày 2026-10-11. Giả thuyết "encoder
cần nhiều bước hơn mới có lợi" không được ủng hộ: khoảng cách đã có từ epoch 0 và không giảm.

**Giới hạn:** một seed mỗi arm, chỉ val; chưa có NB ở budget đầy đủ (dùng NB pilot 0,0042 làm ngưỡng tạm).
Cận trên CI −0,0001 sát 0. Không thử LR encoder khác (dùng chung 1e-5).
