# PLAN 2026-10-31: tinh chỉnh toàn bộ MedCLIP-Swin — có phá được trần của encoder đóng băng không?

**Trạng thái:** user duyệt plan 2026-10-31; code đã viết và push (mục 5). **CHƯA chạy trên GPU** — cần user
duyệt riêng lần chạy. Chỉ đo trên val; không train Stage 1, không sửa YAML, không chấm test.

## 1. Câu hỏi và bằng chứng sẵn có

- Phép đo encoder 2026-10-31 (`PLAN-2026-10-31-encoder-probe.md`): một probe attention trên MedCLIP-Swin
  **đóng băng**, fit trên 40.000 study, đạt macro AP "có bệnh" 0,4263 trên val. Mô hình đang báo cáo (R1C) đạt
  0,4271, chênh −0,0008 [−0,0087; +0,0068]. Không encoder đóng băng nào khác vượt được Swin.
- Tức là trần hiện tại là **trần của biểu diễn Swin đóng băng**. Chỉnh loss, epoch hay TTA đều chạm cùng trần đó.
- **Swin chưa từng được mở khóa trong bất kỳ lần chạy nào.** Mọi lần mở khóa encoder trước đây (run_20260820_ft,
  pha 1c, U5, L42) chỉ mở BioViL (layer3/4, projector) và PubMedCLIP (block 8–11). `encoder_finetune.patterns`
  không có mục nào của `swin`. Vậy các kết quả âm U5/L42 **không** trả lời câu hỏi cho Swin.
- Tài liệu: tinh chỉnh toàn phần thường hơn probe đóng băng vài điểm AUROC trên CheXpert/MIMIC. Chưa rõ với
  MedCLIP-Swin ở 224 px trên dữ liệu này, nên phải đo.

**Câu hỏi duy nhất:** cùng dữ liệu, cùng đầu probe, cùng budget, mở khóa toàn bộ Swin có tăng macro AP thêm
≥ 0,02 so với để đóng băng không?

## 2. Thiết kế

### 2.1 Dữ liệu (giữ nguyên phép đo encoder)

- **Train:** đúng 40.000 study của phép đo encoder (`np.sort(default_rng(16).choice(N, 40000))` trên thứ tự dataset
  train), chia đúng như cũ: 35.654 study fit, 3.961 study holdout (10% study có nhãn, hoán vị seed 16).
- **Val:** toàn bộ 1.808 study. **Test không đụng tới.**
- Chỉ ảnh anchor (`max_aux_views=0`), không có view phụ.
- **Cache ảnh:** một lần duy nhất, lưu ảnh đã pad vuông + resize 224 bicubic dạng uint8 (đúng các bước của
  `medclip_preprocess` TRƯỚC khi chia 255 và chuẩn hóa), cùng `labels.npy`, `keys.npy`. Khoảng 2 GB, nằm ở
  `~/swinft_20261031/cache/` trên máy train (dữ liệu dẫn xuất MIMIC, không bao giờ vào repo). Lý do: mỗi epoch
  khỏi giải mã lại ảnh gốc 7 MP, và hai nhánh thấy đúng cùng một đầu vào.

### 2.2 Hai nhánh, chỉ khác một điều

| | F (đối chứng) | T (tinh chỉnh) |
|---|---|---|
| Swin | đóng băng, `eval()` | mở khóa toàn bộ (~28M tham số) |
| Đầu probe | `build_attention_probe` của `encoder_probe_fit.py`: 49 token lưới + token pooled, 14 query, 3 lớp | giống hệt |
| LR | đầu 3e-4 | đầu 3e-4; Swin giảm dần theo tầng (layer-wise decay 0,7): stage 4 = 5e-5, stage 3 = 3,5e-5, stage 2 = 2,45e-5, stage 1 = 1,7e-5, patch embedding = 1,2e-5 |

Chung cho cả hai nhánh, chốt trước, không dò:

- AdamW, weight decay 0,05 (không áp cho bias/LayerNorm), warmup 5% rồi cosine, **8 epoch**, batch 64, bf16
  autocast, seed 16.
- Augmentation nhẹ trên ảnh cache 224, chỉ lúc train: xoay ±10°, phóng 0,9–1,1, tịnh tiến ±5%, độ sáng/tương
  phản ±0,1. **Không lật ngang** (sai giải phẫu). Val và holdout không augmentation.
- Loss: CE ba lớp từng finding, ô không nhãn (−100) bỏ qua. Giống probe, không trọng số lớp.
- Chọn epoch theo macro AP trên **holdout**. Val chỉ đọc ở epoch đã chọn.
- Ghi theo epoch: train loss, macro AP holdout (để thấy overfit).

Layer-wise decay vì CLAUDE.md ghi nhận các lần mở khóa trước dùng một LR 1e-5 cho mọi tầng và coi đó là "điều
đầu tiên cần sửa nếu thử lại".

## 3. Cổng an toàn

- **G0 (dữ liệu):** `keys.npy` và `labels.npy` của cache phải giống từng phần tử file tương ứng trong
  `~/encprobe_20261031/{train,val}/` (train: 40.000 key; val: 1.808). Trên 64 study val đầu, Swin đóng băng đọc từ
  cache phải cho đặc trưng khớp `swin_global` / `swin_grid` đã lưu (cosine ≥ 0,999 mỗi study, vì đặc trưng cũ lưu
  fp16). Sai → dừng.
- **G1 (script đúng):** nhánh F phải đạt val macro AP ≥ 0,4263 − 0,006 = 0,4203. Thấp hơn nghĩa là script mới
  lệch phép đo cũ → dừng, không đọc nhánh T. (F khác probe cũ ở augmentation và một seed thay vì ba, nên chỉ
  chặn dưới.)
- **G2:** nhánh T phải thật sự đổi trọng số Swin (chuẩn chênh lệch so với pretrained > 0 ở cả stage 1 và
  stage 4). Nhánh F: trọng số Swin giống từng bit lúc đầu.

## 4. Quy tắc đọc, chốt TRƯỚC khi chạy

- **Chỉ số quyết định:** macro AP "có bệnh" (P hoặc U so với N, trên `s = p_pos/(p_pos+p_neg)`), trung bình 14
  finding, val n = 1.808. Cùng hàm `macro_ap` của `encoder_probe_fit.py`. Δ = T − F, bootstrap ghép cặp theo
  study, 1.000 lần, seed 16.
- **ĐI TIẾP** khi Δ ≥ +0,02 **và** cận dưới CI > 0: tinh chỉnh phá được trần. Bước sau (mỗi bước hỏi riêng):
  một seed thứ hai cho T và F, rồi plan đưa Swin đã tinh chỉnh vào Stage 1 (ví dụ khởi tạo Swin của pha 1b từ T,
  hoặc mở Swin trong 1b theo đúng công thức này) và so với C42 theo quy tắc pilot cũ.
- **KHÔNG ĐỦ** khi 0 < Δ < 0,02 hoặc CI chạm 0: ghi kết quả. Tinh chỉnh trên 40.000 study không dời trần đủ để
  đáng train lại Stage 1. Thử với toàn bộ 220.216 study chỉ khi user muốn (khoảng 5–6 lần chi phí).
- **DỪNG** khi Δ ≤ 0.
- Ngưỡng 0,02 giống phép đo encoder: phần MHCAC thêm trên probe chỉ khoảng 0,01, nên chênh nhỏ hơn khó còn lại
  sau khi train lại Stage 1. Ngưỡng này cũng lớn gấp gần 5 lần NB 0,0042 của các pilot, nên một seed mỗi nhánh
  là chấp nhận được cho phép đo này (không phải cho một kết luận cuối).
- **Chỉ mô tả:** AUROC "có bệnh"; AP từng finding, nhất là finding hiếm (Pleural Other, Fracture, Enlarged
  Cardiomediastinum, Lung Lesion, Pneumothorax, Consolidation); T so với R1C (0,4271) và với `attn_swin` cũ
  (0,4263) trên cùng study; đường cong holdout theo epoch.

## 5. Code (đã viết 2026-10-31)

- `scripts/swin_finetune_cache.py`: cache 224 uint8 theo mục 2.1. Dùng `MIMIC_CXR_Dataset` với cùng thiết lập như
  `encoder_probe_features.py` (augmentation tắt, `max_aux_views=0`); giá trị uint8 lấy lại chính xác từ `swin_image`
  của dataset (đảo chuẩn hóa), nên ảnh chỉ giải mã một lần. Từ chối split `test`.
- `scripts/swin_finetune_fit.py`:
  - `fit --arm {frozen,finetune} --cache-dir <c> --encprobe-dir ~/encprobe_20261031 --out-dir <o>/{F,T}`: chạy G0
    trước (cả hai nhánh), train, chọn epoch theo holdout, G2 sau train, G1 ở nhánh F. Ghi `val_probs.npy` (chỉ trên
    máy) và `report.json`. Cổng hỏng → thoát mã khác 0.
  - `compare --frozen-dir <o>/F --finetune-dir <o>/T --val-cache <c>/val [--mhcac R1C=<npz>] --report ...`: Δ, CI,
    kết luận `GO_ON` / `NOT_ENOUGH` / `STOP` (= ĐI TIẾP / KHÔNG ĐỦ / DỪNG ở mục 4).
  - Hai chi tiết chốt khi viết code: đầu probe giữ weight decay 0,05 trên mọi tham số như probe cũ (để F so được với
    0,4263); quy tắc "không weight decay cho bias/LayerNorm" áp cho tham số Swin. LayerNorm cuối của Swin nhận LR của
    stage 4. Nhánh T bật drop path 0,1 có sẵn của Swin-Tiny khi train; nhánh F để Swin ở `eval()`.
- `tests/test_swin_finetune.py` (9 test): cache uint8 = ảnh của `medclip_preprocess` trước chuẩn hóa, và chuẩn hóa
  lại cho đúng `medclip_preprocess`; nhóm LR theo tầng phủ mỗi tham số đúng một lần, LR và weight decay đúng; tham số
  lạ bị từ chối; affine đơn vị giữ nguyên ảnh và mọi ma trận có định thức > 0 (không lật); augmentation chỉ chạy khi
  có generator và giữ giá trị trong [0, 1]; split holdout giống phép đo encoder; warmup + cosine; quy tắc đọc.
- Cập nhật `struct/`, CLAUDE.md, README cùng commit.

## 6. Chạy (một card, một lần chạy, một thư mục)

- Chuỗi `~/swinft_chain.sh`, chạy đúng một lần bằng `setsid nohup`, từ một bản clone riêng ở commit đã push.
  Kiểm tra trước: thư mục ra chưa tồn tại, GPU trống (< 1.000 MiB), không có `pretraining.train` nào đang chạy,
  ổ dữ liệu là `ntfs3`, `/home` còn ≥ 10 GB.
- Thứ tự: cache val → cache train → G0 → nhánh F (G1) → nhánh T (G2) → so sánh.
- Cần giữ `~/encprobe_20261031/{train,val}/` (keys, labels, đặc trưng Swin) cho G0, nên chưa dọn thư mục đó.

**Chi phí ước tính:** cache khoảng 20–30 phút (giới hạn bởi giải mã ảnh); F khoảng 5–10 phút; T khoảng 15–25
phút; tổng khoảng 1 giờ GPU. VRAM ước tính dưới 10 GB. Ổ đĩa khoảng 2 GB.

## 7. Giới hạn

- Một seed mỗi nhánh; 40.000 study, không phải 220.216; chỉ ảnh anchor; độ phân giải 224 của MedCLIP.
- Đầu probe khác MHCAC: phép đo cho biết trần của encoder có dời không, chưa cho biết Stage 1 tăng bao nhiêu.
- MedCLIP được pretrain có MIMIC-CXR, split không xác minh được: lợi thế (nếu có) cũng có ở nhánh F.
- Một bộ siêu tham số chốt trước cho T; không dò LR. Nếu kết quả là KHÔNG ĐỦ, chưa loại trừ một công thức khác.
