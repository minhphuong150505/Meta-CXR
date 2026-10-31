# PLAN 2026-10-31: cho BioViL-T thấy ảnh lần chụp trước — có tăng được phân loại không?

**Trạng thái:** ĐÃ CHẠY 2026-10-31 09:50 → 10:08 (`aadb250`). **Kết luận theo quy tắc chốt trước: KHÔNG ĐỦ**
(Δ = B − A +0,0037 [+0,0001; +0,0081]). Xem "Execution report" cuối file. Chỉ đo trên val; không train Stage 1, không
sửa YAML, không chấm test.

## 1. Vì sao

- BioViL-T là encoder **hai ảnh**: ảnh hiện tại và ảnh lần chụp trước của cùng bệnh nhân
  (`MultiImageEncoder.forward(current_image, previous_image=None)`). Đầu ra 512 chiều mỗi ô = 256 chiều "tĩnh" của
  ảnh hiện tại + 256 chiều "khác biệt" do transformer thời gian tính từ hai ảnh.
- **Pipeline không bao giờ truyền ảnh trước** (`ImageModel.forward` chỉ nhận một ảnh; kiểm tra bằng grep 2026-10-31).
  Khi thiếu ảnh trước, 256 chiều "khác biệt" là **một hằng số học được** (`missing_previous_emb`, lặp lại ở mọi ô và
  mọi ảnh). Vậy một nửa đầu ra BioViL-T hiện không mang thông tin gì.
- Phép đo encoder 2026-10-31 (đặc trưng đã xóa, chỉ còn số): BioViL-T đọc đúng (một ảnh) 0,4039; Swin 0,4263;
  Swin + BioViL-T 0,4315 (+0,0052 [+0,0010; +0,0098]).
- Câu hỏi duy nhất: cùng dữ liệu, cùng probe, thêm ảnh lần chụp trước vào BioViL-T có tăng macro AP của
  Swin + BioViL-T thêm ≥ 0,02 không?

## 2. Thiết kế

### 2.1 Ảnh lần chụp trước (`prior_frontal_table`)

- Lần chụp trước = study **sớm hơn hẳn** (theo `StudyDate` + `StudyTime` trong metadata, `merge_asof` với
  `allow_exact_matches=False`, giống `training/dataio/study_context.py`) gần nhất của **cùng bệnh nhân**, trong số các
  study có ít nhất một ảnh thẳng (PA/AP). Trong study đó lấy PA trước AP, rồi `dicom_id` nhỏ nhất. Hai study sớm hơn cùng một thời điểm: lấy `study_id` lớn hơn.
- Split chia theo bệnh nhân, nên lần chụp trước luôn nằm cùng split với study hiện tại; val chỉ đọc ảnh val.
- Chỉ dùng khi ảnh anchor hiện tại là ảnh thẳng (PA/AP); anchor nghiêng thì coi như không có ảnh trước (BioViL-T học
  trên cặp ảnh thẳng).
- Ảnh trước qua đúng phép biến đổi BioViL của dataset (`geometric_trans` + `optical_trans`, augmentation tắt).
- Nhãn của lần chụp trước không được đọc ở đâu.

### 2.2 Đặc trưng (`scripts/biovil_prior_features.py`)

- Cùng study như phép đo encoder: 40.000 study train (`default_rng(16)`), toàn bộ 1.808 study val; chỉ ảnh anchor.
- Một lượt ghi: `swin_{global,grid}`; `biovil_{global,grid}` (không ảnh trước, đúng như phép đo encoder);
  `biovilprior_{global,grid}` (có ảnh trước khi có, không thì bằng `biovil`); `has_prior.npy`, `prior_days.npy`,
  `labels.npy`, `keys.npy`.
- Kiểm tra lúc chạy: (a) phép biến đổi dùng cho ảnh trước tái tạo đúng từng bit tensor `image` của dataset trên 4 study
  đầu; (b) ở study có ảnh trước, 256 chiều tĩnh gần như không đổi (cosine ≥ 0,999) và 256 chiều khác biệt có đổi.

### 2.3 Probe (`scripts/biovil_prior_fit.py`)

`fit_attention` của `encoder_probe_fit.py` không đổi (3 seed, 12 epoch, chọn epoch trên holdout 10%, cùng split), các
tổ hợp: `swin`, `biovil`, `biovilprior`, `swin+biovil` (A), `swin+biovilprior` (B).

## 3. Cổng

- **G0:** macro AP tính lại từ npz val của R1C = 0,4271 và nhãn khớp sau khi ghép theo key (như phép đo encoder).
- **G1 (tái lập phép đo cũ):** `attn_swin`, `attn_biovil`, `attn_swin+biovil` mỗi cái lệch ≤ 0,003 so với 0,4263 /
  0,4039 / 0,4315. Sai → dừng, không đọc B.
- Kiểm tra (a), (b) ở mục 2.2 sai → dừng.

## 4. Quy tắc đọc, chốt TRƯỚC khi chạy

- **Chỉ số quyết định:** macro AP "có bệnh" (P hoặc U so với N, trên `p_pos/(p_pos+p_neg)`), 14 finding, toàn bộ val
  n = 1.808. Δ = B − A, bootstrap ghép cặp theo study, 1.000 lần, seed 16.
- **ĐI TIẾP** (`GO_ON`) khi Δ ≥ +0,02 và cận dưới CI > 0. Bước sau, hỏi riêng: plan đưa ảnh trước vào Stage 1 (cần sửa
  dataset và luồng BioViL, train lại 1a + 1b).
- **KHÔNG ĐỦ** (`NOT_ENOUGH`) khi 0 < Δ < 0,02 hoặc CI chạm 0.
- **DỪNG** (`STOP`) khi Δ ≤ 0.
- **Chỉ mô tả:** cùng Δ trên nhóm val có ảnh trước; `attn_biovilprior` − `attn_biovil`; B − `attn_swin`; tỉ lệ study có
  ảnh trước; AP từng finding.
- Ngưỡng 0,02 giống phép đo encoder và tinh chỉnh Swin. Ảnh trước chỉ có ở một phần study, nên Δ trên toàn val bị pha
  loãng; quy tắc vẫn đọc trên toàn val vì đó là thứ Stage 1 được chấm.

## 5. Chạy

`~/biovilprior_chain.sh` một lần, bản clone riêng ở commit đã push; kiểm tra GPU trống, không có `pretraining.train`,
ổ dữ liệu `ntfs3`, thư mục ra `~/biovilprior_20261031` chưa có. Thứ tự: đặc trưng val → đặc trưng train → probe.
Ước tính: đặc trưng khoảng 20–30 phút (giải mã thêm ảnh trước), probe khoảng 20 phút; khoảng 18 GB ổ đĩa (đặc trưng
fp16, chỉ trên máy train).

## 5b. Code (viết 2026-10-31)

`scripts/biovil_prior_features.py` (`prior_frontal_table`, `prior_lookup`, kiểm tra (a)/(b) lúc chạy),
`scripts/biovil_prior_fit.py` (G0, G1, quy tắc, `verdict`), `tests/test_biovil_prior_probe.py` (3 test: chỉ lấy study
sớm hơn hẳn của cùng bệnh nhân có ảnh thẳng, PA trước AP, hòa thời điểm, anchor nghiêng không có ảnh trước, quy tắc đọc,
G1).

## 6. Giới hạn

- Probe, không phải MHCAC; 40.000 study; một lần chạy (3 seed của đầu probe).
- Ảnh trước là lần chụp gần nhất, bất kể cách bao lâu; chỉ ảnh thẳng.
- Theo bài báo BioViL-T (chưa kiểm tra lại trong phiên này), lợi ích của ảnh trước chủ yếu ở bài toán tiến triển; với
  phân loại tĩnh có thể nhỏ.
- Bài META-CXR không dùng ảnh trước (chưa kiểm tra): nếu đi tiếp thì đó là thay đổi đầu vào so với bài báo.

## Execution report (2026-10-31)

- **Chạy:** `~/biovilprior_chain.sh` một lần, bản clone `~/Meta-CXR-biovilprior` ở `aadb250`; bộ test của bản clone qua
  hết trên máy train. Đặc trưng val 26 s, train 13 phút, probe 4 phút. Đầu ra `~/biovilprior_20261031/` (7,0 GB, chỉ trên
  máy train).
- **Ảnh trước:** val 1.246 / 1.808 study (68,9%), train 27.046 / 40.000 (67,6%). Khoảng cách trung vị 2 ngày (val), 3 ngày
  (train): phần lớn lần chụp trước nằm trong cùng đợt nằm viện.
- **Kiểm tra (a) qua** (val và train). **(b) qua:** nửa tĩnh cosine thấp nhất 0,99997; nửa khác biệt đổi trung bình
  0,775 (giá trị tuyệt đối mỗi phần tử).
- **G0 qua:** R1C 0,4271. **G1 qua:** `attn_swin` 0,42629, `attn_biovil` 0,40390, `attn_swin+biovil` 0,43151 (ghi trước
  0,4263 / 0,4039 / 0,4315).

| probe | val macro AP | AUROC "có bệnh" | macro AP, nhóm có ảnh trước (n = 1.246) |
|---|---:|---:|---:|
| Swin | 0,4263 | 0,7945 | 0,4255 |
| BioViL-T, không ảnh trước | 0,4039 | 0,7823 | 0,4037 |
| BioViL-T, có ảnh trước | 0,4103 | 0,7871 | 0,4124 |
| A: Swin + BioViL-T, không ảnh trước | 0,4315 | 0,8004 | 0,4299 |
| **B: Swin + BioViL-T, có ảnh trước** | **0,4352** | **0,8043** | 0,4339 |
| R1C (mô hình đang báo cáo) | 0,4271 | 0,8017 | — |

- **Quy tắc:** B − A **+0,0037 [+0,0001; +0,0081] → KHÔNG ĐỦ** (dưới +0,02; cũng dưới NB 0,0042 của các pilot).
- **Mô tả:** cùng Δ trên nhóm có ảnh trước +0,0040 [+0,0002; +0,0082] — không lớn hơn toàn val, nên mức tăng nhỏ không
  phải do bị pha loãng bởi study không có ảnh trước. BioViL-T có ảnh trước − không: +0,0064 [+0,0004; +0,0121].
  B − Swin +0,0089 [+0,0044; +0,0149].
- **AP từng finding (A → B):** tăng ở Pneumonia 0,427 → 0,447, Pneumothorax 0,374 → 0,397, Fracture 0,021 → 0,032, Lung
  Opacity 0,477 → 0,485; giảm ở Consolidation 0,253 → 0,245, Pleural Effusion 0,778 → 0,773. Chỉ mô tả.
- **Đọc kết quả:** ảnh trước có mang thêm một chút thông tin (dương ở mọi so sánh, cận dưới vừa trên 0), nhưng quá nhỏ để
  đáng sửa dataset và train lại 1a + 1b. Cả bộ BioViL-T đúng + ảnh trước thêm vào Swin chỉ được +0,0089 trong probe.
  Không có bước tiếp theo tự động.
