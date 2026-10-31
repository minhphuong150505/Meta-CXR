# PLAN 2026-10-31: cho BioViL-T thấy ảnh lần chụp trước — có tăng được phân loại không?

**Trạng thái:** user đồng ý thử ("bạn cứ thử đi", 2026-10-31). Quy tắc đọc dưới đây được commit TRƯỚC khi chạy.
Chỉ đo trên val; không train Stage 1, không sửa YAML, không chấm test.

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
