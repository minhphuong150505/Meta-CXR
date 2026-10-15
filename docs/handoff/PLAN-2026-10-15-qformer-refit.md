# PLAN 2026-10-15 — Sửa soft token sụp: train lại META-Former trên checkpoint 1c đóng băng

Trạng thái: **smoke chạy 2026-10-15, cổng smoke không đạt (PR 3,15 < 4); chờ user quyết định.** Quy tắc dưới đây chốt và commit
trước khi có bất kỳ kết quả nào. Stage 1 gốc: `run_20261005_paper/phase1c` (checkpoint của mọi
số Stage 1 đã báo cáo và của Stage 2 P2 + LR 3e-4).

## Vì sao (đã đo 2026-10-11, `scripts/diagnose_qformer_collapse.py`, 200 study val)

| | khởi tạo BLIP-2 | sau 1a | sau 1c (Stage 2 đang dùng) |
|---|---:|---:|---:|
| cosine giữa 32 token của một study | 0,445 | 0,9989 | 0,9997 |
| số hướng hiệu dụng | 2,94 | 1,002 | 1,001 |
| độ giống bản đồ cross-attention giữa query | 0,27 | 1,000 | 0,906 |

- Ở hai tầng cross-attention đầu, cả 32 query dồn vào token CLS của PubMedCLIP (vị trí 196, 100%
  study); trạng thái trùng nhau và các tầng sau (dùng chung trọng số) không tách lại được. Không mục
  tiêu nào của 1a đòi query khác nhau (ITC lấy max theo query, LM cần một prefix, ITM lấy trung bình
  và ở mức ngẫu nhiên suốt 1a).
- Đầu vào Q-Former không chuẩn hóa: sau StreamAdapter chỉ BioViL có LayerNorm; PubMedCLIP và Swin
  vào thẳng. 1b làm norm token PubMedCLIP 19,6 → 1.713, Swin 12 → 413 khi Q-Former đã đóng băng.
- Hậu quả: một vector giữ bệnh phổ biến, mất bệnh khu trú/hiếm (Fracture AUROC 0,492 so với MHCAC 0,684).

## Cách sửa (code trong commit này, mọi cờ mặc định tắt)

1. `model.qformer_input_norm: per_stream` — một LayerNorm mỗi luồng cho bản sao token mà Q-Former
   đọc (`vision_encoders/qformer_inputs.py::QformerInputNorm`); MHCAC vẫn đọc token gốc.
2. `model.loss.lambda_query_diversity` (1,0) — `query_diversity_loss` = `log(Q/PR)/log(Q)` của số
   hướng hiệu dụng (participation ratio) phần thay đổi theo ảnh của 32 query (mỗi query trừ trung
   bình của nó trong batch).
3. Bỏ ITM.
4. Pha `phase1q` (`pretraining/configs/mimic_cxr_full.yaml`, không nằm trong chuỗi mặc định): nạp
   checkpoint 1c, **bỏ qua** phía Q-Former (`init_from.skip` → khởi tạo BLIP-2 như 1a) và đầu MPC
   (`init_from.ignore: [mpc_heads]`: `lambda_mpc 0` nên model không dựng chúng; thêm sau smoke tích
   hợp, xem báo cáo thực thi), chỉ train
   Q-Former + query + chiếu ITC + norm; mọi thứ khác đóng băng; đầu vào đa góc chụp đúng như lúc
   dựng record Stage 2 (`eval_mode_modules: [view_fusion]` giữ dropout/bỏ ảnh phụ của view fusion ở
   chế độ eval); không cache đặc trưng; GradCache batch 128 dùng lại token ảnh của lượt no-grad
   (phía trước Q-Former đóng băng); 2 epoch, chọn theo val loss.
5. `scripts/merge_qformer_refit.py` — checkpoint của 1q KHÔNG chứa các block encoder mà 1c đã mở
   khóa, nên mọi nơi dùng nó phải đọc file gộp: checkpoint 1c với phía Q-Former thay bằng bản refit;
   từ chối nếu bất kỳ tensor nào khác (MHCAC, adapter, fusion, buffer BatchNorm) lệch dù 1 bit.
6. Stage 2: `--stage1-options model.qformer_input_norm=per_stream` (cả hai CLI; vào danh tính cache
   chỉ khi có); báo lỗi nếu checkpoint và config không khớp về norm; `generate_stage2_reports.py
   --soft-token-ablation permute_across` (đổi soft token giữa các study, giữ cue và bối cảnh).

### Hai chỗ đổi so với bản trình bày cho user (đổi TRƯỚC khi chạy, lý do ghi ở đây)

- **Dạng loss.** Bản trình bày: trung bình cosine² giữa các cặp query sau khi trừ trung bình theo
  batch. Dạng đó bị lách được: một query mang thông tin ảnh + 31 query gần như hằng số — phần còn lại
  của 31 query chỉ là nhiễu dropout, hướng ngẫu nhiên, cosine ≈ 0, loss ≈ nhỏ nhất trong khi vẫn sụp.
  Dạng participation ratio không bị lách theo cách đó (query hằng số không góp phương sai). Cả hai
  kiểu lách được ghim bằng test (`tests/test_qformer_inputs.py`).
- **Cổng sụp.** Bản trình bày: "cosine giữa 32 token ≤ 0,8". Cosine thô tính cả thành phần hằng
  chung (vd. bias của LayerNorm cuối) là sụp, nên có thể đánh trượt một Q-Former mang thông tin đa
  dạng — đúng lý do loss phải trừ trung bình. Thay bằng số hướng hiệu dụng sau khi trừ trung bình
  theo batch ≥ 4; cosine thô vẫn được báo cáo bên cạnh. Cổng bản đồ attention giữ nguyên.

## Cổng và quy tắc (chốt trước)

**Cổng sụp** (`diagnose_qformer_collapse.py`, 200 study val, file gộp): `xattn_mean_query_map_cos`
≤ 0,6 **và** `final_pr_batch_centred` ≥ 4 **và** cổng ITC (256 cặp val) R@5 trên mức ngẫu nhiên cả
hai chiều, ở epoch được chọn. Mốc: đo cùng script trên checkpoint 1c và khởi tạo BLIP-2.

**Bất biến Stage 1:** `merge_qformer_refit.py` thành công (mọi tensor chung ngoài phía Q-Former trùng
từng bit với 1c). Không có thì không có gì phía sau.

**Cổng thông tin:** probe tuyến tính cùng giao thức (`probe_image_features.py`, 60.000 study train
đầu theo thứ tự dataset, ảnh sạch, cùng ảnh cho hai bên; L2 chọn trên 10% giữ lại của train), AUROC
trung bình trên phần giữ lại: token mới ≥ token cũ − 0,01. Test chỉ báo cáo.

**Pilot Stage 2:** A = soft token 1c, B = soft token refit; cùng 10.000 study train, công thức P2 +
`--lora-lr 3e-4`, 1 epoch, cùng seed; cue MHCAC giống hệt (cùng MHCAC). Chấm cả 1.360 study val,
beam 4 / lp 2, giao thức bài báo + CheXbert, bootstrap ghép cặp 2.000 lần seed 16. **Nhận B** nếu
B − A: BERTScore có cận dưới > 0 **và** CheXbert micro F1-14 không giảm có ý nghĩa (cận trên > 0).
Kèm theo (chỉ báo cáo): mỗi nhánh, greedy 500 study val với soft token thật và soft token của study
khác (`--soft-token-ablation permute_across`) — mức tụt đo việc adapter đọc kênh ảnh.

Nhận B mới hỏi user về train Stage 2 đầy đủ 2 epoch (~47,7 h) + test. Không nhận: giữ mô hình đã báo
cáo, ghi kết quả âm.

## Chạy (máy train, `~/Meta-CXR-v3`, manifest v3)

```bash
cd ~/Meta-CXR-v3 && git pull
EXPECT_REV=<commit> setsid nohup bash ~/qrefit_pipeline.sh > ~/qrefit_20261015.log 2>&1 < /dev/null &
```

`~/qrefit_pipeline.sh` (+ `~/qrefit_gate.py`, `~/qrefit_build_records.py`) → `~/qrefit_20261015/`:

| bước | việc | ước tính |
|---|---|---|
| 0 | chẩn đoán mốc: checkpoint 1c và khởi tạo BLIP-2 | ~10 phút |
| 1 | smoke 300 update (`run.truncate_train=38400`): C = công thức 1a trên cùng đặc trưng 1c đóng băng (không norm, không loss mới, có ITM), M = bản sửa; gộp + chẩn đoán cả hai; **cổng sụp trên M** (C chỉ để đối chiếu) | ~1,5 h |
| 2 | refit 2 epoch → gộp → chẩn đoán → **cổng sụp** | ~6–8 h (đo ở smoke) |
| 3 | cache record (60.000 train + test, ảnh sạch) cho token cũ và mới → probe → **cổng thông tin** | ~1 h |
| 4 | pilot A/B: train, beam cả val, greedy 500 thật/hoán đổi | ~6 h |
| 5 | chấm: B − A (beam), độ tụt khi hoán đổi (mỗi nhánh) → `QREFIT_PIPELINE_DONE` | ~20 phút |

## Điều kiện dừng

GPU không rảnh, ổ dữ liệu không phải `ntfs3`, thư mục output đã tồn tại, checkout không đúng commit,
log không xác nhận đã bỏ qua phía Q-Former khi nạp, bước nào trả mã khác 0, gộp checkpoint bị từ
chối, hoặc một cổng không đạt (`SMOKE_GATE_FAILED` / `REFIT_GATE_FAILED` / `INFO_GATE_FAILED`, mã 3).
Không thử phương án dự phòng (che token CLS/pooled khỏi K/V, query kiểu DETR) khi chưa hỏi user.

## Giới hạn biết trước

- Q-Former được train sau MHCAC, phần còn lại đóng băng — khác thứ tự của bài báo (1a trước MHCAC).
- Lỗi token BioViL bị trộn (`TONG_HOP_2026-10-11.md` 11.6) không sửa ở đây.
- Pilot 10.000 study, 1 epoch yếu hơn run đầy đủ; pilot âm không loại trừ hiệu ứng ở quy mô đầy đủ.
- Không phải giao thức bài báo: prompt có indication và FINDINGS lần chụp trước.

## Báo cáo thực thi

### Smoke tích hợp (2 update, `~/qrefit_micro.sh`, không phải kết quả) — 2026-10-15

- Lần 1 (12:47, `4b430ff`): dừng sau 15 s khi nạp checkpoint 1c — 18 tensor `mpc_heads.*` mà model
  `phase1q` (`lambda_mpc 0`) không dựng. Sửa bằng `init_from.ignore: [mpc_heads]` (báo lỗi nếu model
  có module đó); file gộp giữ đầu MPC của 1c.
- Lần 2 (13:00, `6c72f97`): mọi bước tích hợp chạy được — nạp checkpoint bỏ qua 364 tensor (346 phía
  Q-Former + 18 đầu MPC); trainable chỉ phía Q-Former (Qformer 186,28M, norm 0,01M, query, chiếu ITC);
  GradCache tự chọn "dùng lại token ảnh"; val loss gồm `loss_query_div`; cổng ITC chạy; gộp: 346
  tensor thay, 6 tensor norm thêm, 445 tensor chung trùng từng bit với 1c, 82 tensor chỉ có ở 1c
  (block encoder 1c mở + đầu MPC) giữ nguyên; chẩn đoán có `--options` chạy, không có thì bị từ
  chối; dựng record với `--stage1-options` chạy (missing=758, unexpected=0). Số đo 2 update không
  phải kết quả và không được dùng. Script smoke tự hỏng ở một dòng in tóm tắt của chính nó (không
  thuộc pipeline).

### Pipeline — khởi chạy 2026-10-15 13:05:50, `6c72f97`, một lần

`EXPECT_REV=6c72f97 setsid nohup bash ~/qrefit_pipeline.sh > ~/qrefit_20261015.log` → `~/qrefit_20261015/`.
Thứ tự smoke đổi so với bảng: M trước, C sau và C không thể dừng pipeline (chỉ để đối chiếu).

- **Lần 1 dừng 13:07:39 (`ABORT: train smoke_M rc=137`)**: OOM killer của kernel 1 phút sau khi train bắt
  đầu (`journalctl -k`: global_oom, task=python). Batch 128 KHÔNG có cache = ảnh thật của ba encoder và ảnh
  phụ, ~0,9 GB/batch; 8 worker × prefetch 2 ≈ 15 GB shared memory, mỗi worker ~1,6 GB RSS, trên máy 30 GB.
  Smoke tích hợp (2 batch) không thấy được. 5 worker DataLoader mồ côi (ppid 1) còn giữ ~7 GB + shm sau khi
  tiến trình chính chết; đã dừng theo PID. Sửa: `phase1q.run.num_workers: 4`; pipeline dọn worker mồ côi khi
  một bước train hỏng.
- Hai chẩn đoán mốc của lần 1 đã xong (200 study val, chỉ số tổng hợp): checkpoint 1c — `final_within_cos`
  0,9997, `final_pr_batch_centred` 1,0010, `xattn_mean_query_map_cos` 0,7974, `query_diversity_loss` 0,9997;
  khởi tạo BLIP-2 — 0,4542 / 4,6002 / 0,2807 / 0,5634. Cổng (PR ≥ 4, bản đồ ≤ 0,6) nằm giữa hai mốc này.
- **Lần 2 (13:14:15, `2a3bdac`): smoke chạy hết, cổng smoke KHÔNG đạt → `SMOKE_GATE_FAILED` 14:32:44, pipeline
  dừng như đã chốt.** RAM ổn định ~8/30 GB trống với 4 worker. Smoke M 300 update 25:01 (5,0 s/update), smoke
  C 45:38 (9,1 s/update, có ITM). Gộp cả hai: 346 tensor thay, 445 tensor chung trùng từng bit với 1c.

| 200 study val | 1c | khởi tạo BLIP-2 | C (công thức 1a, 300 update) | M (bản sửa, 300 update) | cổng |
|---|---:|---:|---:|---:|---|
| `final_pr_batch_centred` | 1,0010 | 4,2907 | 1,1889 | **3,1457** | ≥ 4 — M trượt |
| `xattn_mean_query_map_cos` | 0,7974 | 0,2837 | 0,5296 | 0,5305 | ≤ 0,6 — đạt |
| `query_diversity_loss` | 0,9997 | 0,5835 | 0,9514 | 0,6721 | — |
| `final_within_cos` (thô) | 0,9997 | 0,4561 | 0,8635 | 0,9024 | chỉ báo cáo |
| ITC R@5 i2t / t2i (ngẫu nhiên 0,02) | — | — | 0,2461 / 0,2773 | 0,2695 / 0,2578 | đạt |
| ITC `delta_nats` | — | — | 1,5099 | 1,1428 | — |

  Đọc: không có bản sửa (C), thông tin ảnh của 32 query sụp lại về ~1,2 hướng ngay trong 300 update trên cùng đặc
  trưng; có bản sửa (M) giữ 3,15 hướng, căn chỉnh ITC không kém. Bản đồ attention của hai nhánh như nhau ở
  300 update. Cả 300 update nằm trong warm-up LR (500 update). Khởi tạo BLIP-2 đo 4,60 ở lần 1 và 4,29 ở lần 2:
  với `--checkpoint none` phần chiếu chung là khởi tạo ngẫu nhiên, nên mốc này dao động. Chờ user quyết định
  bước tiếp (chạy refit đầy đủ với cổng cuối giữ nguyên, tăng trọng số loss rồi smoke lại, hay phương án dự phòng).
