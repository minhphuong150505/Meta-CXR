# PLAN 2026-10-15 — Sửa soft token sụp: train lại META-Former trên checkpoint 1c đóng băng

Trạng thái: **lần 1 (15/10) trượt cổng cuối (PR 2,32 < 4; loss đa dạng bị lách qua dropout). Lần 2
(16/10, đường đi của query không còn nhiễu chỉ có lúc train): kiểm tra vi mô, smoke, cổng refit (PR theo
batch 26,9) và cổng thông tin (probe 0,7845 so với 0,7669) đạt, nhưng pilot Stage 2 KHÔNG nhận B (BERTScore
B − A +0,0007 [−0,0057; +0,0067]) → giữ mô hình đã báo cáo, ghi kết quả âm. Hoán đổi soft token làm cả hai
adapter tụt rõ: Stage 2 có đọc kênh ảnh, kể cả token đã sụp.** Quy tắc dưới đây chốt và commit
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
  với `--checkpoint none` phần chiếu chung là khởi tạo ngẫu nhiên, nên mốc này dao động.
- **Quyết định của user (2026-10-15, phương án A): bỏ qua cổng smoke, chạy refit đầy đủ.** Đây là đổi quy
  tắc SAU khi thấy kết quả smoke, ghi rõ ở đây. Lý do trình cho user: cả 300 update nằm trong warm-up LR, và
  đối chứng C cho thấy bản sửa đang chặn sụp (3,15 so với 1,19). **Cổng cuối không đổi:** sau 2 epoch, ở epoch
  được chọn, `final_pr_batch_centred` ≥ 4 và `xattn_mean_query_map_cos` ≤ 0,6 và ITC R@5 trên mức ngẫu nhiên;
  không đạt thì dừng trước probe và pilot. Chạy tiếp bằng `START_AT=refit` trong cùng `~/qrefit_20261015/`,
  log `~/qrefit_20261015_refit.log`.

### Refit 2 epoch — 14:53:25 → 19:48:49, `cebd51b`: `REFIT_GATE_FAILED`, pipeline dừng đúng quy tắc

- Train 2 epoch: `Training time 4:53:39`, ~4,99 s/update, không OOM; `journalctl -k` từ 14:50 không có
  OOM/Xid/lỗi NTFS. Epoch được chọn: 1 (epoch cuối). Gộp: 346 tensor thay, 6 tensor norm thêm, 445 tensor
  chung trùng từng bit với 1c, 82 tensor chỉ có ở 1c giữ nguyên.
- Val (chế độ eval), epoch 0 / 1: ITC 3,0408 / 2,9134; LM 2,4321 / 2,3738; `loss_query_div` 0,7636 /
  0,7666. Trung bình lúc train, epoch 0 / 1: ITC 3,6449 / 2,9509; LM 2,8063 / 2,3982; `loss_query_div`
  **0,0506 / 0,0142** (update đầu tiên: 0,7247).
- Cổng ITC epoch 1 (256 cặp val): `loss_itc` 2,8833 so với ngẫu nhiên 5,5452 (`delta_nats` +2,6619), hạng
  trung bình của cặp đúng 9,26 / 10,23 (ngẫu nhiên 127,5), R@1 0,3008 / 0,2930, R@5 0,5781 / 0,5625.

| 200 study val | 1c | M 300 update | refit 2 epoch | cổng |
|---|---:|---:|---:|---|
| `final_pr_batch_centred` | 1,0010 | 3,1457 | **2,3166** | ≥ 4 — **trượt** |
| `xattn_mean_query_map_cos` | 0,7974 | 0,5305 | 0,5065 | ≤ 0,6 — đạt |
| ITC R@5 i2t / t2i | ≈0,17 (ghi 2026-10-11) | 0,2695 / 0,2578 | 0,5781 / 0,5625 | trên ngẫu nhiên — đạt |
| `query_diversity_loss` | 0,9997 | 0,6721 | 0,7579 | — |
| `final_within_cos` (thô) | 0,9997 | 0,9024 | 0,9405 | — |

- Theo tầng (refit): cosine thô giữa 32 query tăng dần tới 0,698 sau 9 tầng, nhảy lên 0,931 sau tầng thứ
  10 (tầng chỉ có self-attention giữa các query) và 0,940 ở đầu ra; cross-attention đầu có `query_map_cos`
  0,161, cross-attention cuối 0,921. CLS của PubMedCLIP (index 196) là token được nhìn nhiều nhất ở
  cross-attention đầu cho 23% study (M 300 update: 37%; 1a cũ: 100%). Sụp không còn nằm ở đầu vào mà ở các
  tầng cuối.
- **Vì sao loss đa dạng không giữ được: nó bị "lách" qua dropout.** Loss tính trên trạng thái query lúc
  train, khi Q-Former còn dropout 0,1. Probe một lần (`~/qrefit_dropout_probe.py` +
  `~/qrefit_dropout_run.sh`, không vào git; cùng 200 study val, chỉ số tổng hợp,
  `~/qrefit_20261015/diag/dprobe_*.json`):

| 200 study val | 1c | C 300 update | M 300 update | refit 2 epoch |
|---|---:|---:|---:|---:|
| PR_bc, chế độ eval (không dropout) | 1,00 | 1,19 | 3,15 | 2,32 |
| PR_bc, chỉ bật dropout của Q-Former | 1,28 | 1,56 | 27,40 | 30,11 |
| loss đa dạng như lúc train (cụm 16 study, có dropout) | 0,894 | 0,831 | 0,042 | 0,018 |
| độ lớn nhiễu dropout / phần phụ thuộc ảnh | 0,36 | 0,41 | 2,77 | 3,90 |

  Không có loss đa dạng (1c, C), nhiễu dropout chỉ bằng ~0,4 lần phần phụ thuộc ảnh và gần như không đổi
  PR. Có loss (M), mô hình học cách khuếch đại nhiễu dropout lên 2,8 lần (300 update) rồi 3,9 lần (2
  epoch) phần phụ thuộc ảnh. Nhiễu gần đẳng hướng (PR của riêng nhiễu 23–26 trên 32), nên lúc train loss
  đọc thấy 27–30 hướng và về ~0,01–0,05 — không còn áp lực lên hình học thật (chế độ eval, cũng là thứ
  Stage 2 nhận), vốn giảm 3,15 → 2,32 trong khi ITC/LM train tiếp. Chế độ train đầy đủ của mô hình (thêm
  che token đầu vào và dropout ở module khác) cho cùng kết quả (30,43), nên nguyên nhân là dropout của
  Q-Former. Độ lệch giữa loss lúc train (0,014) và loss val (0,77) có sẵn trong log từ epoch 0; tôi đã
  không đối chiếu hai số này. Đây là lỗi thiết kế loss (tính trên trạng thái có dropout), không phải lỗi
  dữ liệu hay phần cứng.
- **Kết luận.** Cổng cuối trượt; pipeline dừng trước cổng thông tin và pilot như đã chốt, không chạy gì
  thêm. Lần refit này không kiểm được thiết kế dự định, vì sau vài trăm update loss đa dạng hết tác dụng.
  Phần còn lại của bản sửa (norm từng luồng, khởi tạo lại BLIP-2, bỏ ITM, 2 epoch) vẫn cho PR 2,32 so với
  1,00 của 1c, và ITC R@5 0,58 so với ≈0,17. Chờ user chọn bước tiếp.

## Lần 2 — đường đi của query không còn nhiễu chỉ có lúc train (user chọn phương án A, 2026-10-15)

**Thay đổi (chốt TRƯỚC khi chạy):**

- Khóa mới `model.qformer_dropout` (vắng = dropout 0,1 của BERT, giống từng bit):
  `set_dropout_probability` đặt `p` cho mọi lớp dropout trong Q-Former ngay sau khi dựng; module vẫn ở
  chế độ train, vì gradient checkpointing của `BertEncoder` chỉ bật khi `self.training`
  (`Qformer.py:530`). Đưa Q-Former vào `eval_mode_modules` sẽ tắt luôn checkpointing.
- `phase1q`: `qformer_dropout: 0.0` và `feature_mask_ratio: 0.0` (che token đầu vào là nguồn nhiễu chỉ
  có lúc train còn lại trên đường đi của query; view fusion đã ở eval từ lần 1, encoder đóng băng giữ
  eval). Lượt train của query = lượt eval.
- `Blip2Qformer.__init__` từ chối `lambda_query_diversity > 0` nếu thiếu một trong hai khóa
  (`assert_noise_free_query_path`).
- Không đổi: λ = 1, LR, batch 128, 2 epoch, `trainable`, cổng cuối, cổng thông tin, quy tắc pilot.
- Hệ quả phụ: ITC và LM của 1q cũng train không dropout và không che token (hai regulariser này mất).

**Quy tắc:**

0. **Kiểm tra vi mô** (~2 phút GPU): `~/qrefit_dropout_probe.py` trên file gộp của lần 1 với các khóa
   1q. Ở chế độ train của mô hình, trạng thái query phải trùng chế độ eval: độ lệch tương đối ≤ 1e-6 và PR
   lệch ≤ 1e-4. Không đạt thì dừng.
1. **Smoke 600 update** (`run.truncate_train=76800`, qua hết warm-up 500 update; ~55 phút), chỉ bản sửa
   (đối chứng C đã có từ lần 1). **Cổng smoke** = cổng sụp của cổng cuối (`final_pr_batch_centred` ≥ 4,
   `xattn_mean_query_map_cos` ≤ 0,6, ITC R@5 trên ngẫu nhiên cả hai chiều) **và** |loss đa dạng lúc
   train − val| ≤ 0,05. Giá trị train là trung bình các update được log trong 1/4 cuối epoch; giá trị val
   lấy từ `phase_metrics` của epoch đó. Không đạt → `SMOKE_GATE_FAILED`, dừng, ghi kết quả âm. **Không có
   ngoại lệ cho cổng này** (đã nói với user trước khi chạy).
2. Refit 2 epoch → gộp → chẩn đoán → **cổng cuối như lần 1**; độ lệch train/val chỉ báo cáo.
3. Cổng thông tin, pilot A/B, chấm điểm, quy tắc nhận B: như lần 1.

**Chạy:** `~/qrefit2_pipeline.sh` (+ `~/qrefit2_gap.py`, `~/qrefit_dropout_probe.py`, cùng
`~/qrefit_gate.py` và `~/qrefit_build_records.py` của lần 1, không đổi) → `~/qrefit2_20261015/`, log
`~/qrefit2_20261015.log`. Ước tính nếu qua mọi cổng: kiểm tra vi mô 2 phút + smoke ~55 phút + refit ~5 h
+ cổng thông tin ~1 h + pilot ~6 h + chấm ~20 phút ≈ 13 h.

**Giới hạn biết trước thêm:** không còn nhiễu ngẫu nhiên để lách, nhưng PR vẫn không phân biệt biến
thiên phụ thuộc ảnh có ích với vô ích; điều đó chỉ cổng thông tin và pilot đo được.

### Báo cáo thực thi lần 2 — khởi chạy 2026-10-16 07:55:40, `3b69007`, một lần

- User duyệt chạy sáng 16/10 ("ok chạy đi"). Trước khi chạy (07:42): GPU rảnh, RAM trống 28/30 GB,
  `/home` còn 197 GB, ổ dữ liệu `ntfs3`, checkout sạch ở `3b69007`, 5 script trên máy khớp sha256, kernel
  không có lỗi mới từ 14/10 (chỉ các dòng `ntfs3` của lần dọn ổ và lần hết RAM 13:07 ngày 15/10 của lần 1
  với 8 worker).
- Dừng `~/stage2_p2_score.sh` (chạy từ 11/10): vòng chờ dấu `P2FULL_ALL_DONE` trong một log đã ngừng ghi
  từ 11/10, nên không bao giờ chạy; việc chấm điểm nó định làm đã làm riêng trên 2.800 study test.
- Pipeline pid 241259, log `~/qrefit2_20261015.log` (tên thư mục giữ như plan dù chạy ngày 16/10).
- **Kiểm tra vi mô — `MICRO_CHECK_PASSED` lúc 07:56:08** (28 giây). File gộp của lần 1, 200 study val,
  với các khóa 1q: chỉ bật dropout của Q-Former, và chế độ train đầy đủ của mô hình, đều cho độ lệch
  tương đối 0,00 so với chế độ eval và PR_bc 2,3166 = 2,3166 (lần 1, cùng mô hình với dropout 0,1: 30,11).
  Log smoke ghi `Q-Former dropout set to 0.0 on 61 layers` và `feature_mask_ratio` 0,0.

### Smoke 600 update — 07:56:08 → 08:46:40: `SMOKE_GATE_PASSED`

- Train: `Total time 0:46:41`, 4,67 s/update (lần 1 ~4,99), không OOM. Gộp: 346 tensor thay, 6 tensor
  norm thêm, 445 tensor chung trùng từng bit với 1c, 82 tensor chỉ có ở 1c giữ nguyên; epoch 0.
- `loss_query_div` lúc train (giá trị của update được log): 0,890 (update 0) → 0,673 (50) → 0,340 (100)
  → 0,179 (150) → 0,117 (200) → 0,089 (250) → 0,075 (300) → 0,069 (350) → 0,067 (400) → 0,063 (450);
  trung bình 1/4 cuối 0,0596. Val (chế độ eval, `phase_metrics` epoch 0): 0,0690. Lần này loss lúc train
  giảm và loss val giảm theo; lần 1 thì train 0,014 trong khi val 0,77. Val ITC 3,0873, LM 2,5945.
- Cổng ITC (256 cặp val): R@5 0,4570 / 0,4297, `delta_nats` +2,1171.

| 200 study val | 1c | lần 1: M 300 update | lần 1: refit 2 epoch | **lần 2: smoke 600** | cổng |
|---|---:|---:|---:|---:|---|
| `final_pr_batch_centred` | 1,0010 | 3,1457 | 2,3166 | **26,4733** | ≥ 4 — đạt |
| `xattn_mean_query_map_cos` | 0,7974 | 0,5305 | 0,5065 | **0,3643** | ≤ 0,6 — đạt |
| ITC R@5 i2t / t2i | ≈0,17 | 0,2695 / 0,2578 | 0,5781 / 0,5625 | **0,4570 / 0,4297** | trên ngẫu nhiên — đạt |
| \|loss đa dạng train − val\| | — | 0,65 | 0,75 | **0,0094** | ≤ 0,05 — đạt |
| `query_diversity_loss` (eval) | 0,9997 | 0,6721 | 0,7579 | 0,0547 | — |
| `final_within_cos` (thô) | 0,9997 | 0,9024 | 0,9405 | 0,2274 | — |
| `final_pr` (thô, không trừ trung bình) | 1,0006 | 1,1028 | 1,0847 | 12,9776 | — |
| `meanimage_vs_real_token_cos` | 0,8051 | 0,7126 | 0,9032 | 0,5425 | — |
| `pooled_across_studies_cos` | 0,3416 | 0,9753 | 0,9601 | 0,8972 | — |

- Độ lệch của lần 1 ở dòng thứ tư là `~/qrefit2_gap.py` chạy lại trên log lần 1.
- Theo tầng: cosine thô giữa 32 query 0,33–0,37 qua 9 tầng đầu, 0,28–0,33 ở hai tầng sau, 0,227 ở đầu ra;
  không còn bước nhảy ở các tầng cuối như lần 1 (0,698 → 0,931 → 0,940). `query_map_cos` của
  cross-attention đầu 0,255, cuối 0,496 (lần 1: 0,161 và 0,921). Cross-attention đầu chia khối lượng
  BioViL 0,46 / PubMedCLIP 0,34 / Swin 0,19; CLS của PubMedCLIP (index 196) nhận 0,028 khối lượng nhưng
  vẫn là token được nhìn nhiều nhất cho 47% study (lần 1 refit: 23%).
- Đọc kết quả: thay toàn bộ token ảnh bằng token trung bình của study đổi đầu ra nhiều hơn (cosine 0,54,
  1c 0,81), tức query dựa vào cấu trúc từng token chứ không chỉ trung bình. Ngược lại, trung bình của 32
  token giống nhau giữa các study hơn 1c (0,90 so với 0,34; khởi tạo BLIP-2 0,80): phần thay đổi theo ảnh
  nằm ở từng query, không ở trung bình của chúng. PR 26,5 chỉ nói phần thay đổi theo ảnh trải trên nhiều
  hướng; nó có ích hay không là việc của cổng thông tin và pilot.
- Refit 2 epoch bắt đầu ngay sau cổng (08:46:40); ước tính xong ~13:30.

### Refit 2 epoch — 08:46:40 → 13:33:16: `REFIT_GATE_PASSED`

- Train 2 epoch × 1.740 update: `Training time 4:44:49`; epoch 0 2:20:30 (4,85 s/update), epoch 1
  2:21:31 (4,88 s/update). Không OOM, không dòng `Traceback`/`NaN` trong log; `journalctl -k` từ 07:55
  không có Oops/Xid/NVRM/lỗi `ntfs3`/hết RAM. Checkpoint chọn theo val loss: epoch 1, epoch cuối
  (5,3852 → 5,1857). Gộp: như smoke (346 thay, 6 norm thêm, 445 trùng từng bit với 1c, 82 giữ nguyên),
  `refit_epoch` 1.
- RAM: swap tăng dần trong lúc train (0,5 GB lúc 09:52, 4,1/8 GB lúc 10:34), không lần đo nào (10 phút
  một lần) chạm ngưỡng báo động (RAM trống < 1,5 GB hoặc swap > 7,2 GB); train xong còn 0,6 GB swap,
  RAM trống 21,9 GB. Hai epoch gần cùng tốc độ (4,85 / 4,88 s/update).

| | epoch 0, train (TB) | epoch 0, val | epoch 1, train (TB) | epoch 1, val |
|---|---:|---:|---:|---:|
| ITC | 3,5627 | 2,9230 | 2,8494 | 2,7896 |
| LM | 2,7207 | 2,3987 | 2,3254 | 2,3357 |
| `loss_query_div` | 0,0965 | 0,0635 | 0,0517 | 0,0604 |
| tổng | 6,3799 | 5,3852 | 5,2265 | 5,1857 |

- Trung bình train của epoch 0 gồm cả warm-up (loss đa dạng bắt đầu ở 0,895), nên cao hơn val. Val ở
  chế độ eval (`phase_metrics`); lần 1, epoch 0 / 1: ITC 3,0408 / 2,9134, LM 2,4321 / 2,3738, loss đa
  dạng 0,7636 / 0,7666.
- `loss_query_div` lúc train (giá trị của update được log): 0,895 (update 0) → 0,113 (200) → 0,069 (400)
  → 0,058 (600) → 0,055 (800) → 0,053 (1.000) → 0,052 (1.400) → 0,052 (1.600); cả epoch 1 nằm trong
  0,051–0,056. Độ lệch train/val (chỉ báo cáo; trung bình 1/4 cuối epoch): epoch 0 0,0521 / 0,0635 →
  0,0114; epoch 1 0,0510 / 0,0604 → 0,0094 (lần 1: 0,014 / 0,77).
- Cổng ITC (256 cặp val; quét 416 study, 61,5 % có FINDINGS dùng được):

| | epoch 0 | epoch 1 | ngẫu nhiên |
|---|---:|---:|---:|
| `loss_itc` | 3,0240 | 2,7188 | 5,5452 |
| `delta_nats` | +2,5211 | +2,8263 | 0 |
| hạng TB của cặp đúng, i2t / t2i | 10,41 / 11,98 | 7,53 / 9,18 | 127,5 |
| R@1 i2t / t2i | 0,2500 / 0,2344 | 0,3242 / 0,3047 | 0,0039 |
| R@5 i2t / t2i | 0,5586 / 0,5703 | 0,6250 / 0,6094 | 0,0195 |
| nhiệt độ (học) | 0,0889 | 0,0851 | — |

| 200 study val | 1c | lần 1: refit 2 epoch | lần 2: smoke 600 | **lần 2: refit 2 epoch** | cổng |
|---|---:|---:|---:|---:|---|
| `final_pr_batch_centred` | 1,0010 | 2,3166 | 26,4733 | **26,8605** | ≥ 4 — đạt |
| `xattn_mean_query_map_cos` | 0,7974 | 0,5065 | 0,3643 | **0,3412** | ≤ 0,6 — đạt |
| ITC R@5 i2t / t2i | ≈0,17 | 0,5781 / 0,5625 | 0,4570 / 0,4297 | **0,6250 / 0,6094** | trên ngẫu nhiên — đạt |
| \|loss đa dạng train − val\| | — | 0,75 | 0,0094 | 0,0094 | chỉ báo cáo |
| `query_diversity_loss` (eval) | 0,9997 | 0,7579 | 0,0547 | 0,0505 | — |
| `final_within_cos` (thô) | 0,9997 | 0,9405 | 0,2274 | 0,2722 | — |
| `final_pr` (thô, không trừ trung bình) | 1,0006 | 1,0847 | 12,9776 | 8,0787 | — |
| `meanimage_vs_real_token_cos` | 0,8051 | 0,9032 | 0,5425 | 0,6617 | — |
| `pooled_across_studies_cos` | 0,3416 | 0,9601 | 0,8972 | 0,9485 | — |

- Theo tầng: cosine thô giữa 32 query 0,110 ở đầu vào, 0,27–0,34 qua các tầng, 0,272 ở đầu ra.
  `query_map_cos` của 6 tầng cross-attention: 0,198 / 0,296 / 0,333 / 0,338 / 0,339 / 0,544 (lần 1
  refit: 0,161 ở tầng đầu, 0,921 ở tầng cuối). Cross-attention đầu chia khối lượng BioViL 0,44 /
  PubMedCLIP 0,36 / Swin 0,20; CLS của PubMedCLIP (index 196) nhận 0,029 khối lượng, vẫn là token được
  nhìn nhiều nhất (trung bình qua head và query) cho 47 % study; 57 token khác nhau đứng đầu ở ít nhất
  một study.
- Đọc kết quả. Hai epoch giữ hình học của smoke (PR theo batch 26,5 → 26,9; bản đồ 0,36 → 0,34) trong
  khi ITC tốt lên (R@5 0,46 → 0,63; lần 1 0,58; trước 1c 0,65). Lần 1 thì ngược lại: ITC tốt lên còn PR
  giảm 3,15 → 2,32. So với smoke, PR thô giảm 12,98 → 8,08 và cosine giữa trung bình token của các study
  tăng 0,90 → 0,95: trung bình của 32 token càng giống nhau giữa các study, phần thay đổi theo ảnh nằm ở
  độ lệch của từng query. Probe của cổng thông tin chuẩn hóa từng chiều (z-score trên train) trước khi
  fit, nên phần chung đó tự nó không làm probe thua; cổng đo 14 nhãn đọc tuyến tính được từ trung bình 32
  token đến đâu so với token cũ.
- Pipeline tự sang cổng thông tin lúc 13:33:16: dựng record Stage 1 (60.000 study train + test) cho token
  cũ rồi token mới, ~28 study/s, rồi probe tuyến tính từng bên; ước tính xong ~15:00.

### Cổng thông tin — 13:33:16 → 14:44:53: `INFO_GATE_PASSED`

- Record Stage 1 dựng từ ảnh sạch, cùng ảnh cho hai bên: 60.000 study train đầu theo thứ tự dataset
  (45.806 có FINDINGS hợp lệ) và cả test (2.800 có FINDINGS hợp lệ); ~30 study/s, mỗi bên ~36 phút. Token
  cũ: `run_20261005_paper/phase1c`; token mới: file gộp của refit, dựng với `qformer_input_norm=per_stream`.
  Probe trên trung bình 32 token (chuẩn hóa từng chiều): fit 40.679 study train, chọn L2 trên 4.519 study
  train giữ lại, test 2.780 study có nhãn. `auroc_mean` = trung bình mọi ô (nhãn, lớp) đủ ca, một-đối-còn-lại.

| `auroc_mean` | MHCAC | token cũ | token mới | mới − cũ |
|---|---:|---:|---:|---:|
| train giữ lại (**cổng**: mới ≥ cũ − 0,01) | — | 0,7669 | **0,7845** | +0,0176 — đạt |
| test, mọi ô | 0,7454 | 0,7155 | 0,7325 | +0,0170 |
| test, lớp Negative | 0,7800 | 0,7351 | 0,7519 | +0,0168 |
| test, lớp Positive | 0,7847 | 0,7506 | 0,7626 | +0,0120 |
| test, lớp Uncertain | 0,6162 | 0,6199 | 0,6460 | +0,0261 |
| L2 chọn trên train giữ lại | — | 1e-4 | 1e-2 | |

- Token cũ khớp probe 2026-10-11 (0,713; MHCAC 0,745 trên cùng 2.780 study). Khoảng cách tới MHCAC trên
  test: −0,0299 → −0,0129. Không có CI: probe không ghi xác suất từng study, và cổng chốt trước là so sánh
  điểm trên phần giữ lại; số test chỉ báo cáo.

| test, AUROC lớp Positive | MHCAC | token cũ | token mới |
|---|---:|---:|---:|
| No Finding | 0,8262 | 0,8227 | 0,8242 |
| Enlarged Cardiomediastinum | 0,6495 | 0,6512 | 0,7002 |
| Cardiomegaly | 0,7557 | 0,7506 | 0,7602 |
| Lung Opacity | 0,7275 | 0,7279 | 0,7275 |
| Lung Lesion | 0,7482 | 0,6867 | 0,7026 |
| Edema | 0,8430 | 0,8407 | 0,8387 |
| Consolidation | 0,7659 | 0,7386 | 0,7362 |
| Pneumonia | 0,7255 | 0,6871 | 0,7179 |
| Atelectasis | 0,7590 | 0,7364 | 0,7488 |
| Pneumothorax | 0,8653 | 0,8312 | 0,8175 |
| Pleural Effusion | 0,8987 | 0,8930 | 0,9019 |
| Pleural Other | 0,8330 | 0,6905 | 0,6658 |
| Fracture | 0,6841 | 0,5527 | 0,6317 |
| Support Devices | 0,9046 | 0,8991 | 0,9028 |

- Đọc kết quả: token mới giữ được thông tin 14 nhãn và nhỉnh hơn token cũ (lớp Uncertain tăng nhiều
  nhất). Tăng rõ nhất ở Fracture (+0,079) và Enlarged Cardiomediastinum (+0,049, cao hơn MHCAC); giảm ở
  Pleural Other (−0,025) và Pneumothorax (−0,014). Hai nhãn hiếm (Fracture, Pleural Other) có ít ca
  dương trên test nên dao động lớn. Probe tuyến tính chỉ nói thông tin có trong token; Stage 2 có đọc nó
  hay không là việc của pilot.
- Pilot A bắt đầu ngay sau cổng (14:44:53): dựng record Stage 1 cho 10.000 study train (68 % lúc 14:48).

### Pilot A/B — 14:44:53 → 21:02:02: **B KHÔNG được nhận** (quy tắc chốt trước), `QREFIT2_PIPELINE_DONE`

- Hai nhánh, cùng 10.000 study train đầu, công thức P2 + LR 3e-4, 1 epoch (**476 update** ở batch 2 × 8 cho
  mỗi nhánh), seed 16; cue `cutpoints` từ cùng MHCAC. A: soft token `run_20261005_paper/phase1c`; B: file gộp
  của refit, dựng với `qformer_input_norm=per_stream`. Hai nhánh nạp Stage 1 như nhau (`missing=758
  unexpected=0`). Mỗi nhánh: train ~72 phút (gồm dựng record), beam cả val ~80 phút (4.705 / 4.788 s sinh),
  greedy 500 ~17 phút mỗi lượt; 0 lỗi sinh, VRAM đỉnh 8,9 GiB. Chấm điểm 3,7 phút. Pilot 6 h 17 phút; cả
  pipeline lần 2 07:55:40 → 21:02:02 (13 h 06 phút), 0 lỗi kernel, báo động RAM không kích hoạt.
- Val loss (300 study val, teacher-forced): A 1,18770, B 1,18495 (−0,0028, không có CI). Train loss
  1,21095 / 1,21145.

Beam 4 / lp 2 trên cả 1.360 study val: cùng study, cùng thứ tự, reference trùng từng dòng; 327 / 1.360 báo
cáo giống hệt nhau giữa A và B. Bootstrap ghép cặp 2.000 lần, seed 16.

| | A (token cũ) | B (token refit) | B − A, CI 95% |
|---|---:|---:|---|
| **BERTScore** (quy tắc: cận dưới > 0) | 0,4645 | 0,4652 | **+0,0007 [−0,0057; +0,0067] — không đạt** |
| **CheXbert micro F1-14** (quy tắc: cận trên > 0) | 0,4804 | 0,4898 | +0,0094 [−0,0047; +0,0234] — đạt |
| CheXbert macro F1-14 | 0,3699 | 0,3660 | −0,0040 [−0,0231; +0,0145] |
| CheXbert micro F1-5 | 0,5191 | 0,5331 | +0,0140 [−0,0061; +0,0342] |
| CheXbert macro F1-5 | 0,4539 | 0,4653 | +0,0114 [−0,0140; +0,0373] |
| ROUGE-L | 0,3591 | 0,3563 | −0,0028 [−0,0081; +0,0023] |
| CIDEr | 0,5304 | 0,5081 | −0,0223 [−0,0641; +0,0184] |
| BLEU-4 (cột điểm: corpus; hiệu: trung bình theo study) | 0,1904 | 0,1883 | −0,0060 [−0,0117; −0,0004] |
| lexicon CE macro F1 | 0,3160 | 0,3034 | −0,0126 [−0,0291; +0,0029] |
| BLEU-1 / METEOR (corpus, không có CI) | 0,4300 / 0,1782 | 0,4335 / 0,1792 | |
| số từ, trung vị (reference: 45) | 35 | 36 | |

- **Quyết định theo quy tắc: không nhận B.** Giữ mô hình đã báo cáo (Stage 1 `run_20261005_paper/phase1c`,
  Stage 2 P2 + LR 3e-4); không đề xuất train Stage 2 đầy đủ trên token refit. Ngoài BLEU-4 theo study, không
  hiệu nào khác 0; cận trên của BLEU-4 (−0,0004) nằm trong 0,0005 quanh 0, theo chuẩn của repo là "ở biên",
  không kết luận được theo hướng nào.

Mức tụt khi hoán đổi soft token (chỉ báo cáo): greedy, 500 study val (cùng 500 study cho hai nhánh, tập con
của 1.360). N = soft token của chính study; P = soft token của study khác (`--soft-token-ablation
permute_across`). Cue, indication và FINDINGS lần chụp trước giữ nguyên.

| P − N, CI 95% | nhánh A (token cũ) | nhánh B (token refit) |
|---|---|---|
| BERTScore | −0,0404 [−0,0535; −0,0273] | −0,0372 [−0,0491; −0,0253] |
| ROUGE-L | −0,0370 [−0,0491; −0,0249] | −0,0356 [−0,0474; −0,0243] |
| CIDEr | −0,1749 [−0,2732; −0,0820] | −0,1717 [−0,2725; −0,0861] |
| BLEU-4 (theo study) | −0,0317 [−0,0441; −0,0197] | −0,0290 [−0,0405; −0,0176] |
| lexicon CE macro F1 | −0,0669 [−0,1011; −0,0318] | −0,0212 [−0,0557; +0,0134] |
| CheXbert micro F1-14 | −0,0833 [−0,1129; −0,0536] | −0,0606 [−0,0884; −0,0328] |
| CheXbert macro F1-14 | −0,0786 [−0,1186; −0,0382] | −0,0372 [−0,0729; +0,0003] |
| CheXbert micro F1-5 | −0,0852 [−0,1250; −0,0452] | −0,0487 [−0,0887; −0,0093] |
| CheXbert macro F1-5 | −0,0985 [−0,1415; −0,0569] | −0,0178 [−0,0697; +0,0323] |
| báo cáo không đổi khi hoán đổi | 81 / 500 | 96 / 500 |
| điểm N: BERTScore / CheXbert micro F1-14 | 0,4830 / 0,4920 | 0,4860 / 0,4913 |

- **Đọc kết quả.**
  1. Cả hai adapter đều đọc kênh soft token, kể cả nhánh A với token đã sụp: thay bằng token của study khác
     làm tụt rõ mọi chỉ số. Token đã sụp (một vector hiệu dụng mỗi study) vẫn mang thông tin ảnh mà Stage 2
     dùng; "kênh ảnh chỉ còn một vector" đúng về hình học, nhưng không có nghĩa là Stage 2 bỏ qua ảnh. Đây là phép thử
     ngoài phân phối: mức tụt gộp cả việc mất ảnh đúng lẫn bị ảnh sai dẫn lạc, nên nó chứng minh adapter đọc
     kênh này chứ không đo chính xác kênh này đóng góp bao nhiêu.
  2. Token refit không được đọc nhiều hơn: điểm ước lượng mức tụt của B bằng hoặc nhỏ hơn của A. Không có CI
     cho hiệu giữa hai mức tụt, nên không kết luận B "ít phụ thuộc ảnh hơn".
  3. Hình học tốt hơn (PR theo batch 1,00 → 26,9) và probe nhỉnh hơn (+0,018 trên phần giữ lại) không thành
     báo cáo tốt hơn ở ngân sách pilot. Khớp với probe: lượng thông tin nhãn thêm vào nhỏ, và ở ngân sách
     này Stage 2 không lấy thêm được gì đo được từ các hướng mới.
  4. Khoảng 1/4 báo cáo beam giống hệt nhau giữa hai nhánh, và 16–19 % báo cáo greedy không đổi khi hoán đổi
     token: với một phần đáng kể study, đầu ra do phần chữ của prompt (FINDINGS lần trước, indication, cue)
     quyết định.
- Giới hạn: một seed mỗi nhánh; 476 update, nên pilot âm không loại trừ hiệu ứng ở 2 epoch đầy đủ; chỉ val,
  không chạm test; không phải giao thức bài báo (prompt có indication và FINDINGS lần chụp trước).
- File trên máy train: `~/qrefit2_20261015/score/{pilot_beam,reliance_A,reliance_B}_{paper,ce}.json`, adapter
  `pilot_{A,B}/adapters/medgemma_qlora_meta_cxr_qformer_with_mhcac_prompt/`, báo cáo sinh `gen_*/` (dữ liệu
  bệnh nhân, chỉ ở máy train).
