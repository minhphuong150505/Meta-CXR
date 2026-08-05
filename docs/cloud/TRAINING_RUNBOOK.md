# Runbook — Toàn bộ quá trình train META-CXR

Tổng hợp trạng thái tính đến **2026-07-21**. Đọc file này trước khi chạy bất cứ thứ gì.
Đi kèm: `VM_SPEC.md` (phần cứng), `GCP_L4_TRAINING_SETUP_GUIDE.md` (thao tác chi tiết).

> **Quy ước trong file:** ✅ = đã verify bằng lệnh thật. ⚠️ = ước lượng, chưa đo.
> ⛔ = đang chặn. Mọi con số thời gian đều là ⚠️ — **chưa có phép đo nào trên GPU.**

---

## 1. Bức tranh tổng thể

```
Chest X-Ray 448×448
  ↓
[STAGE 1]  Vision encoders (FROZEN) → View fusion → MHCAC → Q-Former
           ~100–300 giờ ⚠️ · 10 epoch · output: checkpoint_best.pth
  ↓
[STAGE 2]  Q-Former features → MedGemma 1.5 4B + QLoRA
           ~90–115 giờ ⚠️ · 1 epoch · output: LoRA adapter
  ↓
       Báo cáo X-quang (phần FINDINGS)
```

Hai stage chạy **tuần tự**. Stage-2 chế độ `qformer` cần `checkpoint_best.pth` của
Stage-1; chế độ `native` thì không (đó là ablation baseline).

---

## 2. Dữ liệu — ✅ đã xong, không cần làm lại

Preprocessing hoàn tất 2026-07-20. Splits nằm ở
`gs://<DATA_BUCKET>/processed/full_allviews/`:

| Split | Rows | Studies |
|---|---|---|
| train | 365.293 | 220.216 |
| val | 2.963 | |
| test | 5.082 | |

- Toàn bộ p10–p19, giữ mọi view (AP/PA/LATERAL/LL/UNKNOWN) → `multi_view` bật được
- Patient- và study-disjoint ✅ đã verify
- 4.851 dòng train (1,33%) không có nhãn CheXpert → vẫn train report generation,
  **bị mask khỏi classification loss**
- 32,6% report không có tag FINDINGS → parser lấy phần thân narrative
- `image_path` trong CSV là **đường dẫn tương đối** — đừng đổi thành tuyệt đối

---

## 3. Stage 1 — BLIP2/Q-Former alignment

**Config production: `pretraining/configs/mimic_cxr_full_l4.yaml`**
(`mimic_cxr_2gpu.yaml` là legacy — `multi_view: false`, `warmup_steps: 32000` không
bao giờ ramp xong. Đừng copy giá trị từ đó.)

### Kiến trúc
- Encoder **đóng băng**: BioViL-T (1408) + PubMedCLIP (768) + SwinV2; RadDINO tắt
- View fusion: mỗi encoder một module, chạy trên output thô **trước** FC projection.
  Khởi tạo zero ở `W_O` và FFN cuối → identity tuyệt đối ở step 0
- MHCAC: 14 bệnh × 3 lớp (Positive/Negative/Uncertain)
  - **teacher** = ảnh + text báo cáo, **chỉ khi train**
  - **student** = chỉ ảnh ← đây là thứ inference dùng
  - teacher chưng cất vào student qua KL loss, teacher được detach
- Q-Former: 32 query token, cross-attention mỗi block thứ 2, loss ITC + ITM + LM,
  ITC có negative queue 1024 sample

### Hyperparameters
| | |
|---|---|
| `max_epoch` | **20** (early stopping patience 5) |
| `batch_size_train` | 2 |
| `accum_grad_iters` | 32 → effective batch **64** |
| `num_workers` | 4 |
| `amp` | true |
| `warmup_steps` | 300 (đếm theo **optimizer update**, không phải microbatch) |
| `study_sampling` | true — **1 sample = 1 study**, không phải 1 ảnh |
| `anchor_priority` | `[PA, AP, lateral]`, `max_aux_views: 1` |
| `lambda_mpc` / `lambda_view_consistency` | 0.1 / 0.05 |
| `selection_metric` | `macro_auprc` — calibrate threshold F1 sau khi chọn checkpoint |
| `uncertain_policy` | `ignore_uncertain` cho classification loss và binary metrics |
| `save_freq` | 3 |

> `study_sampling: true` đổi định nghĩa epoch từ ~365k dòng ảnh thành **~220k study**.
> Mọi số throughput cũ đều vô nghĩa vì lý do này.

### Checkpoint
```
checkpoint_last.pth   ← lưu CUỐI MỖI EPOCH, bất kể save_freq  (điểm resume)
checkpoint_best.pth   ← khi f1_positive_macro (toàn val split) cải thiện
checkpoint_2/5/8.pth  ← save_freq: 3 → snapshot cố định, không bị ghi đè
```
`save_freq: 3` giữ thêm snapshot ở epoch 2/5/8 để một run phải dựng lại có điểm
xuất phát cố định, thay vì chỉ có `checkpoint_last` đang cuộn. Resume qua
`run.resume_ckpt_path`. **Test split bị giữ hoàn toàn ngoài việc chọn checkpoint**,
chỉ chạy một lần từ `checkpoint_best` sau khi train xong.

⚠️ Mỗi snapshot là một file đầy đủ — 3 file thêm sẽ chiếm thêm dung lượng cả trên
boot disk lẫn GCS. Kiểm `df -h` sau epoch 2 trước khi để run chạy tiếp không giám sát.

### Lệnh
```bash
cloud/run_stage1.sh          # đường đúng — có upload GCS
```
→ ghi vào `~/meta-cxr-output/stage1/$RUN_ID/`, rồi upload
`gs://$GCS_BUCKET/stage1/$STAGE1_RUN/$RUN_ID/`

⚠️ Chạy thẳng `python -m pretraining.train` thì output rơi vào `pretraining/outputs/`
(đường dẫn tương đối, trong repo) và **không upload GCS**. Có trong `.gitignore` nên
không lộ, nhưng VM xóa là mất.

---

## 4. Stage 2 — MedGemma QLoRA

**`google/medgemma-1.5-4b-it`** — 4-bit NF4 double-quant, compute bfloat16,
LoRA `r=8, alpha=16, target_modules="all-linear"` (vision tower không train).

### Hai chế độ
| Mode | Cần Stage-1? | Mô tả |
|---|---|---|
| `qformer` | **có** | 32 soft token từ Q-Former, chiếu qua `img_proj`, **thay thế** embedding tại vị trí `<image_soft_token>` |
| `native` | **không** | image tower gốc của MedGemma — ablation baseline, không nhận nhãn Stage-1 |
| `both` | có | chạy tuần tự cả hai |

Findings P/N/U đi vào LM dưới dạng **text** qua `utils/prompter.py`.

### Hyperparameters
`--train-epochs 1` · `--batch-size 2` · `--grad-accum 8` · `--max-length 768`
· `--max-new-tokens 256` · `--val-generation-limit 300`

### Ba nút cổ chai đã biết
1. **Feature extraction dùng `batch_size=1`** → 6–12 giờ ⚠️, ghi một file `.pt`
   khoảng 10–11 GB cho 220k dòng
2. **`generate()` chạy từng record, không batch** → full test 5.082 dòng ≈ 25–30 giờ ⚠️
3. **`train_fine` chỉ lưu adapter SAU epoch cuối** → xem §6

### Lệnh
```bash
cloud/run_stage2.sh                    # STAGE2_IMAGE_MODE điều khiển mode
# hoặc trực tiếp:
python training/run_medgemma_qlora.py --checkpoint-root <dir> --output-dir <dir>
```
Output mặc định: `training/outputs/medgemma_qlora_full/`

---

## 5. ⛔ Blocker — thứ tự bắt buộc

Không bỏ qua bước nào; mỗi bước chặn bước sau.

### B1. `GPUS_ALL_REGIONS = 0` — chặn cứng, **cần bạn thao tác**
Đã thử tạo VM và thất bại thật:
```
ERROR: Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.
```
GCP có **hai** quota GPU độc lập. `NVIDIA_L4_GPUS` per-region = 1 ✅ nhưng
`GPUS_ALL_REGIONS` global = 0 ⛔. Global = 0 thì không tạo được GPU ở đâu cả.

Nguyên nhân: tài khoản còn ở **free trial**, và đơn xin quota GPU **bị từ chối tự động**
khi còn trial.

1. Console → Billing → **"Activate full account"** (credit còn lại được giữ nguyên,
   vẫn tiêu trước, không mất gì)
2. Rồi mới xin `GPUS_ALL_REGIONS` → **1** tại IAM & Admin → Quotas

### B2. `configs/env_config.yaml` vẫn là bản Kaggle — **4/4 dòng sai**
```yaml
output_dir:     "/kaggle/temp/output"        # không tồn tại trên VM
checkpoint_dir: "/kaggle/temp/checkpoints"   # không tồn tại trên VM
gcs_bucket:     "gs://meta-cxr-checkpoint"   # ⛔ bucket ĐÃ BỊ XÓA (404)
gcs_project:    "mimic-cxr-jpg-491409"       # ⛔ sai project, billing đã TẮT
```
Bucket checkpoint thật đã tạo 2026-07-20 có tên khác. Không sửa thì training tốn
hàng chục giờ GPU rồi chết ở bước upload.

### B3. Quota `SSD_TOTAL_GB` = 500 < 1000 cần cho full run
`pd-ssd` và `pd-balanced` **đều** tính vào quota này. Boot 200 + data 800 = 1000 > 500.
Xin nâng lên ≥ 1100. **Không** dùng `pd-standard` thay thế — HDD, 800 GB chỉ ~600 IOPS,
GPU sẽ ngồi chờ disk. Smoke run thì không cần data disk nên không vướng.

### B4. W&B project name lệch
`env_config.yaml` đang để `project: "meta-cxr-encoder-comparison"`. `full_l4.yaml`
không tự khai `wandb_project` nên run production sẽ đổ lẫn vào project encoder-comparison.
Sửa 1 dòng, nhưng phát hiện lúc đang chạy thì log đã lẫn rồi.

---

## 6. Rủi ro mất tiền — đọc kỹ nếu ngân sách hẹp

**Stage-2 không có điểm cứu hộ nào.** `train_fine` chỉ lưu adapter sau epoch cuối,
mà `--train-epochs 1` nghĩa là suốt 55–70 giờ ⚠️ **không có checkpoint trung gian**.
Sập ở giờ thứ 60 → mất trắng, chạy lại từ đầu.

| | Điểm cứu hộ |
|---|---|
| Stage-1 | `checkpoint_last.pth` mỗi epoch + snapshot 2/5/8 (`save_freq: 3`) ✅ |
| Stage-2 | **không có gì** ✗ |

Nên sửa trước khi đốt tiền GPU: thêm save giữa epoch, hoặc chia thành nhiều epoch ngắn.
**Không dùng Spot VM** cho tới khi việc này xong.

---

## 7. Ngân sách — credit ~$230 (6 triệu VND)

| Hạng mục | Giờ ⚠️ | Tiền ⚠️ |
|---|---|---|
| Smoke run | 2–3 | $3 |
| Stage-1 (**10 epoch**) | 100–300 | **$93–286** |
| Stage-2 feature extract | 6–12 | $6–12 |
| Stage-2 QLoRA 1 epoch | 55–70 | $52–70 |
| Test gen 5.082 dòng | 25–30 | $24–30 |
| Disk 1TB (~10+ ngày, **tính cả khi VM tắt**) | — | $25–40 |
| **Stage-1 + `qformer`** | ~190–415 | **$203–441** |
| **Nếu `both`** | ~270–520 | **$280–540** |

### ⛔ Ở 10 epoch, ngân sách $230 KHÔNG đủ

Đây không còn là "vừa khít" như bản 3 epoch. Riêng Stage-1 đã có thể ăn **$93–286**,
tức là cận trên của nó **một mình đã vượt toàn bộ credit** trước khi Stage-2 bắt đầu.
Chỉ nhánh cận dưới ($203) mới lọt, và nó giả định mọi ước lượng đều rơi vào đầu tốt nhất
của khoảng — với code **chưa từng chạy trên GPU** thì đó không phải giả định an toàn.

Điểm mấu chốt: `early_stop_patience: 10 == max_epoch` nghĩa là **early stop không bao giờ
kích hoạt**. Run sẽ trả tiền đủ 10 epoch kể cả khi F1 đã phẳng từ epoch 4. `checkpoint_best`
vẫn giữ đúng epoch tốt nhất, nên đây là rủi ro **tiền**, không phải rủi ro chất lượng.

**Phải làm trước khi commit tiền vào full run:**
1. Smoke run lấy **giờ/epoch thật**, rồi nhân 10 và tính lại bảng này. Đây là việc bắt buộc,
   không phải tuỳ chọn — khoảng $93–286 quá rộng để lập kế hoạch.
2. Nếu giờ/epoch thật ở nửa trên của khoảng → **giảm `max_epoch`**, hoặc hạ
   `early_stop_patience` xuống 2–3 để run tự dừng khi F1 phẳng.
3. `STAGE2_IMAGE_MODE=qformer` thay vì `both` → tiết kiệm **$77–99**
4. `--test-limit 500` thay vì 5.082 → tiết kiệm **~$25**
5. `gcloud compute instances stop` ngay sau mỗi giai đoạn (disk vẫn tính tiền,
   nhưng GPU — phần đắt — thì không)

---

## 8. Đã sẵn sàng ✅ vs chưa verify ⚠️

### Sẵn sàng
| | |
|---|---|
| Token HF + quyền gated MedGemma | ✅ `config.json` tải được http=200 |
| W&B API key | ✅ GraphQL 200, entity `phuongnm150505-uit` |
| Bucket data | ✅ UBLA + PAP enforced |
| Bucket checkpoint | ✅ đã tạo, PAP enforced |
| Fix `common.sh` (PAP/UBLA) | ✅ đã commit + push |
| Splits train/val/test | ✅ đã build, đã verify disjoint |
| Quota L4 per-region | ✅ = 1 |

### Chưa verify — smoke run sinh ra để trả lời
1. **`transformers==4.53.2` có load nổi MedGemma 1.5 không** — pin từ giữa 2025,
   model cập nhật 2026-04-13. **Câu hỏi số 1**; hỏng thì biết trong ~15 phút
2. **torch 2.5.1+cu124 trên driver 580** — image CUDA 12.4 đã bị GCP gỡ, chỉ còn 12.9.
   Dự kiến chạy được nhờ CUDA minor-version compatibility, nhưng chưa thử
3. **VRAM thật của Stage-1** — 3 encoder + Q-Former + MHCAC + queue 1024 trên 24 GB.
   `batch_size 2` + amp nên nhiều khả năng vừa, nhưng là suy luận
4. **Đỉnh RAM khi dựng feature cache** — file 10–11 GB trên máy 32 GB. Theo dõi `free -g`
5. **Toàn bộ throughput** — không có phép đo nào cho recipe hiện tại

> **Chưa có gì trong đợt rewrite Stage-1/Stage-2 từng chạy trên GPU.**

---

## 9. Thứ tự thực hiện

```
□ B1  Upgrade billing → paid                      [BẠN, Console]
□ B1  Xin quota GPUS_ALL_REGIONS = 1              [BẠN, Console]
□ B3  Xin quota SSD_TOTAL_GB ≥ 1100 (nộp song song, chỉ cần cho full run)
□ B2  Viết lại configs/env_config.yaml cho VM
□ B4  Sửa wandb project name
□     Dựng VM (VM_SPEC.md §5)
□     SMOKE RUN — native mode, ~2–3 giờ, $3
        python training/run_medgemma_qlora.py --image-mode native \
          --train-limit 500 --val-limit 50 --test-limit 20 --no-upload
      → trả lời câu hỏi 1, 2 ở §8 và cho throughput THẬT
□     Tính lại §7 bằng số đo thật (giờ/epoch × 10), quyết có đủ credit chạy full không
□     Nếu giờ/epoch cao → hạ max_epoch hoặc early_stop_patience (§7)
□ §6  Thêm checkpoint giữa epoch cho Stage-2
□     Stage-1 full (10 epoch)
□     Stage-2 qformer (1 epoch)
```

**Smoke run là bước bản lề.** Nó biến toàn bộ bảng ⚠️ ở §7 thành số đo, với giá $3 —
trước khi cam kết $138–241 vào một pipeline chưa từng chạy trên GPU.
