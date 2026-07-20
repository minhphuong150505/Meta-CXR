> **KHÔNG CÒN HIỆU LỰC.** Đường chạy Kaggle đã bị gỡ: MIMIC-CXR và mọi dẫn xuất
> (report đã làm sạch, split CSV, feature, prediction, checkpoint) không được
> publish thành Kaggle Dataset — PhysioNet DUA cấm phân phối lại. Xem
> `configs/kaggle_datasets.yaml` (`policy.storage: private-gcs-only`). Đường chạy
> được hỗ trợ là 1 GPU L4 với `pretraining/configs/mimic_cxr_full_l4.yaml` và
> `cloud/run_stage1.sh` / `cloud/run_stage2.sh`, ghi vào bucket GCS riêng tư.
> Giữ file này chỉ để tham chiếu lịch sử.

# META-CXR — Hướng Dẫn Chạy Training Trên Kaggle

## Tổng Quan Pipeline

```
Chest X-Ray (MIMIC-CXR-JPG)
  → Vision Encoders (BioViL-T / PubMedCLIP)
  → MHCAC Classification (14 nhóm bệnh)
  → BLIP2 Q-Former
  → Vicuna-7B
  → Radiology Report
```

Training chạy trên **Kaggle 2× T4 GPU** (miễn phí), chia thành **nhiều session** nếu cần do giới hạn 12h/session. Checkpoint được tự động lưu mỗi 3 epoch và push lên Kaggle Dataset để resume session sau.

---

## Thiết Lập Một Lần (Setup)

### 1. Tạo Kaggle API Token

1. Vào [kaggle.com](https://www.kaggle.com) → **Account** → **Settings** → cuộn xuống **API**
2. Click **"Create New Token"** → file `kaggle.json` tải về máy
3. Mở file, lấy 2 giá trị: `username` và `key`

### 2. Thêm Secrets Vào Notebook Kaggle

Mở notebook `META_CXR_kaggle.ipynb` trên Kaggle, rồi:

**Settings (⚙️ bên phải) → Add-ons → Secrets → "+ Add New Secret"**

Thêm 3 secrets sau:

| Label | Value |
|-------|-------|
| `WANDB_API_KEY` | API key từ [wandb.ai](https://wandb.ai) → Settings → API Keys |
| `KAGGLE_USERNAME` | `username` trong `kaggle.json` (vd. `minhphuong150505`) |
| `KAGGLE_KEY` | `key` trong `kaggle.json` |

> **Tại sao cần `KAGGLE_USERNAME` + `KAGGLE_KEY`?**  
> Cell 8 dùng Kaggle CLI để push checkpoint lên dataset của bạn. CLI cần xác thực qua hai giá trị này. Cell 8 tự đọc từ Secrets và cấu hình — **bạn không cần tạo `~/.kaggle/kaggle.json` thủ công**.

### 3. Thiết Lập GPU

**Settings → Accelerator → GPU T4 x2** (bắt buộc — DDP cần 2 GPU)  
**Settings → Internet → On** (để clone GitHub và download packages)

### 4. Đính Kèm 2 Input Datasets

**Notebook Settings → Add Data** → tìm và thêm 2 datasets:

| Dataset | Slug | Nội dung |
|---------|------|----------|
| MIMIC-CXR-JPG-LITE | `mimic-cxr-jpg-lite` | JPG images + metadata CSVs |
| mimic-cxr-reported | `mimic-cxr-reported` | `mimic_cxr_cleaned.csv` + `.txt` reports |

> **`mimic_cxr_cleaned.csv` cần có sẵn trong dataset `mimic-cxr-reported`.**  
> Nếu chưa có, chạy notebook `01_generate_mimic_cxr_cleaned_csv.ipynb` một lần để tạo, rồi upload lên dataset đó.

---

## Câu Hỏi Thường Gặp

### Có cần tạo dataset checkpoint thủ công không?

**Không.** Cell 8 tự động:
- Lần đầu: tạo mới dataset `meta-cxr-checkpoints` (private, dưới tên tài khoản của bạn)
- Các lần sau: push version mới lên dataset đó

Bạn chỉ cần thêm `KAGGLE_USERNAME` và `KAGGLE_KEY` vào Secrets là xong.

---

## Chạy Session 1 (Train từ đầu)

Chạy các cell theo thứ tự:

```
Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 → Cell 6 → Cell 7 → Cell 8
```

| Cell | Tên | Làm gì |
|------|-----|--------|
| **1** | Install Dependencies | Cài packages còn thiếu, verify 2 GPUs |
| **2** | WandB Setup | Đăng nhập WandB để track experiments |
| **3** | Clone Repository | Clone code từ GitHub về `/kaggle/working/META-CXR` |
| **4** | Verify Datasets | Kiểm tra 2 input datasets, copy `mimic_cxr_cleaned.csv` |
| **5** | Write env_config.yaml | Tạo file config paths cho Kaggle environment |
| **6** | Launch Training | Chạy DDP training 2-GPU, **auto-detect resume nếu có checkpoint** |
| **7** | Display Results | Hiển thị metrics, predictions, checkpoint list |
| **8** | Push Checkpoints | Tự push checkpoints lên Kaggle Dataset |

**Sau khi Cell 8 chạy xong**, bạn sẽ thấy:
```
✅ Done: https://www.kaggle.com/datasets/minhphuong150505/meta-cxr-checkpoints
Session tiếp theo: Notebook Settings → Add Data → tìm 'meta-cxr-checkpoints' → Add. Cell 6 sẽ tự resume.
```

---

## Chạy Session 2+ (Resume từ checkpoint)

**Bước duy nhất cần làm thủ công:**

1. Mở notebook trên Kaggle
2. **Settings → Add Data** → tìm dataset `meta-cxr-checkpoints` (của bạn) → **Add**
3. Chạy lại các cell **1 → 8** như bình thường

Cell 6 sẽ tự động phát hiện dataset checkpoint đã attach, tìm checkpoint epoch lớn nhất và resume từ đó. Bạn sẽ thấy trong output:
```
Resume from: /kaggle/input/datasets/meta-cxr-checkpoints/checkpoint_2.pth
```

---

## Cấu Hình Training

Sửa `pretraining/configs/mimic_cxr_2gpu.yaml` để điều chỉnh:

```yaml
run:
  max_epoch: 10        # tổng số epoch (tính cả các session trước)
  save_freq: 3         # lưu checkpoint mỗi N epoch (mặc định 3)
  batch_size_train: 8  # batch/GPU — giảm xuống 4 nếu OOM
  accum_grad_iters: 4  # effective batch = 8 × 2 GPUs × 4 = 64
  init_lr: 2e-4
  warmup_steps: 1000
```

> **Lưu ý `max_epoch` khi resume:**  
> `max_epoch` là tổng epoch kể từ epoch 0. Nếu session 1 chạy epoch 0–4 (max_epoch=5), session 2 cần set `max_epoch: 10` để chạy thêm epoch 5–9.

---

## Workflow Nhiều Session

```
Session 1: epoch 0 → 4
  - Checkpoint được lưu tại epoch 2 (save_freq=3, epoch 2 = lần đầu thỏa (2+1)%3==0)
  - checkpoint_best.pth lưu bất cứ khi nào val metric cải thiện
  - Cell 8 push → dataset meta-cxr-checkpoints (version 1)

Session 2: epoch 5 → 9
  - Attach dataset meta-cxr-checkpoints → Cell 6 auto-resume từ checkpoint_2.pth
  - Checkpoint lưu tại epoch 5, 8
  - Cell 8 push → dataset meta-cxr-checkpoints (version 2)
```

Sau mỗi session, `/kaggle/working/output/` chứa tối đa:
- `checkpoint_2.pth`, `checkpoint_5.pth`, `checkpoint_8.pth` — checkpoint lịch sử
- `checkpoint_best.pth` — checkpoint có val metric tốt nhất

Tổng dung lượng: ~4GB cho 10 epoch (3 numbered + 1 best).

---

## Tải Checkpoint Về Máy Local

Sau khi Cell 8 push thành công, tải về máy:

```bash
# Cài kaggle CLI
pip install kaggle

# Tải toàn bộ checkpoint dataset về ./checkpoints/
kaggle datasets download minhphuong150505/meta-cxr-checkpoints -p ./checkpoints --unzip
```

---

## Troubleshooting

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| Cell 8: `Kaggle API credentials not found` | Chưa thêm secrets | Thêm `KAGGLE_USERNAME` + `KAGGLE_KEY` vào Kaggle Secrets |
| Cell 6: training không resume | Dataset checkpoint chưa attach | Settings → Add Data → thêm `meta-cxr-checkpoints` |
| `RuntimeError: iostream error` khi save | Disk full | Tăng `save_freq` (ví dụ 5) để ít checkpoint hơn |
| `RuntimeError: find_unused_parameters` | DDP bug | Đã fix trong `runner_base.py` — pull code mới nhất |
| `FileNotFoundError: mimic_cxr_cleaned.csv` | CSV chưa có trong dataset | Chạy `01_generate_mimic_cxr_cleaned_csv.ipynb` một lần |
| `CUDA out of memory` | Batch size quá lớn | Giảm `batch_size_train` từ 8 → 4 |
| Chỉ 1 GPU | Accelerator sai | Settings → Accelerator → **GPU T4 x2** |
| Cell 3 clone chậm | Network throttle | Bình thường, đợi vài phút |

---

## Cấu Trúc Files Quan Trọng

```
META-CXR/
├── META_CXR_kaggle.ipynb         # Notebook chính — chạy theo thứ tự Cell 1→8
├── 01_generate_mimic_cxr_cleaned_csv.ipynb  # Chạy 1 lần để tạo mimic_cxr_cleaned.csv
├── configs/
│   ├── kaggle_datasets.yaml      # Slugs datasets, checkpoint config (sửa tại đây)
│   └── env_config.yaml           # Auto-generated bởi Cell 5 — không sửa trực tiếp
├── pretraining/configs/
│   └── mimic_cxr_2gpu.yaml       # Hyperparameters training (sửa tại đây)
└── CHECKPOINT_WORKFLOW.md        # Chi tiết kỹ thuật về checkpoint workflow
```
