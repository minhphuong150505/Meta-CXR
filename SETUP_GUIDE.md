# META-CXR — Hướng Dẫn Cài Đặt và Chạy Trên Kaggle

## Tổng Quan

Project META-CXR được refactor để:
- Chỉ sử dụng **MIMIC-CXR-JPG** dataset
- Đọc data từ **GCS bucket** `gs://mimic-cxr-jpg-data` (project: `mimic-cxr-jpg-491409`)
- Train song song trên **2 GPU** via PyTorch DDP
- Chạy hoàn toàn trên **Kaggle** (2x T4 GPU miễn phí)
- Cấu hình qua **YAML config** (không hardcode)

---

## Cấu Trúc Files Quan Trọng

```
META-CXR/
├── configs/
│   ├── env_config.yaml          # Config paths, WandB, Java (overwrite bởi Kaggle Cell 5)
│   └── env_config.yaml.example  # Template cho môi trường local
├── pretraining/configs/
│   └── mimic_cxr_2gpu.yaml      # Config training 2-GPU DDP
├── META_CXR_kaggle.ipynb        # Notebook chạy trên Kaggle
├── local_config.py              # Load từ env_config.yaml (không còn hardcode)
└── ...
```

---

## Phần 1 — Lấy GCS Service Account Key

### Bước 1 — Tạo Service Account

1. Vào [Google Cloud Console](https://console.cloud.google.com) → chọn project **`mimic-cxr-jpg-491409`**
2. Menu trái: **IAM & Admin** → **Service Accounts**
3. Click **"+ Create Service Account"**
   - Name: ví dụ `kaggle-mimic-reader`
   - Click **Create and Continue**
4. Gán role: tìm và chọn **"Storage Object Viewer"** (đủ quyền đọc GCS)
5. Click **Done**

### Bước 2 — Tạo JSON Key

1. Trong danh sách Service Accounts, click vào account vừa tạo
2. Tab **"Keys"** → **"Add Key"** → **"Create new key"**
3. Chọn **JSON** → **Create**
4. File `.json` sẽ tự download về máy — **giữ file này an toàn, không commit lên git**

File JSON có dạng:
```json
{
  "type": "service_account",
  "project_id": "mimic-cxr-jpg-491409",
  "private_key_id": "...",
  "private_key": "-----BEGIN RSA PRIVATE KEY-----\n...",
  "client_email": "kaggle-mimic-reader@mimic-cxr-jpg-491409.iam.gserviceaccount.com",
  ...
}
```

---

## Phần 2 — Cấu Hình Kaggle Notebook

### Bước 1 — Thiết lập GPU

1. Mở notebook `META_CXR_kaggle.ipynb` trên Kaggle
2. **Settings** (bên phải) → **Accelerator** → chọn **"GPU T4 x2"**
3. Bật **"Internet"** (cần để clone GitHub và download packages)

### Bước 2 — Thêm GCS Secret

1. Bên phải màn hình: **Add-ons** → **Secrets**
2. Click **"+ Add New Secret"**
3. Điền:
   - **Label**: `GCS_SERVICE_ACCOUNT`
   - **Value**: paste **toàn bộ nội dung** file JSON vừa download (từ `{` đến `}`)
4. Click **"Attach to notebook"**

### Bước 3 — Push Code Lên GitHub (bắt buộc)

Notebook sẽ clone repo từ GitHub ở Cell 3, vì vậy **tất cả thay đổi code phải được push trước**:

```bash
git add configs/ pretraining/configs/mimic_cxr_2gpu.yaml META_CXR_kaggle.ipynb \
        local_config.py model/lavis/data/ReportDataset.py \
        model/lavis/runners/runner_base.py pretraining/train.py
git commit -m "refactor: YAML config, MIMIC-CXR only, 2-GPU DDP, Kaggle notebook"
git push
```

---

## Phần 3 — Chạy Notebook (7 Cells theo thứ tự)

### Cell 1 — Cài Dependencies
- Cài các package còn thiếu trên Kaggle: `omegaconf`, `pycocoevalcap`, `torchinfo`, `wandb`, `peft`, v.v.
- Download NLTK punkt data
- Kiểm tra số GPU (phải thấy **2 GPUs**)

### Cell 2 — GCS Authentication
- Đọc secret `GCS_SERVICE_ACCOUNT` từ Kaggle Secrets
- Lưu credentials ra `/kaggle/working/gcs_credentials.json`
- Xác minh kết nối tới bucket — phải in ra `Bucket 'mimic-cxr-jpg-data' accessible: True`

### Cell 3 — Clone GitHub Repository
- Clone `https://github.com/DasithEdirisinghe/META-CXR.git` về `/kaggle/working/META-CXR`
- Nếu đã tồn tại thì `git pull` để lấy code mới nhất

### Cell 4 — Download Data từ GCS
- Liệt kê cấu trúc bucket để xác nhận
- Sync toàn bộ data xuống `/kaggle/working/data/`
- Kiểm tra 4 file CSV quan trọng:
  - `mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv`
  - `mimic-cxr/report_processed/mimic_cxr_cleaned.csv`
  - `data/data_files/mimic-cxr-2.0.0-chexpert.csv`
  - `data/data_files/mimic-cxr-2.0.0-metadata.csv`

> **Lưu ý**: Bước này tốn thời gian nhất nếu dataset lớn. Dùng `gsutil -m rsync` để tự skip file đã có khi chạy lại.

### Cell 5 — Tạo `configs/env_config.yaml`
- Tự phát hiện `JAVA_HOME` trên Kaggle
- Ghi file `configs/env_config.yaml` với đường dẫn Kaggle (`/kaggle/working/data`)
- In ra nội dung file để xác nhận

### Cell 6 — Launch Training 2-GPU
Chạy lệnh:
```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --master_port=12355 -m pretraining.train \
  --cfg-path pretraining/configs/mimic_cxr_2gpu.yaml
```
- Output được stream trực tiếp ra notebook
- Mỗi epoch có thể mất 30–90 phút tùy kích thước dataset

### Cell 7 — Xem Kết Quả Đánh Giá
- Đọc và hiển thị log metrics theo từng epoch (dưới dạng bảng pandas)
- Hiển thị sample predictions
- Liệt kê các checkpoint đã lưu

---

## Phần 4 — Cấu Hình Nâng Cao

### Thay Đổi Hyperparameters

Sửa file `pretraining/configs/mimic_cxr_2gpu.yaml`:

```yaml
run:
  max_epoch: 10           # số epoch
  batch_size_train: 8     # batch/GPU (T4=16GB, giảm nếu OOM)
  accum_grad_iters: 4     # effective batch = 8 x 2 GPUs x 4 = 64
  init_lr: 2e-4           # learning rate
  warmup_steps: 1000
```

### Chạy Ở Môi Trường Local

1. Copy và điều chỉnh config:
```bash
cp configs/env_config.yaml.example configs/env_config.yaml
# Sửa data_root, output_dir, java.home trong env_config.yaml
```

2. Chạy single GPU:
```bash
python -m pretraining.train --cfg-path pretraining/configs/blip2_pretrain_stage1.yaml
```

3. Chạy 2-GPU DDP:
```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_2gpu.yaml
```

### WandB Experiment Tracking

Để bật WandB logging, sửa `configs/env_config.yaml`:
```yaml
wandb:
  entity: "your-wandb-username"
  project: "meta-cxr"
```

---

## Cấu Trúc GCS Bucket Yêu Cầu

```
gs://mimic-cxr-jpg-data/
├── mimic-cxr-jpg/
│   └── 2.1.0/
│       ├── mimic-cxr-2.0.0-split.csv
│       └── files/
│           ├── p10/...   (JPG images)
│           └── p19/...
├── mimic-cxr/
│   └── report_processed/
│       └── mimic_cxr_cleaned.csv
└── data/
    └── data_files/
        ├── mimic-cxr-2.0.0-chexpert.csv
        └── mimic-cxr-2.0.0-metadata.csv
```

---

## Troubleshooting

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `Bucket accessible: False` | Sai credentials hoặc thiếu quyền | Kiểm tra service account có role `Storage Object Viewer` |
| `FileNotFoundError: env_config.yaml` | Cell 5 chưa chạy | Chạy Cell 5 trước Cell 6 |
| `CUDA out of memory` | Batch size quá lớn | Giảm `batch_size_train` từ 8 xuống 4 trong `mimic_cxr_2gpu.yaml` |
| `ModuleNotFoundError` | PYTHONPATH chưa set | Cell 6 đã set `PYTHONPATH=/kaggle/working/META-CXR` tự động |
| Chỉ thấy 1 GPU | Kaggle accelerator sai | Settings → Accelerator → GPU T4 x2 |
| `MISSING: mimic_cxr_cleaned.csv` | Cấu trúc bucket khác | Chạy `!gsutil ls gs://mimic-cxr-jpg-data/` để kiểm tra và điều chỉnh paths |
