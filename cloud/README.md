# META-CXR Kaggle Orchestration

Pipeline tự động hóa: **GCP VM** dùng **Kaggle CLI** push 2 notebook lên Kaggle chạy bằng GPU T4×2, pull output về VM rồi mirror lên **Google Cloud Storage**.

## Kiến trúc

```
Local máy bạn          GCP VM (no GPU)              Kaggle Cloud           GCS
─────────────          ───────────────              ─────────────          ───
push_from_local.sh ─▶  setup_vm.sh                 (T4 x2)
                       run_stage1.sh ──push──▶  meta-cxr-stage1-train
                                    ◀──output──                    ──▶  gs://meta-cxr-checkpoint/stage1/<run-id>/
                                      checkpoint/log từ notebook ──▶  gs://meta-cxr-checkpoint/<run-name>/
                       run_stage2.sh ──push──▶  meta-cxr-stage2-eval
                                    ◀──output──                    ──▶  gs://meta-cxr-checkpoint/stage2/<run-id>/
```

## Tham số (đã hardcode trong `env.sh`)

| Biến | Giá trị |
|------|---------|
| GCP project | `mimic-cxr-jpg-491409` |
| VM instance | `instance-20260521-072851` |
| Zone | `us-central1-a` |
| GCS bucket | `meta-cxr-checkpoint` |
| Kaggle user | `phuong20052` |
| Stage 1 kernel | `phuong20052/meta-cxr-stage1-train` |
| Stage 2 kernel | `phuong20052/meta-cxr-stage2-eval` |

## Quy trình end-to-end

### 1. Trên máy local — đẩy scripts + notebook lên VM

```bash
cd /home/phuong/Documents/KLTN/META_CXR_again
bash META-CXR/cloud/push_from_local.sh
```

### 2. Copy Kaggle credentials

Lấy `kaggle.json` từ <https://www.kaggle.com/settings> → **Create New Token**.

```bash
gcloud compute ssh instance-20260521-072851 \
  --zone=us-central1-a --project=mimic-cxr-jpg-491409 \
  --command="mkdir -p ~/.kaggle"

gcloud compute scp ~/Downloads/kaggle.json \
  instance-20260521-072851:~/.kaggle/kaggle.json \
  --zone=us-central1-a --project=mimic-cxr-jpg-491409

gcloud compute ssh instance-20260521-072851 \
  --zone=us-central1-a --project=mimic-cxr-jpg-491409 \
  --command="chmod 600 ~/.kaggle/kaggle.json"
```

### 3. SSH vào VM, chạy setup (1 lần)

```bash
gcloud compute ssh instance-20260521-072851 \
  --zone=us-central1-a --project=mimic-cxr-jpg-491409

cd ~/meta-cxr-cloud/cloud
bash setup_vm.sh
```

`setup_vm.sh` sẽ:
- Cài `kaggle` CLI qua pip
- Verify scope của VM service account (cần `storage-full` hoặc `cloud-platform`)
- Tạo bucket `gs://meta-cxr-checkpoint` nếu chưa có
- Tạo placeholder `stage1/.keep`, `stage2/.keep`

### 4. Thêm Kaggle Secrets cho mỗi kernel (1 lần, qua Kaggle Web UI)

Sau khi push kernel lần đầu (Bước 5), vào `https://www.kaggle.com/code/phuong20052/meta-cxr-stage1-train/edit` → **Settings → Add-ons → Secrets**:

| Secret | Value |
|--------|-------|
| `KAGGLE_USERNAME` | `phuong20052` |
| `KAGGLE_KEY` | (key trong `kaggle.json`) |
| `WANDB_API_KEY` | từ <https://wandb.ai/authorize> |
| `GCP_SERVICE_ACCOUNT_JSON` | service-account JSON có quyền ghi `gs://meta-cxr-checkpoint` |
| `HF_TOKEN` | (chỉ stage 2) từ <https://huggingface.co/settings/tokens> |

### 5. Chạy stage 1

```bash
cd ~/meta-cxr-cloud/cloud
nohup bash run_stage1.sh > stage1_$(date +%s).log 2>&1 &
tail -f stage1_*.log
```

- Push notebook lên Kaggle, poll `kaggle kernels status` mỗi 5 phút.
- Khi `complete`, pull kernel output để debug/mirror.
- Notebook Cell 6 tự resume từ `gs://meta-cxr-checkpoint/<run-name>/checkpoint_last.pth`.
- Notebook upload checkpoint/log/manifest trực tiếp lên `gs://meta-cxr-checkpoint/<run-name>/`.
- Kernel output được mirror lên `gs://meta-cxr-checkpoint/stage1/<run-id>/`.
- Mỗi Kaggle session tối đa 12h. Nếu chưa xong 15 epoch, chạy lại `run_stage1.sh`; notebook sẽ resume từ GCS.

### 6. Chạy stage 2 (sau khi stage 1 hội tụ)

```bash
nohup bash run_stage2.sh > stage2_$(date +%s).log 2>&1 &
```

Output ở `gs://meta-cxr-checkpoint/stage2/<run-id>/`.

## Lưu ý quan trọng

**Stage 1 checkpoint**: không dùng Kaggle checkpoint dataset nữa. `kernels/train/kernel-metadata.json` chỉ attach 3 data datasets; checkpoint được đọc/ghi qua GCS bằng Kaggle Secret `GCP_SERVICE_ACCOUNT_JSON`.

**Service account scope**: Nếu `setup_vm.sh` cảnh báo thiếu storage scope:
1. Stop VM.
2. GCP Console → VM Instance → Edit → **Cloud API access scopes** → "Allow full access to all Cloud APIs" → Save.
3. Start VM lại.

**Cost notice**: VM chỉ làm orchestrator (không GPU) — dùng `e2-small` là đủ. Chi phí chính ở Kaggle (free) và GCS storage (~$0.02/GB/tháng).

## Files trong thư mục này

```
cloud/
├── README.md                       # File này
├── env.sh                          # Tham số tập trung
├── setup_vm.sh                     # Setup VM (1 lần)
├── push_from_local.sh              # Chạy trên máy local
├── run_stage1.sh                   # Stage 1 training
├── run_stage2.sh                   # Stage 2 eval
├── lib/common.sh                   # log(), poll_kernel(), upload_gcs()
└── kernels/
    ├── train/kernel-metadata.json
    └── eval/kernel-metadata.json
```

## Verification

```bash
# Bucket exist?
gsutil ls gs://meta-cxr-checkpoint/

# Kernel push thành công?
kaggle kernels list -m | grep meta-cxr

# Status?
kaggle kernels status phuong20052/meta-cxr-stage1-train

# Output sau khi chạy?
gsutil ls gs://meta-cxr-checkpoint/stage1/
gsutil ls gs://meta-cxr-checkpoint/stage2/
gsutil ls gs://meta-cxr-checkpoint/04_biovil_pubmedclip/
```
