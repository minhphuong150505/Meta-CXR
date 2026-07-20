> **KHÔNG CÒN HIỆU LỰC.** Checklist này đi qua Kaggle. Đường chạy hiện tại là
> local → VM GPU → GCS riêng tư, không có bước Kaggle nào.

# Những việc bạn cần làm tiếp theo

Checklist tuần tự từ máy local → VM → Kaggle → GCS. Mỗi bước có lệnh copy-paste sẵn.

---

## Bước 1 — Đẩy scripts + notebook lên VM (chạy trên máy local)

```bash
cd /home/phuong/Documents/KLTN/META_CXR_again
bash META-CXR/cloud/push_from_local.sh
```

**Kết quả mong đợi**: SCP thành công thư mục `cloud/` + 2 notebook lên `~/meta-cxr-cloud/` trên VM.

---

## Bước 2 — Lấy Kaggle API token

1. Mở <https://www.kaggle.com/settings>
2. Cuộn xuống mục **API** → click **Create New Token**
3. Trình duyệt sẽ tải về file `kaggle.json` (thường ở `~/Downloads/kaggle.json`)

---

## Bước 3 — Copy kaggle.json lên VM (chạy trên máy local)

```bash
gcloud compute ssh instance-20260521-072851 \
  --zone=us-central1-a --project=mimic-cxr-jpg-491409 \
  --command="mkdir -p ~/.kaggle"

gcloud compute scp ~/Downloads/kaggle.json \
  instance-20260521-072851:~/.kaggle/kaggle.json \
  --zone=us-central1-a --project=mimic-cxr-jpg-491409
```

---

## Bước 4 — SSH vào VM và chạy setup (1 lần duy nhất)

```bash
gcloud compute ssh instance-20260521-072851 \
  --zone=us-central1-a --project=mimic-cxr-jpg-491409
```

Sau đó trên VM:

```bash
cd ~/meta-cxr-cloud/cloud
bash setup_vm.sh
```

**Kết quả mong đợi**:
- `kaggle` CLI cài xong
- Bucket `gs://meta-cxr-checkpoints` được tạo (hoặc đã tồn tại)
- In ra layout: `gs://meta-cxr-checkpoints/stage1/`, `stage2/`

**Nếu có cảnh báo về service account scope**:
1. Trên local: `gcloud compute instances stop instance-20260521-072851 --zone=us-central1-a --project=mimic-cxr-jpg-491409`
2. Vào GCP Console → VM Instance → Edit → **Cloud API access scopes** → chọn **"Allow full access to all Cloud APIs"** → Save
3. Start lại: `gcloud compute instances start instance-20260521-072851 --zone=us-central1-a --project=mimic-cxr-jpg-491409`
4. SSH lại và chạy lại `bash setup_vm.sh`

---

## Bước 5 — Chuẩn bị cho session push stage 1 lần ĐẦU TIÊN

Vì dataset `meta-cxr-checkpoints` trên Kaggle chưa tồn tại, phải xóa nó khỏi `dataset_sources` cho session đầu (Cell 8 của notebook sẽ tự `kaggle datasets create`).

Trên VM, sửa file `~/meta-cxr-cloud/cloud/kernels/train/kernel-metadata.json` — bỏ dòng `"phuong20052/meta-cxr-checkpoints"`:

```bash
cd ~/meta-cxr-cloud/cloud
nano kernels/train/kernel-metadata.json
```

Sau khi sửa, file phải còn 3 dataset sources hợp lệ (json valid, không thừa dấu phẩy).

Sau session 1 xong (Cell 8 đã tạo dataset), add lại dòng đó cho các session tiếp theo.

---

## Bước 6 — Push stage 1 lên Kaggle (lần đầu, để khởi tạo kernel)

```bash
cd ~/meta-cxr-cloud/cloud
bash run_stage1.sh
```

Script sẽ push notebook lên Kaggle. **Khi nó bắt đầu poll status**, bạn có thể `Ctrl+C` lần đầu để dừng — chỉ cần kernel xuất hiện trên Kaggle web UI là đủ để làm Bước 7.

---

## Bước 7 — Thêm Kaggle Secrets vào kernel (qua Web UI, 1 lần)

Truy cập 2 link (sau khi đã push kernel ở Bước 6):

- Stage 1: <https://www.kaggle.com/code/phuong20052/meta-cxr-stage1-train/edit>
- Stage 2: <https://www.kaggle.com/code/phuong20052/meta-cxr-stage2-eval/edit> (sau khi push stage 2)

Trong mỗi kernel, mở **Settings → Add-ons → Secrets**, thêm:

| Secret name | Value | Áp dụng |
|-------------|-------|---------|
| `KAGGLE_USERNAME` | `phuong20052` | cả 2 kernel |
| `KAGGLE_KEY` | giá trị `"key"` trong `kaggle.json` | cả 2 kernel |
| `WANDB_API_KEY` | lấy từ <https://wandb.ai/authorize> | cả 2 kernel |
| `HF_TOKEN` | lấy từ <https://huggingface.co/settings/tokens> (cần accept MedGemma terms tại <https://huggingface.co/google/medgemma-1.5-4b-it>) | chỉ stage 2 |

---

## Bước 8 — Chạy stage 1 chính thức (background, ~12h/session)

```bash
cd ~/meta-cxr-cloud/cloud
nohup bash run_stage1.sh > stage1_$(date +%s).log 2>&1 &
```

Theo dõi:

```bash
tail -f stage1_*.log
```

**Output đi đâu**:
- Trên VM: `~/meta-cxr-cloud/cloud/outputs/stage1/<run-id>/`
- Trên GCS: `gs://meta-cxr-checkpoints/stage1/<run-id>/` (gồm `checkpoint_best.pth`, `checkpoint_last.pth`, `manifest.json`)

**Lặp lại bước 8** sau mỗi 12h — notebook Cell 6 tự resume từ `checkpoint_last.pth`. Nhớ:
- Sau session 1: add lại dòng `"phuong20052/meta-cxr-checkpoints"` vào `kernels/train/kernel-metadata.json`.
- Attach dataset `meta-cxr-checkpoints` vào kernel qua Kaggle UI (Add Data) nếu chưa.

---

## Bước 9 — Chạy stage 2 (sau khi stage 1 hội tụ)

```bash
cd ~/meta-cxr-cloud/cloud
nohup bash run_stage2.sh > stage2_$(date +%s).log 2>&1 &
tail -f stage2_*.log
```

Output ở `gs://meta-cxr-checkpoints/stage2/<run-id>/` (gồm `eval_results.json`, sample reports).

---

## Bước 10 — Kiểm tra kết quả

Trên VM hoặc local (cần `gsutil` đã auth):

```bash
gsutil ls gs://meta-cxr-checkpoints/stage1/
gsutil ls gs://meta-cxr-checkpoints/stage2/

# Download checkpoint best về local
gsutil cp gs://meta-cxr-checkpoints/stage1/<RUN_ID>/ckpt/checkpoint_best.pth ./

# Xem manifest
gsutil cat gs://meta-cxr-checkpoints/stage1/<RUN_ID>/manifest.json
```

---

## Troubleshooting

| Triệu chứng | Cách xử lý |
|-------------|------------|
| `kaggle: command not found` | Chạy `bash setup_vm.sh`, hoặc `export PATH="$HOME/.venvs/kaggle-cli/bin:$PATH"` |
| `You must authenticate before...` | Copy `kaggle.json` vào `~/.kaggle/kaggle.json`, rồi chạy `chmod 600 ~/.kaggle/kaggle.json` |
| `kaggle kernels push` báo 403 | Kiểm tra username/key trong `~/.kaggle/kaggle.json`, đảm bảo username trùng `phuong20052` |
| Chạy bằng `sudo` rồi báo thiếu `kaggle` | Không dùng `sudo`; chạy `bash run_stage1.sh` bằng user thường |
| `gsutil` báo `AccessDeniedException` | Service account VM thiếu scope — xem Bước 4 warning |
| Kernel status = `error` | Mở Kaggle web UI, xem log cell nào fail. Hay gặp: thiếu Kaggle Secret, dataset chưa attach |
| Notebook chạy quá 12h | Kaggle tự kill. Chạy lại `run_stage1.sh`, Cell 6 sẽ resume |
| `kaggle datasets download` báo 404 (lần đầu) | Bình thường — Cell 8 sẽ `kaggle datasets create` trong session đó |
