# Handoff: META-CXR 7-encoder training on GCP L4 VM

**Mục tiêu**: chạy 7 lần train BLIP2 stage-1 (encoders + Q-Former + MHCAC) trên GCP L4 VM với 7 combo encoder (3 single + 3 pair + 1 all-three). Log wandb, checkpoint upload `gs://meta-cxr-checkpoint/{run_name}/`. **30 epochs/run, bs=4 grad_accum=4 (eff bs=16)**.

---

## Cập nhật 2026-05-22 01:10 ICT

- ✅ Đã cài `libgl1 libglib2.0-0` trên VM, lỗi `ImportError: libGL.so.1` đã hết.
- ✅ Đã sửa `~/wait_and_run.sh`: smoke gate giờ check `SMOKE_PASS`, sau khi import `Blip2Qformer` và đọc CSV thành công.
- ✅ Smoke test đầy đủ pass: CUDA OK, `MIMIC_CXR_Dataset` OK, `Blip2Qformer` OK.
- ✅ Đã thêm hỗ trợ `run.truncate_train/val/test` trong `pretraining/train.py` để dry-run nhỏ không cần chạy hết split.
- ✅ Đã sửa `runner_base.py` để `num_workers: 0` không truyền `prefetch_factor`.
- ✅ Dry-run tạm `/tmp/meta_cxr_dryrun.yaml` pass: 8 mẫu train, 4 bước, checkpoint lưu ở `/home/phuong/output/dryrun_biovil_only/.../checkpoint_last.pth`, peak GPU khoảng 5 GB.
- ⏸️ Chưa launch full 7 runs trong tmux để tránh bắt đầu job nhiều ngày khi chưa xác nhận.

## Cập nhật 2026-05-22

- ✅ Đã thêm early stopping theo `val loss` trong `model/lavis/runners/runner_base.py`.
- ✅ 7 config `pretraining/configs/encoder_comparison/*.yaml` đã bật:
  - `early_stop_patience: 5`
  - `early_stop_min_delta: 1e-4`
- ✅ Code/config early stopping đã sync lên VM.
- ⏹️ VM `meta-cxr-l4` đã stop, trạng thái `TERMINATED`. Run đang chạy đã bị ngắt trước khi hết epoch 0, nên chưa có checkpoint epoch đầy đủ cho run hiện tại.

---

## 1. Đã làm xong

### Infrastructure
- ✅ VM `meta-cxr-l4` (zone `asia-southeast1-c`, on-demand, g2-standard-8 + 1×L4 24GB, 200GB pd-ssd, image `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`)
- ✅ External IP: `35.185.191.111`
- ✅ Service account scope `cloud-platform` (đọc/ghi GCS được)

### Code + Configs
- ✅ `META-CXR/` đã upload + extract tại `~/META-CXR/` trên VM
- ✅ **7 YAML configs** đã tạo: `META-CXR/pretraining/configs/encoder_comparison/0[1-7]_*.yaml` (vary `model.encoders.{biovil,pubmedclip,swin}` flags)
- ✅ **Launcher**: `META-CXR/cloud/run_encoder_comparison.sh` — chạy tuần tự 7 configs, upload ckpt + log lên `gs://meta-cxr-checkpoint/`
- ✅ **Wrapper**: `~/wait_and_run.sh` trên VM — wait sync → smoke test → launch; smoke gate đã sửa để check `SMOKE_PASS`
- ✅ `~/META-CXR/configs/env_config.yaml` — points to `/home/phuong/data/{p10,processed,csv,report_p10}` + wandb entity `phuongnm150505-uit` project `meta-cxr-encoder-comparison`

### Data
- ✅ Sync xong từ `gs://mimic-cxr-jpg-data/` → `~/data/`:
  - `p10/`: 56 GB (40K+ images)
  - `processed/`: 47 MB (`train.csv`, `val.csv`, `test.csv`)
  - `report_p10/`: 112 MB
  - `csv/`: 105 MB (split, chexpert, metadata, mimic_cxr_cleaned)
- ✅ `~/logs/sync_done.marker` tồn tại

### Wandb
- ✅ Login với entity `phuongnm150505-uit`. ⚠️ **API key đã lộ trong chat trước** — vào https://wandb.ai/authorize revoke + tạo key mới sau.

### Dependencies đã cài
- ✅ `wandb`, `omegaconf==2.3.0`, `iopath`, `timm`, `pandas`, `scikit-image`, `accelerate`, `sentencepiece`, `protobuf`, `opencv-python<4.10`, `iterative-stratification`, `einops`, `fairscale`, `pycocoevalcap`, `webdataset`, `decord`, `ftfy`, `regex`, `numpy<2`
- ✅ `hi-ml-multimodal` (provides `health_multimodal`)
- ✅ `transformers==4.30.2` (downgrade từ 5.9 — LAVIS fork dùng API cũ)
- ✅ `bitsandbytes`, `datasets`, `nltk`, `pytorch_lightning`, `torchinfo`, `fire`, `loralib`, `peft`
- ✅ `openjdk-11` (cho CheXpert labeler)
- ✅ symlink `python → python3` ở `/usr/local/bin/`

---

## 2. Trạng thái hiện tại — SMOKE + DRY-RUN OK, CHƯA LAUNCH FULL

Sequence lỗi đã gặp khi smoke test:
1. ❌ `ModuleNotFoundError: No module named 'health_multimodal'` → **FIXED** (pip install hi-ml-multimodal)
2. ❌ `ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'` → **FIXED** (downgrade transformers 5.9 → 4.30.2)
3. ❌ `ImportError: libGL.so.1: cannot open shared object file` → **FIXED** (`apt install libgl1 libglib2.0-0`)
4. ❌ `ValueError: prefetch_factor option could only be specified in multiprocessing` khi dry-run `num_workers: 0` → **FIXED** trong `runner_base.py`

Trạng thái mới:
- Smoke import pass.
- Dry-run nhỏ pass.
- GPU đang rảnh sau dry-run.
- Full 7-run training chưa được start.

---

## 3. Việc cần làm tiếp

### 3.1 Fix libGL — DONE
```bash
gcloud compute ssh meta-cxr-l4 --zone=asia-southeast1-c --command='
sudo apt-get update -qq && sudo apt-get install -y libgl1 libglib2.0-0
'
```

### 3.2 Fix smoke test gate logic — DONE
File trên VM: `~/wait_and_run.sh`. Đã đổi gate từ `grep -q "Core libs OK"` sang `grep -q "SMOKE_PASS"` và thêm import `Blip2Qformer` vào smoke test.

### 3.3 Smoke test đầy đủ — DONE
```bash
gcloud compute ssh meta-cxr-l4 --zone=asia-southeast1-c --command='
cd ~/META-CXR
python -c "
import sys
sys.path.insert(0, \".\")
sys.path.insert(0, \"model\")
import torch
print(\"torch=\", torch.__version__, \"cuda=\", torch.cuda.is_available())
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset
print(\"Dataset OK\")
from model.lavis.models.blip2_models.blip2_qformer import Blip2Qformer
print(\"Blip2Qformer OK\")
print(\"SMOKE_PASS\")
"
'
```
Kết quả mới: pass với `torch=2.9.1+cu129`, CUDA OK, `Dataset OK`, `Blip2Qformer OK`, `SMOKE_PASS`.

### 3.4 Dry-run ngắn trên config nhẹ nhất — DONE
Đã chạy dry-run tạm `/tmp/meta_cxr_dryrun.yaml` với `truncate_train: 8`, `batch_size_train: 2`, `num_workers: 0`, không validate/test. Kết quả: 4 bước train pass, checkpoint tạm lưu OK, peak GPU khoảng 5 GB.

Nếu muốn chạy lại dry-run:
```bash
gcloud compute ssh meta-cxr-l4 --zone=asia-southeast1-c --command='
cd ~/META-CXR
WANDB_MODE=offline \
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m pretraining.train --cfg-path /tmp/meta_cxr_dryrun.yaml 2>&1 | tee /tmp/dryrun.log
'
```
Quan sát:
- Có load data OK không (`MIMIC_CXR_Dataset` đọc CSV + ảnh)
- VRAM peak (xem `nvidia-smi -l 5` trong terminal khác) — nếu >22GB, giảm `batch_size_train`
- Có forward/backward + optimizer step không
- Có lưu `checkpoint_last.pth` không

Nếu OK → launch full.

### 3.5 Launch full 7 runs trong tmux
```bash
gcloud compute ssh meta-cxr-l4 --zone=asia-southeast1-c --command='
rm -rf ~/output/0[1-7]_* ~/logs/0[1-7]_*.log
tmux kill-session -t train 2>/dev/null || true
tmux new -d -s train "bash ~/META-CXR/cloud/run_encoder_comparison.sh 2>&1 | tee ~/logs/master_run.log"
sleep 2; tmux ls
'
```
Bỏ qua `wait_and_run.sh` (sync đã xong, không cần wait nữa).

### 3.6 Monitor
- Wandb: https://wandb.ai/phuongnm150505-uit/meta-cxr-encoder-comparison
- SSH: `tmux attach -t train` (Ctrl+B D để detach)
- Log từng run: `tail -f ~/logs/0X_*.log`

### 3.7 Khi train xong (3-6 ngày)
```bash
gcloud compute instances stop meta-cxr-l4 --zone=asia-southeast1-c
```
Checkpoint đã trên `gs://meta-cxr-checkpoint/`. Tải về local:
```bash
gcloud storage cp -r gs://meta-cxr-checkpoint/ ./checkpoints_from_vm/
```

---

## 4. Lưu ý / rủi ro đã thấy

| Vấn đề | Workaround |
|--------|-----------|
| `numpy<2` cần thiết, nhưng `opencv-python>=4.10` cần `numpy>=2` | Đã pin `opencv-python<4.10` |
| `transformers` mới (5.x) phá API | Đã pin `4.30.2` |
| LAVIS fork rất cũ, nhiều `ImportError` ẩn | Fix dần — có thể còn 2-5 lỗi nữa khi import `Blip2Qformer` |
| L4 24GB có thể OOM với 3 encoder + Q-Former + bs=4×accum=4 | Giảm bs=2 grad_accum=8, hoặc tắt 1 encoder |
| On-demand $0.70/h = ~$50-100 cho 7 runs | Nhớ `gcloud instances stop` khi xong |
| Wandb API key đã lộ trong chat | **REVOKE** ở https://wandb.ai/authorize sau khi dùng xong |

---

## 5. File quan trọng

**Trên máy local** (`/home/phuong/Documents/KLTN/META_CXR_again/`):
- `META-CXR/cloud/run_encoder_comparison.sh`
- `META-CXR/cloud/HANDOFF_L4_TRAINING.md` (file này)
- `META-CXR/pretraining/configs/encoder_comparison/0[1-7]_*.yaml` (7 configs)

**Trên VM** (`~`):
- `~/META-CXR/` (full repo)
- `~/META-CXR/configs/env_config.yaml` (paths)
- `~/wait_and_run.sh` (wrapper — KHÔNG dùng nữa vì gate bug, dùng cách 3.5)
- `~/data/` (60GB synced)
- `~/output/` (kết quả train, sẽ được upload lên GCS)
- `~/logs/` (logs)

---

## 6. Commands cheat sheet

```bash
# SSH
gcloud compute ssh meta-cxr-l4 --zone=asia-southeast1-c

# Stop VM
gcloud compute instances stop meta-cxr-l4 --zone=asia-southeast1-c

# Start lại
gcloud compute instances start meta-cxr-l4 --zone=asia-southeast1-c

# Delete hoàn toàn (mất disk + data)
gcloud compute instances delete meta-cxr-l4 --zone=asia-southeast1-c

# Check tmux trên VM
tmux attach -t train

# Watch GPU
nvidia-smi -l 5

# Check bucket
gcloud storage ls -r gs://meta-cxr-checkpoint/ | head -50
```

---

## 7. Hướng dẫn cho session mới

Khi mở session mới hand-off, **paste prompt sau cho Claude**:

> Tôi đang setup train META-CXR trên GCP L4 VM `meta-cxr-l4` (zone asia-southeast1-c).
> Đọc file `META-CXR/cloud/HANDOFF_L4_TRAINING.md` để hiểu trạng thái.
> Việc cần làm theo thứ tự:
> 1. Run mục 3.1 (apt install libGL)
> 2. Run mục 3.3 (smoke test) — fix dần các ImportError còn lại
> 3. Run mục 3.4 (dry-run 1 epoch) — verify pipeline + VRAM
> 4. Run mục 3.5 (launch full 7 runs trong tmux)
> Sau bước 4 thì exit session, tôi tự monitor wandb. Cố gắng dùng càng ít SSH calls càng tốt — batch lệnh.

**Budget cảnh báo**: Session vừa rồi đốt ~$265 cho setup + 2 fix lỗi. Session mới nên dùng Sonnet (không phải Opus) cho debug deps để giảm cost. `/model sonnet-4-6` trước khi bắt đầu.
