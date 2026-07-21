> **Cap nhat 2026-07-20.** Tai lieu nay da duoc chinh lai cho full dataset
> p10-p19. Cac dinh danh duoi day deu da verify bang `gcloud` va dang song:
>
> | Hang muc | Gia tri |
> |---|---|
> | Project | `$GCP_PROJECT` — dung project dang bat billing |
> | Zone / region | `us-central1-a` / `us-central1` — chon theo bucket data o US multi-region |
> | Data bucket | `gs://$GCS_DATA_BUCKET` (US multi-region, PAP enforced) |
> | Checkpoint bucket | `gs://$GCS_BUCKET` (us-central1, PAP enforced) |
>
> Ban cu dung `gs://meta-cxr-checkpoint`, `gs://mimic-cxr-jpg-data` va project
> `mimic-cxr-jpg-491409` — **da bi xoa hoac tat billing**; moi tham chieu toi chung
> da duoc thay. Khi chay script van nen truyen ten qua `GCS_BUCKET` / `GCP_PROJECT`
> trong `cloud/env.sh` thay vi sua cung trong lenh.

# Setup Google Cloud VM L4 de train META-CXR

File nay ghi lai quy trinh tao may ao Google Cloud va train model truc tiep tren VM GPU, dua tren lan setup `meta-cxr-l4` cho META-CXR.

Muc tieu cu the:
- Tao VM `g2-standard-8` co 1 GPU NVIDIA L4 24GB.
- Dong bo code, data, config len VM.
- Cai dependency Python/system.
- Smoke test import, dry-run nho, roi launch training dai trong `tmux`.
- Log len Weights & Biases va upload checkpoint/log len Google Cloud Storage.

## 0. Thong so da dung

| Hang muc | Gia tri |
|---|---|
| GCP project | `$GCP_PROJECT` |
| VM name | `meta-cxr-l4` |
| Zone | `us-central1-a` |
| Region | `us-central1` |
| Machine type | `g2-standard-8` |
| GPU | `1 x NVIDIA L4 24GB` |
| Boot disk | `200GB pd-ssd` |
| Image | PyTorch GPU image: `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` |
| VM API scope | `cloud-platform` |
| Data bucket | `gs://$GCS_DATA_BUCKET/` |
| Checkpoint bucket | `gs://$GCS_BUCKET/` |
| Remote repo path | `~/META-CXR` |
| Remote data path | `~/data` |
| Remote logs path | `~/logs` |
| Remote output path | `~/output` |
| Wandb entity/project | `$WANDB_ENTITY/meta-cxr-encoder-comparison` |

## 1. Chuan bi local

Can co:
- Google Cloud SDK: `gcloud`
- Da login Google Cloud:

```bash
gcloud auth login
gcloud config set project "$GCP_PROJECT"
```

Enable API neu project moi:

```bash
gcloud services enable compute.googleapis.com storage.googleapis.com
```

Kiem tra quota L4:

```bash
gcloud compute regions describe us-central1 \
  --format="table(quotas.metric,quotas.limit,quotas.usage)"
```

Can quota cho `NVIDIA_L4_GPUS` va CPU trong region `us-central1`.

## 2. Tao bucket checkpoint

Bucket nay de launcher upload checkpoint va log sau moi run.

**Bucket nay da duoc tao san (2026-07-20)** — phan nay giu lai de tai lap khi can.
Kiem tra truoc:

```bash
gcloud storage ls gs://$GCS_BUCKET/
```

Neu can tao lai: `--public-access-prevention` la **bat buoc**, khong phai tuy chon.
`cloud/lib/common.sh:require_private_bucket` va
`training/stage2_utils.py:private_bucket_violations` deu tu choi bucket khong
enforce PAP, nen thieu co nay thi Stage-1 va Stage-2 fail ngay gate dau tien.

```bash
gcloud storage buckets create gs://$GCS_BUCKET \
  --project="$GCP_PROJECT" \
  --location=us-central1 \
  --uniform-bucket-level-access \
  --public-access-prevention
```

## 3. Tao VM GPU L4

Cach de it sai nhat la tao qua Google Cloud Console:

1. Compute Engine -> VM instances -> Create instance.
2. Name: `meta-cxr-l4`.
3. Region/zone: `us-central1-a`.
4. Machine family: GPU.
5. Machine type: `g2-standard-8`.
6. GPU: `NVIDIA L4`, count `1`.
7. Boot disk: Deep Learning VM / PyTorch GPU image, Ubuntu 22.04, CUDA 12.9, NVIDIA 580.
8. Boot disk size: `200GB`, type `pd-ssd`.
9. Identity and API access: set access scopes to **Allow full access to all Cloud APIs**.
10. Create.

CLI template neu muon tao bang terminal:

```bash
# Tim image PyTorch GPU phu hop truoc, vi ten image co the thay doi theo thoi gian.
gcloud compute images list \
  --project=deeplearning-platform-release \
  --filter='name~pytorch AND name~ubuntu-2204' \
  --limit=20

# Sau do thay IMAGE_NAME bang image thuc te.
IMAGE_NAME="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"

gcloud compute instances create meta-cxr-l4 \
  --project="$GCP_PROJECT" \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --image="$IMAGE_NAME" \
  --image-project=deeplearning-platform-release \
  --scopes=cloud-platform
```

Kiem tra VM:

```bash
gcloud compute instances describe meta-cxr-l4 \
  --zone=us-central1-a \
  --format='value(status,networkInterfaces[0].accessConfigs[0].natIP)'
```

SSH:

```bash
gcloud compute ssh meta-cxr-l4 --zone=us-central1-a
```

Tren VM, kiem tra GPU:

```bash
nvidia-smi
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## 4. Upload code len VM

Tu may local:

```bash
cd /home/phuong/Documents/KLTN/META_CXR_again

tar --exclude='.git' \
    --exclude='wandb' \
    --exclude='__pycache__' \
    -czf /tmp/META-CXR.tar.gz META-CXR

gcloud compute scp /tmp/META-CXR.tar.gz \
  meta-cxr-l4:~/META-CXR.tar.gz \
  --zone=us-central1-a

gcloud compute ssh meta-cxr-l4 --zone=us-central1-a --command='
rm -rf ~/META-CXR
tar -xzf ~/META-CXR.tar.gz -C ~
'
```

Neu chi sua mot file va muon sync nhanh:

```bash
gcloud compute scp META-CXR/pretraining/train.py \
  meta-cxr-l4:~/META-CXR/pretraining/train.py \
  --zone=us-central1-a
```

## 5. Dong bo data tu GCS ve VM

Full dataset (p10-p19) la ~571 GiB anh, khong vua boot disk 200GB. Gan them mot
Persistent Disk rieng va mirror nguyen bucket vao do.

> **Chon region theo vi tri bucket data.** Bucket data nam o **US multi-region**
> (kiem tra: `gcloud storage buckets describe gs://$GCS_DATA_BUCKET`). Dat VM va PD
> trong mot region US -- guide nay dung `us-central1-a` -- thi 571 GiB di trong cung
> multi-region va khong tinh phi egress. Ban cu dat VM o `asia-southeast1-c`: keo qua
> chau luc vua cham vua ton egress (~0.12 USD/GB, khoang 70 USD).

> **Kiem tra quota SSD truoc.** `pd-ssd` va `pd-balanced` deu tinh vao
> `SSD_TOTAL_GB`, mac dinh chi **500GB/region** (da verify:
> us-central1/us-east1/us-west1/us-west4 deu = 500).
> Boot disk 200GB pd-ssd + data disk 800GB pd-balanced = 1000GB, **vuot quota**.
> Chon mot trong hai:
> - Xin nang `SSD_TOTAL_GB` len >= 1100GB (Console > IAM & Admin > Quotas), hoac
> - Dung `--type=pd-standard` cho data disk: tinh vao `DISKS_TOTAL_GB` (limit
>   4096GB, du) va re hon, doi lai HDD doc cham hon dang ke voi 377k file JPEG nho
>   — dang ke vi feature extraction chay `batch_size=1`, `num_workers=2`.
>
> Quota GPU da du: `NVIDIA_L4_GPUS = 1` o ca 4 region US, dung bang nhu cau. Khong
> can xin them.

```bash
# Tao + gan PD 800GB (chay tu may local). Doi --type=pd-standard neu chua nang quota SSD.
gcloud compute disks create meta-cxr-data \
  --size=800GB --type=pd-balanced --zone=us-central1-a

gcloud compute instances attach-disk meta-cxr-l4 \
  --disk=meta-cxr-data --zone=us-central1-a
```

Tren VM, format va mount lan dau:

```bash
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard \
  /dev/disk/by-id/google-meta-cxr-data
sudo mkdir -p /mnt/disks/mimic
sudo mount -o discard,defaults /dev/disk/by-id/google-meta-cxr-data /mnt/disks/mimic
sudo chown -R "$USER":"$USER" /mnt/disks/mimic

# Tu dong mount lai sau khi reboot
echo "/dev/disk/by-id/google-meta-cxr-data /mnt/disks/mimic ext4 discard,defaults,nofail 0 2" \
  | sudo tee -a /etc/fstab
```

Mirror bucket. **Giu nguyen tien to `files/`** — `image_path` trong processed CSV la
duong dan tuong doi (`files/p1X/...`) va `ReportDataset._row_visual` noi no vao
`mimic_cxr_jpg_root`; mat tien to nay thi moi lookup anh deu truot.

```bash
mkdir -p ~/logs
BUCKET=gs://$GCS_DATA_BUCKET

# ~571 GiB, chay trong tmux — mat nhieu gio
gcloud storage rsync -r "$BUCKET/files"     /mnt/disks/mimic/files
gcloud storage rsync -r "$BUCKET/processed" /mnt/disks/mimic/processed
gcloud storage cp "$BUCKET"/mimic-cxr-2.0.0-*.csv.gz /mnt/disks/mimic/

# Chi can neu chay lai preprocessing; training doc tu processed/ nen co the bo qua
gcloud storage rsync -r "$BUCKET/mimic-cxr-reports" /mnt/disks/mimic/mimic-cxr-reports

touch ~/logs/sync_done.marker
du -sh /mnt/disks/mimic/*
```

Kiem tra truoc khi train — so dong phai khop split da verify
(train 365,293 / val 2,963 / test 5,082):

```bash
wc -l /mnt/disks/mimic/processed/full_allviews/{train,val,test}.csv
ls /mnt/disks/mimic/files | head    # phai thay p10 .. p19
```

## 6. Tao `configs/env_config.yaml` tren VM

File nay map code voi data path tren VM. `data_root` la dong duy nhat can doi neu
ban mount o cho khac.

```bash
cat > ~/META-CXR/configs/env_config.yaml <<'YAML'
paths:
  data_root: "/mnt/disks/mimic"

  mimic_cxr_jpg_root: "${paths.data_root}"

  # pandas doc .csv.gz truc tiep, khong can giai nen
  split_csv:    "${paths.data_root}/mimic-cxr-2.0.0-split.csv.gz"
  chexpert_csv: "${paths.data_root}/mimic-cxr-2.0.0-chexpert.csv.gz"
  metadata_csv: "${paths.data_root}/mimic-cxr-2.0.0-metadata.csv.gz"
  # File nay khong ton tai trong bucket va khong code nao doc no; key chi can co
  # mat de local_config.py khong KeyError.
  reports_csv:  "${paths.data_root}/mimic_cxr_cleaned.csv"

  processed_dir:       "${paths.data_root}/processed/full_allviews"
  processed_train_csv: "${paths.processed_dir}/train.csv"
  processed_val_csv:   "${paths.processed_dir}/val.csv"
  processed_test_csv:  "${paths.processed_dir}/test.csv"

  output_dir: "/home/phuong/output"
  checkpoint_dir: "/home/phuong/checkpoints"

  # Heredoc nay quoted ('YAML') nen KHONG expand bien shell -- dien ten that vao
  # hai dong duoi, trung voi GCS_BUCKET / GCP_PROJECT ban export o cloud/env.local.sh.
  # Bucket phai co UBLA + public-access prevention enforced.
  gcs_bucket:  "gs://DIEN-TEN-BUCKET-CHECKPOINT"
  gcs_project: "DIEN-GCP-PROJECT-ID"
wandb:
  entity: "phuongnm150505-uit"
  project: "meta-cxr-encoder-comparison"
java:
  home: "/usr/lib/jvm/java-11-openjdk-amd64"
  path: "/usr/lib/jvm/java-11-openjdk-amd64/bin:"
YAML
```

Verify config resolve dung truoc khi train:

```bash
cd ~/META-CXR && python -c "
from local_config import PROCESSED_TRAIN_CSV, VIS_ROOT
import os
print(PROCESSED_TRAIN_CSV, os.path.isfile(PROCESSED_TRAIN_CSV))
print(VIS_ROOT, os.path.isdir(os.path.join(VIS_ROOT, 'files')))
"
```

## 7. Cai dependency

Tren VM:

```bash
sudo apt-get update -qq
sudo apt-get install -y \
  python3-pip python3-venv \
  openjdk-11-jdk \
  libgl1 libglib2.0-0

python3 -m pip install --upgrade pip
python3 -m pip install \
  "numpy<2" \
  "opencv-python<4.10" \
  "transformers==4.30.2" \
  wandb \
  omegaconf==2.3.0 \
  iopath \
  timm \
  pandas \
  scikit-image \
  accelerate \
  sentencepiece \
  protobuf \
  iterative-stratification \
  einops \
  fairscale \
  pycocoevalcap \
  webdataset \
  decord \
  ftfy \
  regex \
  hi-ml-multimodal \
  bitsandbytes \
  datasets \
  nltk \
  pytorch_lightning \
  torchinfo \
  fire \
  loralib \
  peft

# Mot so script/code cu goi "python", nen tao symlink neu may chi co python3.
sudo ln -sf "$(which python3)" /usr/local/bin/python
```

Ly do cac pin quan trong:
- `transformers==4.30.2`: LAVIS fork dung API cu, ban 5.x gay loi `apply_chunking_to_forward`.
- `numpy<2`: tranh loi ABI voi cac package cu.
- `opencv-python<4.10` + `libgl1`: tranh loi `ImportError: libGL.so.1`.
- `hi-ml-multimodal`: cung cap module `health_multimodal`.

## 8. Login Wandb

Tren VM:

```bash
wandb login
```

Khong paste API key vao chat/log public. Neu key bi lo, revoke tai:

```text
https://wandb.ai/authorize
```

## 9. Chuan bi 7 config encoder comparison

Thu muc:

```text
~/META-CXR/pretraining/configs/encoder_comparison/
```

7 file:

```text
01_biovil_only.yaml
02_pubmedclip_only.yaml
03_swin_only.yaml
04_biovil_pubmedclip.yaml
05_biovil_swin.yaml
06_pubmedclip_swin.yaml
07_all_three.yaml
```

Hyperparameter chinh da dung:

```yaml
run:
  task: image_text_pretrain_eval
  project_name: meta-cxr-encoder-comparison
  wandb_entity: phuongnm150505-uit
  max_epoch: 30
  early_stop_patience: 5
  early_stop_min_delta: 1e-4
  batch_size_train: 4
  batch_size_eval: 4
  accum_grad_iters: 4
  num_workers: 2
  warmup_steps: 1000
  amp: True
  save_freq: 5
  init_lr: 5e-5
  init_lr_q: 5e-5
  init_lr_cls: 5e-5
  min_lr: 1e-6
  weight_decay: 0.02
  lr_sched: linear_warmup_cosine_lr
```

Effective batch size:

```text
batch_size_train 4 x accum_grad_iters 4 = 16
```

## 10. Smoke test truoc khi train

Chay tren local qua SSH:

```bash
gcloud compute ssh meta-cxr-l4 --zone=us-central1-a --command='
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

Ket qua can thay:

```text
cuda= True
Dataset OK
Blip2Qformer OK
SMOKE_PASS
```

## 11. Dry-run nho de test pipeline

Nen chay dry-run truoc khi launch job dai. Trong code hien tai da them ho tro `truncate_train/val/test`, giup test vai sample.

Tao config tam tren VM:

```bash
gcloud compute ssh meta-cxr-l4 --zone=us-central1-a --command='
cp ~/META-CXR/pretraining/configs/encoder_comparison/01_biovil_only.yaml /tmp/meta_cxr_dryrun.yaml
python - <<'"'"'PY'"'"'
from pathlib import Path
p = Path("/tmp/meta_cxr_dryrun.yaml")
s = p.read_text()
s = s.replace("run_name: 01_biovil_only", "run_name: dryrun_biovil_only")
s = s.replace("max_epoch: 30", "max_epoch: 1")
s = s.replace("batch_size_train: 4", "batch_size_train: 2")
s = s.replace("batch_size_eval: 4", "batch_size_eval: 2")
s = s.replace("num_workers: 2", "num_workers: 0")
s = s.replace("warmup_steps: 1000", "warmup_steps: 10")
s = s.replace("accum_grad_iters: 4", "accum_grad_iters: 1")
s = s.replace("output_dir: \"/home/phuong/output/01_biovil_only\"", "output_dir: \"/home/phuong/output/dryrun_biovil_only\"")
s = s.replace("valid_splits: [\"val\"]", "valid_splits: []")
s = s.replace("test_splits: [\"test\"]", "test_splits: []")
s += "\n  log_freq: 1\n  truncate_train: 8\n  truncate_val: 8\n  truncate_test: 8\n"
p.write_text(s)
PY
'
```

Chay dry-run:

```bash
gcloud compute ssh meta-cxr-l4 --zone=us-central1-a --command='
cd ~/META-CXR
WANDB_MODE=offline \
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m pretraining.train --cfg-path /tmp/meta_cxr_dryrun.yaml 2>&1 | tee /tmp/dryrun.log
'
```

Lan setup nay dry-run pass:

```text
8 train samples
4 train steps
checkpoint_last.pth saved
peak GPU ~5GB
```

## 12. Launcher full 7 runs

File launcher:

```text
~/META-CXR/cloud/run_encoder_comparison.sh
```

No chay tuan tu:

```text
01_biovil_only
02_pubmedclip_only
03_swin_only
04_biovil_pubmedclip
05_biovil_swin
06_pubmedclip_swin
07_all_three
```

Sau moi run:
- Log nam o `~/logs/{run_name}.log`.
- Output nam o `~/output/{run_name}`.
- Upload len `gs://$GCS_BUCKET/{run_name}/`.

Launch trong `tmux`:

```bash
gcloud compute ssh meta-cxr-l4 --zone=us-central1-a --command='
rm -rf ~/output/0[1-7]_* ~/logs/0[1-7]_*.log ~/logs/master_run.log
tmux kill-session -t train 2>/dev/null || true
tmux new -d -s train "bash ~/META-CXR/cloud/run_encoder_comparison.sh 2>&1 | tee ~/logs/master_run.log"
sleep 3
tmux ls
'
```

## 13. Monitor training

SSH vao VM:

```bash
gcloud compute ssh meta-cxr-l4 --zone=us-central1-a
```

Attach tmux:

```bash
tmux attach -t train
```

Detach tmux:

```text
Ctrl+B, roi bam D
```

Xem GPU:

```bash
nvidia-smi -l 5
```

Xem log:

```bash
tail -f ~/logs/master_run.log
tail -f ~/logs/01_biovil_only.log
```

Kiem tra process:

```bash
ps -eo pid,ppid,cmd | grep -E "pretraining.train|torch.distributed.run|run_encoder_comparison" | grep -v grep
```

Kiem tra bucket:

```bash
gcloud storage ls -r gs://$GCS_BUCKET/ | head -50
```

Wandb:

```text
https://wandb.ai/phuongnm150505-uit/meta-cxr-encoder-comparison
```

## 14. Uoc tinh thoi gian

Run dau `01_biovil_only` da do duoc:

```text
4506 iters/epoch
~0.33-0.35 giay/iter
~25-27 phut train/epoch
30 epochs => ~12.5-13.5 gio rieng train
```

Co validate `val` va `test` cuoi moi epoch:

```text
train: 18024 records
val:    2121 records
test:   2042 records
```

Uoc tinh:
- `01_biovil_only`: khoang 15-20 gio.
- Single encoder runs: khoang 15-20 gio/run.
- Pair encoder runs: co the 20-35 gio/run.
- All-three: co the 30-45 gio.
- Tong 7 runs tuan tu: khoang 5-8 ngay.

## 15. Loi da gap va cach fix

| Loi | Nguyen nhan | Fix |
|---|---|---|
| `ModuleNotFoundError: health_multimodal` | Thieu package BioViL/helper | `pip install hi-ml-multimodal` |
| `ImportError: apply_chunking_to_forward` | `transformers` qua moi | `pip install transformers==4.30.2` |
| `ImportError: libGL.so.1` | OpenCV can system lib | `sudo apt-get install -y libgl1 libglib2.0-0` |
| `opencv-python>=4.10` doi `numpy>=2` | Xung dot voi code/package cu | `pip install "numpy<2" "opencv-python<4.10"` |
| Smoke gate pass sai | Script grep chuoi in qua som | Chi pass khi grep `SMOKE_PASS` |
| `prefetch_factor` voi `num_workers: 0` | PyTorch DataLoader khong cho | Chi truyen `prefetch_factor` khi `num_workers > 0` |
| L4 OOM voi pair/all-three | Qua tai VRAM | Giam `batch_size_train: 2`, tang `accum_grad_iters: 8` de giu effective batch 16 |

## 15.1. Early stopping

Code hien tai co early stopping theo `val loss`.

Config trong 7 YAML:

```yaml
run:
  max_epoch: 30
  early_stop_patience: 5
  early_stop_min_delta: 1e-4
```

Y nghia:
- Van cho phep chay toi da 30 epochs.
- Sau moi epoch, model evaluate tren `val`.
- Neu `val loss` khong giam it nhat `0.0001` trong 5 epochs lien tiep, run dung som.
- `checkpoint_best.pth` duoc luu tai epoch co `val loss` tot nhat.
- `checkpoint_last.pth` van duoc luu moi epoch de resume.

Log khi dung som se co dong:

```text
Early stopping at epoch ...
```

## 16. Stop/start/delete VM

Dung xong phai stop VM de tranh ton tien:

```bash
gcloud compute instances stop meta-cxr-l4 --zone=us-central1-a
```

Start lai:

```bash
gcloud compute instances start meta-cxr-l4 --zone=us-central1-a
```

Delete hoan toan VM va disk:

```bash
gcloud compute instances delete meta-cxr-l4 --zone=us-central1-a
```

Can than: delete se mat data local tren disk VM neu chua upload len GCS.

## 17. Tai checkpoint ve local

Sau khi train xong:

```bash
cd /home/phuong/Documents/KLTN/META_CXR_again
gcloud storage cp -r gs://$GCS_BUCKET/ ./checkpoints_from_vm/
```

## 18. Checklist ngan gon cho lan sau

1. Login `gcloud`, set project.
2. Tao bucket checkpoint.
3. Tao VM L4 voi disk 200GB va scope `cloud-platform`.
4. Upload `META-CXR` len `~/META-CXR`.
5. Sync data ve `~/data`.
6. Tao `configs/env_config.yaml`.
7. Cai apt libs va Python deps.
8. `wandb login`.
9. Smoke test import.
10. Dry-run nho.
11. Launch full trong `tmux`.
12. Monitor wandb/log/GPU.
13. Stop VM khi xong.
