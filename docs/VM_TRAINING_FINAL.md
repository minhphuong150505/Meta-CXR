# VM_TRAINING_FINAL — 2× RTX 3090, 64 GB RAM

Hướng dẫn clone repo và train META-CXR trên VM **2× RTX 3090 (24 GB mỗi card), 64 GB RAM**.

> **Môi trường hiện hành:** user xác nhận đang train trên một host có định danh
> `phuong@minhphuong`. Chưa xác nhận hardware của host đó. Tài liệu này là recipe
> riêng cho 2×3090, không được dùng làm bằng chứng rằng `minhphuong` có cấu hình này;
> hãy chạy `python scripts/vm_preflight.py` trên host trước khi chọn config.

> **Trạng thái (2026-07-22): CPU integration complete; ready for GPU smoke testing.**
> Toàn bộ 465 CPU test pass, config load được, các entrypoint có `--help` (trừ entrypoint
> train cần cài stage-env). **Chưa có bước nào chạy trên GPU trong lần tích hợp này**, nên
> chưa gọi là "full-train-validated" và **không có model metric nào được tái lập**.
> Bổ sung cho `docs/cloud/VM_SPEC.md` (bản đó cho 1× L4 trên GCP).

---

## 0. Ma trận hỗ trợ 2 GPU (đọc trước khi kỳ vọng)

| Component | 1 GPU | 2 GPU DDP | 2 independent jobs | Verified by |
|---|---|---|---|---|
| **Stage 1** (BLIP-2 / Q-Former + MHCAC) | ✅ | ✅ `torchrun --nproc_per_node=2` — `init_distributed_mode` + `DistributedSampler` + checkpoint chỉ ở rank-0 | (không cần) | code inspection (`dist_utils.py`, `runner_base.py`). **Chưa GPU-test** |
| **Stage 2** (MedGemma QLoRA) | ✅ | ❌ **không có DDP** — code ghi rõ *"need DDP, not a wider device_map"* | ✅ chạy 2 experiment độc lập, mỗi card 1 job | code inspection. **Chưa GPU-test** |

Kết luận: **Stage 1 dùng được cả 2 GPU qua DDP thật.** **Stage 2 mỗi run 1 GPU** — muốn dùng
hết 2 card thì chạy 2 run độc lập (`CUDA_VISIBLE_DEVICES=0` và `=1`), không phải DDP.

---

## 1. Python & clone

```bash
# Python >= 3.10 (pyproject yêu cầu). 3.10/3.11/3.12 đều được.
python3 --version

git clone git@github.com:minhphuong150505/Meta-CXR-Kaggle.git
cd Meta-CXR-Kaggle
git fetch --tags
git checkout vm-train-ready-20260722      # tag final (annotated) của bản tích hợp này
```

## 2. Hai môi trường tách biệt (Stage 1 vs Stage 2)

Stage 1 và Stage 2 **pin torch/transformers khác nhau, không cùng resolve được** — tạo 2 venv.

```bash
# --- Stage 1 (BLIP-2, torch 2.5.1 / cu124) ---
python3 -m venv .venv-stage1
source .venv-stage1/bin/activate
pip install -U pip
pip install -r requirements-stage1.txt
deactivate

# --- Stage 2 (MedGemma QLoRA) ---
python3 -m venv .venv-stage2
source .venv-stage2/bin/activate
pip install -U pip
pip install -r requirements-stage2.txt
# tùy chọn: generation metrics (nếu không cài, chúng được báo "unavailable", KHÔNG phải 0)
pip install nltk bert-score pycocoevalcap
deactivate
```

## 3. Hugging Face auth (MedGemma là gated)

```bash
export HF_TOKEN=hf_xxx            # KHÔNG commit token; preflight chỉ báo set/unset
# hoặc: huggingface-cli login
```

## 4. Cấu hình đường dẫn dữ liệu

`local_config.py` đọc `configs/env_config.yaml`. **File mẫu trỏ vào path Kaggle cũ** — phải sửa:

```bash
cp configs/env_config.yaml.example configs/env_config.yaml   # nếu chưa có
# Sửa paths.mimic_cxr_jpg_root -> thư mục CHỨA TRỰC TIẾP files/ (mirror của bucket root).
# image_path trong CSV là RELATIVE (files/p1X/pXXXXXXXX/sYYYYYYY/<dicom>.jpg).
```

### Cấu trúc thư mục dữ liệu cần có

```
<mimic_cxr_jpg_root>/
└── files/
    ├── p10/pXXXXXXXX/sYYYYYYY/<dicom>.jpg
    └── ... p11 ... p19
<processed_dir>/                     # split CSVs từ gs://.../processed/full_allviews/
├── train.csv  (365,293 rows / 220,216 studies)
├── val.csv    (2,963)
└── test.csv   (5,082)
```

> Split CSV phải là bản **có `impression_clean` / `impression_valid` /
> `impression_token_count`** (build sau 2026-07-21), nếu không `--section-mode
> findings_and_impression` (mặc định) sẽ fail ở `assert_columns`.

### Thư mục checkpoint / output

```
pretraining/outputs/<run_name>/   # Stage 1: checkpoint_best.pth, checkpoint_last.pth, checkpoint_{2,5,8}.pth
training/outputs/<dir>/           # Stage 2: LoRA adapter + predictions JSONL
checkpoints/                      # LoRA adapter Vicuna (inference.py)
```

## 5. Preflight (không tải model, chạy được nhiều lần)

```bash
source .venv-stage1/bin/activate
python scripts/vm_preflight.py            # kiểm GPU/VRAM/RAM/disk/dev-shm/paths/imports
python scripts/vm_preflight.py --stage 1  # chỉ check liên quan Stage 1
```

Kiểm nhanh 2 GPU:

```bash
python -c "import torch; print(torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

## 6. Stage 1 — train (Q-Former alignment)

Config 2×3090: `pretraining/configs/mimic_cxr_2x3090.yaml` (effective batch **64** giữ nguyên:
`micro 2 × world_size 2 × accum 16`).

```bash
source .venv-stage1/bin/activate

# --- SMOKE (không có limit flag ở Stage 1): chạy 1 GPU, chờ dataset build + vài optimizer
#     step rồi Ctrl-C. Mục tiêu chỉ là forward/backward/save chạy được, KHÔNG train xong. ---
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --standalone --nproc_per_node=1 \
    -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_full_l4.yaml

# --- FULL, 2 GPU DDP ---
python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_2x3090.yaml
```

- DDP tự bật từ env của `torchrun` (`WORLD_SIZE`/`RANK`/`LOCAL_RANK`); device theo `LOCAL_RANK`,
  **không hardcode `cuda:0`**, tôn trọng `CUDA_VISIBLE_DEVICES`.
- Chỉ **rank-0** ghi checkpoint/log; validation aggregate đúng qua các rank
  (`runner_base.py` xử lý padding của `DistributedSampler`).
- `checkpoint_best.pth` cập nhật khi `f1_positive_macro` (toàn bộ val) cải thiện;
  **test split được giữ riêng**, chỉ eval một lần từ `checkpoint_best` sau khi train xong.

### Resume Stage 1

Không có CLI flag; set trong YAML (hoặc thêm `--options run.resume_ckpt_path=<path>`):

```yaml
run:
  resume_ckpt_path: "pretraining/outputs/<run_name>/checkpoint_last.pth"
```

## 7. Stage 2 — train (MedGemma QLoRA, single-GPU)

Mặc định: `--pipeline-mode medgemma_direct`, `--section-mode findings_and_impression`.

```bash
source .venv-stage2/bin/activate

# --- SMOKE ---
CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py \
    --train-limit 500 --val-limit 10 --test-limit 10 --no-upload \
    --output-dir training/outputs/smoke

# --- FULL (1 GPU) ---
CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py \
    --output-dir training/outputs/medgemma_direct_full --no-upload

# --- Dùng hết 2 GPU = 2 experiment ĐỘC LẬP (không phải DDP) ---
CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py \
    --pipeline-mode medgemma_direct --output-dir training/outputs/direct --no-upload &
CUDA_VISIBLE_DEVICES=1 python training/run_medgemma_qlora.py \
    --pipeline-mode meta_cxr_qformer --section-mode findings_only \
    --checkpoint-root pretraining/outputs --output-dir training/outputs/qformer --no-upload &
wait
```

> Q-Former mode là **FINDINGS-only** (ReportDataset không phát IMPRESSION); truyền
> `--section-mode findings_and_impression` cho mode đó sẽ báo lỗi thay vì đổi target ngầm.

### Resume Stage 2

```bash
python training/run_medgemma_qlora.py --resume-from training/outputs/<dir>/<checkpoint> ...
```

## 8. Evaluate (từ prediction đã lưu — không cần GPU)

```bash
source .venv-stage2/bin/activate   # hoặc bất kỳ env có numpy
# Stage 1 classification: AUROC/AUPRC/positive-macro-F1 + bootstrap CI, all-negative baseline
python scripts/evaluate_stage1.py --predictions <preds.jsonl> --output-dir eval_out/stage1
# Threshold calibration — CHỈ trên validation
python scripts/calibrate_thresholds.py --predictions <val_preds.jsonl> --output thresholds.json
# Stage 2 generation: BLEU/ROUGE/METEOR/CIDEr/BERTScore + error analysis + subgroups
python scripts/evaluate_stage2.py --predictions <gen_preds.jsonl> --output-dir eval_out/stage2
```

> Clinical metrics (CheXbert/RadGraph/RadCliQ/RadFact) **chưa được wire** — adapter raise
> `MissingOptionalDependency`/`NotImplementedError`, không bao giờ báo điểm giả. NLG metric
> thiếu package cũng báo "unavailable", không phải 0.

## 9. RAM / workers / /dev/shm

- `num_workers: 4` **mỗi process DDP** (đã set trong `mimic_cxr_2x3090.yaml`) → 8 worker/2 card
  trên 64 GB. Bắt đầu thận trọng; đo `free -g` + tqdm rate rồi mới tăng.
- `pin_memory` bật cho GPU; `persistent_workers` khi `num_workers>0`; `prefetch_factor ~2`.
- **`/dev/shm`:** DataLoader worker chuyển tensor qua shared memory. shm nhỏ → worker chết với
  lỗi "bus error / shared memory". Nếu chạy trong Docker: `docker run --shm-size=16g ...`.
- **Rủi ro RAM Stage-2:** `build_stage1_records` ghi cache `.pt` ~10–11 GB cho 220k row. Trên
  64 GB thường ổn, nhưng theo dõi `free -g` lần chạy đầu.

## 10. Không mất dữ liệu khi VM dừng

- Stage 1 lưu `checkpoint_last.pth` mỗi epoch + `checkpoint_{2,5,8}.pth` (`save_freq: 3`) →
  resume từ `run.resume_ckpt_path`.
- Stage 2 `train_fine` **chỉ lưu adapter sau epoch cuối** → run dài không có crash-recovery;
  cân nhắc chia nhỏ epoch hoặc chạy trong `tmux`/`nohup` và đẩy artifact lên storage ngay khi xong.
- Đẩy output ra ngoài VM (GCS/rsync) sau mỗi mốc; **đừng để checkpoint chỉ nằm trên boot disk**.
- `tmux new -s train` rồi chạy, để SSH rớt không giết job.

## 11. Troubleshooting OOM

| Triệu chứng | Xử lý |
|---|---|
| CUDA OOM Stage 1 | Giảm `batch_size_train` 2→1 và **tăng `accum_grad_iters` gấp đôi** để giữ effective 64; bật lại `amp: true` (đã bật). |
| CUDA OOM Stage 2 | Giảm `--batch-size` (mặc định 2) và tăng `--grad-accum`; giảm `--max-length` (768). NF4 double-quant đã bật sẵn. |
| Worker "bus error" | Tăng `/dev/shm` (`--shm-size`), hoặc giảm `num_workers`. |
| RAM cạn khi build cache | Giảm `num_workers`; theo dõi `free -g`; để cache ghi xuống đĩa thay vì giữ trong RAM. |
| DDP treo lúc init | Đúng `--standalone` (single-node); kiểm 2 GPU đều thấy bằng `nvidia-smi`. |

---

## 12. Giới hạn đã biết (đọc kỹ)

- **Chưa GPU-test bất kỳ bước nào** trong lần tích hợp này. Ma trận §0 dựa trên đọc code, không
  phải chạy thật. Coi bản này là **GPU-smoke-ready**, chưa phải full-train-validated.
- **Stage 2 không có DDP** — dùng 2 GPU nghĩa là 2 job độc lập.
- **Native multiview:** Stage-1 fuse nhiều view (anchor + 1 aux) là thật; nhưng Stage-2
  `medgemma_direct` truyền ảnh anchor cho MedGemma — kiểm lại có truyền ảnh phụ thật không trước
  khi mô tả là "multi-image Stage-2".
- **Prior/temporal metadata:** guard "no prior" tồn tại trong prompt policy, nhưng dữ liệu prior
  thật có được nạp hay không phụ thuộc split — chưa verify trên VM.
- **Dữ liệu & checkpoint chưa có sẵn trên repo** — split CSV + ảnh mount từ private GCS
  (`gs://mimic-cxr-jpg-dataset-phuongnm`, chỉ account `phuongnm150505@gmail.com` đọc được).
- **Không có model metric nào trong tài liệu này** vì pipeline final chưa chạy thực nghiệm.
