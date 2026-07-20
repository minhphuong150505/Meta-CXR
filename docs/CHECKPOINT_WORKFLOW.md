> **KHÔNG CÒN HIỆU LỰC.** Phần auto-push checkpoint lên Kaggle Dataset đã bị gỡ —
> checkpoint huấn luyện trên MIMIC-CXR là dẫn xuất của dữ liệu credentialed và chỉ
> được lưu ở bucket GCS riêng tư. Cơ chế `save_freq` / `checkpoint_best.pth` /
> `run.resume_ckpt_path` mô tả bên dưới vẫn đúng; chỉ phần đích lưu trữ là sai.

# Plan: Train 2 sessions với checkpoint mỗi 3 epoch + auto-push lên Kaggle Dataset

## Context

User muốn:
1. Chia training thành 2 Kaggle session (mỗi session ≤12h) thay vì chạy liên tục
2. Lưu checkpoint **mỗi 3 epoch** (không phải mỗi epoch như hiện tại) → giữ lịch sử (epoch 2, 5, 8, ...)
3. Sau mỗi session, **auto-push checkpoints lên một Kaggle Dataset** qua Kaggle API trong cell notebook
4. Session 2 attach dataset đó → resume từ checkpoint mới nhất

**Vấn đề hiện tại:**
- `runner_base.py:validate()` save checkpoint **mỗi epoch** với filename `checkpoint_{cur_epoch}.pth` → 5 epoch = ~5GB. Đây là root cause của crash `iostream error / unexpected pos` (disk full trên `/kaggle/working/`).
- Không có mechanism push lên Kaggle Dataset → user phải download/upload thủ công

**Hành vi `validate()` hiện tại** (`runner_base.py:398–479`):
- Best metric → `_save_checkpoint(is_best=True)` → `checkpoint_best.pth` (overwrite OK)
- Non-best metric → `_save_checkpoint(is_best=False, is_last=True)` → `checkpoint_last.pth` (line 437, OK)
- Best loss → `_save_checkpoint(is_best=True)` → `checkpoint_best.pth` (OK)
- **Non-best loss → `_save_checkpoint(is_best=False)` → `checkpoint_{cur_epoch}.pth`** (line 472, đầy disk)
- **No-val branch → `_save_checkpoint(is_best=False)` → `checkpoint_{cur_epoch}.pth`** (line 477, đầy disk)

## Approach

### Phần 1: Save checkpoint mỗi 3 epoch

Thêm property `save_freq` (default 1) vào `RunnerBase`. Trong `validate()`, các nhánh non-best chỉ save khi `(cur_epoch + 1) % save_freq == 0`. Filename giữ nguyên dạng `checkpoint_{cur_epoch}.pth` để có lịch sử.

**Sửa `META-CXR/model/lavis/runners/runner_base.py`:**

1. Thêm property sau dòng 369 (cạnh `resume_ckpt_path`):
   ```python
   @property
   def save_freq(self):
       return self.config.run_cfg.get("save_freq", 1)
   ```

2. Sửa nhánh non-best ở line 472 (loss path, dataset có loss val):
   ```python
   else:
       if (cur_epoch + 1) % self.save_freq == 0:
           self._save_checkpoint(cur_epoch, is_best=False)
   ```

3. Sửa nhánh no-val ở line 477:
   ```python
   if not self.evaluate_only:
       if (cur_epoch + 1) % self.save_freq == 0:
           self._save_checkpoint(cur_epoch, is_best=False)
   ```

4. Nhánh agg_metrics line 437 (`is_last=True`): giữ nguyên — `checkpoint_last.pth` overwrite mỗi epoch không tốn disk thêm.

**Sửa `META-CXR/pretraining/configs/mimic_cxr_2gpu.yaml`:**

Thêm vào `run:` block:
```yaml
save_freq: 3
```

### Phần 2: Auto-push lên Kaggle Dataset

Thêm slug dataset checkpoint vào config trung tâm (đồng nhất với pattern user đã chọn: không hardcode):

**Sửa `META-CXR/configs/kaggle_datasets.yaml`** — thêm:
```yaml
datasets:
  ...
  checkpoints:
    slug: "meta-cxr-checkpoints"          # User pre-create dataset rỗng trên Kaggle, hoặc cell sẽ auto-create lần đầu
    title: "META-CXR training checkpoints"
    license: "CC0-1.0"
```

**Thêm cell mới vào `META-CXR/META_CXR_kaggle.ipynb`** (sau cell training, trước cell teardown):

```python
# Cell: Push checkpoints lên Kaggle Dataset
import os, json, subprocess, yaml

with open("configs/kaggle_datasets.yaml") as f:
    CFG = yaml.safe_load(f)

CKPT_CFG  = CFG["datasets"]["checkpoints"]
USERNAME  = os.environ.get("KAGGLE_USERNAME") or json.load(
    open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"]
DATASET_ID = f"{USERNAME}/{CKPT_CFG['slug']}"
OUTPUT_DIR = "/kaggle/working/output"

# Tạo dataset-metadata.json (Kaggle CLI yêu cầu)
metadata = {
    "id": DATASET_ID,
    "title": CKPT_CFG["title"],
    "licenses": [{"name": CKPT_CFG["license"]}],
}
with open(f"{OUTPUT_DIR}/dataset-metadata.json", "w") as f:
    json.dump(metadata, f)

# Liệt kê checkpoints sẽ push
ckpts = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pth"))
print(f"Pushing {len(ckpts)} checkpoint(s): {ckpts}")

# Thử version (dataset đã tồn tại); fallback: create lần đầu
result = subprocess.run(
    ["kaggle", "datasets", "version", "-p", OUTPUT_DIR,
     "-m", f"checkpoints from session", "--dir-mode", "zip"],
    capture_output=True, text=True
)
if result.returncode != 0 and "not found" in (result.stderr + result.stdout).lower():
    print("Dataset chưa tồn tại — tạo mới…")
    result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", OUTPUT_DIR, "--dir-mode", "zip"],
        capture_output=True, text=True
    )

print(result.stdout)
print(result.stderr)
print(f"\n✅ Dataset: https://www.kaggle.com/datasets/{DATASET_ID}")
```

**Lưu ý setup một lần:**
- Kaggle API token: đã có sẵn trong kernel của user (`/kaggle/working/.kaggle/kaggle.json` hoặc env vars `KAGGLE_USERNAME`/`KAGGLE_KEY`)
- `kaggle` CLI: pre-installed trong Kaggle base image
- Dataset slug `meta-cxr-checkpoints` sẽ auto-create lần đầu (private mặc định)

### Phần 3: Resume workflow trong session 2

**Thêm cell trước training cell** trong `META_CXR_kaggle.ipynb`:

```python
# Cell: Resume từ checkpoint dataset (chỉ chạy ở session 2+)
import os, glob, yaml

with open("configs/kaggle_datasets.yaml") as f:
    CFG = yaml.safe_load(f)

CKPT_SLUG = CFG["datasets"]["checkpoints"]["slug"]

def find_mount(slug):  # đã có ở Cell 4 — re-use
    for root in CFG["mount_search_roots"]:
        candidate = os.path.join(root, slug)
        if os.path.isdir(candidate):
            return candidate
    matches = glob.glob(f"/kaggle/input/**/{slug}", recursive=True)
    return matches[0] if matches else None

ckpt_root = find_mount(CKPT_SLUG)
if ckpt_root:
    # Pick checkpoint số epoch lớn nhất (mới nhất)
    ckpts = sorted(
        glob.glob(f"{ckpt_root}/**/checkpoint_[0-9]*.pth", recursive=True),
        key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]),
    )
    if ckpts:
        os.environ["RESUME_CKPT_PATH"] = ckpts[-1]
        print(f"Resume from: {ckpts[-1]}")
    else:
        print("Checkpoint dataset attached nhưng không tìm thấy file — train từ đầu.")
else:
    print("Không có checkpoint dataset — train từ đầu (session 1).")
```

**Sửa cell training để truyền `resume_ckpt_path` vào CLI:**

Thêm logic ở cell launch training (`pretraining.train`):
```python
extra_args = []
if os.environ.get("RESUME_CKPT_PATH"):
    extra_args = ["--options", f"run.resume_ckpt_path={os.environ['RESUME_CKPT_PATH']}"]
# ... pass extra_args vào torch.distributed.run command
```

(Nếu CLI chưa support `--options` override, fallback: ghi tạm vào YAML trước khi launch.)

## Files to modify

| File | Thay đổi |
|---|---|
| `META-CXR/model/lavis/runners/runner_base.py` | Thêm `save_freq` property + 2 chỗ check `% save_freq` |
| `META-CXR/pretraining/configs/mimic_cxr_2gpu.yaml` | Thêm `save_freq: 3` |
| `META-CXR/configs/kaggle_datasets.yaml` | Thêm `datasets.checkpoints` block |
| `META-CXR/META_CXR_kaggle.ipynb` | Thêm 2 cell: resume-detect (trước training), push-to-dataset (sau training) |

## Workflow user-facing

**Session 1 (epoch 0–4):**
1. Mở notebook → chạy Cell 1–6 như cũ
2. Training save checkpoint tại epoch 2 (`checkpoint_2.pth`) + best
3. Cell push tự động chạy → tạo dataset `username/meta-cxr-checkpoints` chứa checkpoints
4. (Optional) Local: `kaggle datasets download username/meta-cxr-checkpoints` để có bản local

**Session 2 (epoch 5–9):**
1. Notebook → Add Data → tìm `meta-cxr-checkpoints` → Add
2. Chạy lại Cell 1–6
3. Cell resume-detect tìm `checkpoint_4.pth` (mới nhất) → set env var
4. Cell training resume từ epoch 5
5. Save tại epoch 5, 8 → push lên dataset (version 2)

## Không thay đổi

- `_save_checkpoint` body — đã đúng, chỉ thay đổi tần suất gọi
- `_load_checkpoint` — resume mechanism existing đã đủ
- Cell 4 (CSV loader) và `01_generate_mimic_cxr_cleaned_csv.ipynb` — không liên quan
- `is_best=True` path — best checkpoint vẫn lưu mỗi khi cải thiện (không bị skip bởi `save_freq`)

## Verification

1. **Smoke test save_freq:** chạy 4 epoch local với `save_freq: 3` → `output/` có `checkpoint_2.pth` + `checkpoint_best.pth` (KHÔNG có `_0`, `_1`, `_3`)
2. **Disk check:** `du -sh /kaggle/working/output/` ≤ 4GB sau 9 epoch (3 numbered + 1 best ≤ ~3.7GB)
3. **Push test:** chạy push cell → mở `kaggle.com/datasets/<username>/meta-cxr-checkpoints` thấy file
4. **Resume test:** session 2 attach dataset → log in `Resume from: .../checkpoint_2.pth` và `start_epoch: 3`
5. **Best preserved:** epoch nào val cải thiện → `checkpoint_best.pth` vẫn cập nhật bất kể `save_freq`
6. **No regression:** training tiếp tục đến `max_epoch` không crash disk
