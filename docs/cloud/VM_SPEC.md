# VM Spec — Stage-1 + Stage-2 trên GCP L4

Thông số máy ảo dùng để train META-CXR. Số liệu quota/GPU trong file này đã được
verify bằng `gcloud` với tài khoản `phuongnm150505@gmail.com` ngày **2026-07-21**.

> **Định danh không nằm trong file này.** Repo `META-CXR/` là public. Tên project và
> bucket đọc từ `cloud/env.local.sh` (git-ignored) — xem `cloud/env.sh` để biết các
> biến cần export. Các lệnh dưới đây dùng `$GCP_PROJECT`, `$GCS_DATA_BUCKET`,
> `$GCS_BUCKET`; source file local trước khi chạy.

---

## 0. ⛔ BLOCKER — chưa tạo được VM (2026-07-21)

Lệnh tạo VM ở §5 **đã chạy và thất bại**:

```
ERROR: Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.
```

GCP có **hai** quota GPU độc lập. Bản đầu của file này chỉ kiểm cái thứ nhất rồi
kết luận sai là "không cần xin thêm":

| Quota | Phạm vi | Giá trị | |
|---|---|---|---|
| `NVIDIA_L4_GPUS` | per-region | 1 | ✅ đạt |
| `GPUS_ALL_REGIONS` | **global, mọi loại GPU** | **0** | ⛔ chặn |

Trần global = 0 thì không tạo được GPU ở bất kỳ region nào, bất kể quota region.

### Nguyên nhân: tài khoản đang ở GCP free trial

Free trial đặt `GPUS_ALL_REGIONS = 0`, và **yêu cầu nâng quota GPU bị từ chối tự động**
khi còn ở trạng thái trial. Xin quota trước khi upgrade là mất công vô ích.

### Cách gỡ — đúng thứ tự

1. **Upgrade billing account sang paid** — Console → Billing → "Activate full account".
   - Credit còn lại **được giữ nguyên**, không mất khi upgrade.
   - Credit vẫn được tiêu trước, chỉ khi hết mới tính tiền thẻ.
   - Đây chỉ là gỡ trần trial, không phải bắt đầu trả tiền ngay.
2. **Sau khi upgrade**, xin nâng `GPUS_ALL_REGIONS` lên **1**:
   Console → IAM & Admin → Quotas → lọc "GPUs (all regions)" → Edit → 1.
   Thường duyệt trong vài phút tới 2 ngày làm việc.
3. Chạy lại lệnh tạo VM ở §5.

### Ghi chú ảnh boot

Image family `common-cu124-*` **đã bị GCP gỡ**. Hiện chỉ còn CUDA 12.9 / driver 580.
§2 và §5 đã cập nhật sang `common-cu129-ubuntu-2204-nvidia-580`. Torch 2.5.1+cu124
dự kiến vẫn chạy nhờ CUDA minor-version compatibility (driver 580 tương thích ngược
runtime 12.4, và torch mang CUDA runtime riêng) — **nhưng chưa verify**, xem §7.

---

## 1. Bối cảnh đã verify

| Hạng mục | Giá trị | Ghi chú |
|---|---|---|
| Project dùng được | 1 project duy nhất | 5 project tồn tại, **chỉ 1 cái bật billing** |
| Zone | `us-central1-a` | region `us-central1` |
| Quota `NVIDIA_L4_GPUS` (per-region) | **1** | us-central1 / us-east1 / us-west1 / us-west4 |
| Quota `GPUS_ALL_REGIONS` (global) | **0 — CHẶN** | ⛔ xem §0. Đây là quota thứ hai, độc lập; per-region PASS không có nghĩa là tạo được GPU |
| Quota `SSD_TOTAL_GB` | **500 GB / region** | ⚠️ ràng buộc chính, xem §4 |
| VM đang có | **0** | chưa dựng gì |
| Bucket data | có sẵn, ~571 GiB ảnh p10–p19 | UBLA + PAP `enforced` |
| Bucket checkpoint | đã tạo 2026-07-20 | us-central1 regional, UBLA + PAP `enforced` |
| API `compute.googleapis.com` | đã bật | |

---

## 2. Máy ảo

| Thông số | Giá trị | Vì sao |
|---|---|---|
| Machine type | **`g2-standard-8`** | 1× L4 gắn cứng vào họ G2 |
| vCPU | **8** | dataloader `num_workers: 4`; multi-view load tới 2 ảnh/sample nên decode JPEG là CPU-bound |
| RAM | **32 GB** | đủ, nhưng có 1 rủi ro — xem §3.2 |
| GPU | **1× NVIDIA L4, 24 GB GDDR6** | Ada Lovelace, bf16 native, hỗ trợ NF4 qua bitsandbytes |
| Boot disk | **200 GB `pd-balanced`** | OS + conda + weights + code |
| Image | Deep Learning VM, **CUDA 12.9 / driver 580** | cu124 image đã bị gỡ; xem §0 |
| Provisioning | on-demand (KHÔNG spot) | xem §6 |

### Vì sao `g2-standard-8` chứ không phải bản nhỏ/lớn hơn

- `g2-standard-4` (4 vCPU / 16 GB): 16 GB RAM quá sát khi Stage-2 dựng feature cache,
  và 4 vCPU không nuôi nổi 4 dataloader worker + main process.
- `g2-standard-16` (16 vCPU / 64 GB): đắt hơn ~60% nhưng **vẫn 1 GPU**. Nút cổ chai là
  GPU và disk IOPS, không phải vCPU. Chỉ nâng nếu đo được dataloader đói thật.

---

## 3. Ngân sách VRAM (24 GB)

### 3.1 Stage-2 — MedGemma QLoRA — **thoải mái**

| Thành phần | Ước tính |
|---|---|
| MedGemma 1.5 4B, 4-bit NF4 double-quant | ~2.5–3 GB |
| LoRA adapter r=8 | < 100 MB |
| Activations @ `batch_size 2`, `max_length 768` | ~4–6 GB |
| Optimizer state (chỉ LoRA param) | < 1 GB |
| **Tổng** | **~8–10 GB / 24 GB** |

Còn dư nhiều. Nếu smoke run xác nhận ổn, có thể tăng `--batch-size` lên 4 và giảm
`--grad-accum` xuống 4 để giữ nguyên effective batch mà chạy nhanh hơn.

### 3.2 Stage-1 — chưa đo được, đây là rủi ro thật

3 vision encoder (BioViL-T 1408 + PubMedCLIP 768 + SwinV2) **frozen** nhưng vẫn tốn
activation khi forward. Cộng Q-Former + MHCAC + ITC negative queue 1024 sample.
`batch_size_train: 2` với `amp: true` — nhỏ, nên nhiều khả năng vừa. **Nhưng chưa
từng chạy trên GPU nào**, nên đây là ước lượng, không phải kết quả đo.

**Rủi ro RAM (không phải VRAM):** `build_stage1_records` trong Stage-2 ghi một file
`.pt` cache khoảng **10–11 GB** cho 220k row. Nếu nó tích trong RAM trước khi ghi
xuống đĩa, 32 GB sẽ căng. Cần theo dõi `free -g` trong lần chạy đầu.

---

## 4. Disk — chỗ khó nhất

`pd-ssd` và `pd-balanced` **đều tính vào `SSD_TOTAL_GB` = 500**. Đây là ràng buộc
quyết định, không phải chi tiết phụ.

### Kịch bản A — Smoke run (làm trước)

| Disk | Dung lượng | Loại |
|---|---|---|
| Boot | 200 GB | `pd-balanced` |
| Data | **không cần** | |

**Tổng 200 / 500 GB — nằm trong quota, không cần xin gì.**

`--train-limit 500` cắt dataset ngay ở loader, nên chỉ cần ảnh của ~570 study
(vài GB), copy thẳng vào boot disk. Không cần mirror 571 GiB.

### Kịch bản B — Full run

| Disk | Dung lượng | Loại |
|---|---|---|
| Boot | 200 GB | `pd-balanced` |
| Data | 800 GB | `pd-balanced` |

**Tổng 1000 GB > quota 500 GB → SẼ BỊ TỪ CHỐI.** Hai đường ra:

1. **Xin nâng `SSD_TOTAL_GB` lên ≥ 1100** (khuyến nghị). Free tier thường duyệt trong
   vài giờ tới 2 ngày làm việc.
2. **Dùng `pd-standard` cho data disk** — tính vào `DISKS_TOTAL_GB`, quota riêng.
   ⚠️ **Không nên.** `pd-standard` là HDD, ~0.75 read IOPS/GB → 800 GB chỉ được
   ~600 IOPS. Training đọc random hàng trăm nghìn JPEG nhỏ; GPU sẽ ngồi chờ disk và
   epoch có thể chậm gấp nhiều lần. Tiết kiệm ~$70/tháng tiền disk để phí hàng trăm
   đô tiền GPU nhàn rỗi là lỗ.

`pd-balanced` 800 GB cho ~4800 IOPS, đủ nuôi dataloader.

> **Nộp đơn quota ngay hôm nay** nếu định chạy full — nó chạy song song với smoke run,
> không tốn gì thêm, và tránh phải chờ về sau.

---

## 5. Lệnh tạo

```bash
source cloud/env.local.sh   # nạp GCP_PROJECT / GCS_* — file này git-ignored

# --- Kịch bản A: smoke run ---
gcloud compute instances create meta-cxr-l4 \
  --project="$GCP_PROJECT" \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --metadata="install-nvidia-driver=True" \
  --scopes=https://www.googleapis.com/auth/cloud-platform

# --- Kịch bản B: thêm data disk (CHỈ sau khi quota SSD đã lên >= 1100) ---
gcloud compute disks create mimic-data \
  --project="$GCP_PROJECT" --zone=us-central1-a \
  --size=800GB --type=pd-balanced

gcloud compute instances attach-disk meta-cxr-l4 \
  --project="$GCP_PROJECT" --zone=us-central1-a --disk=mimic-data
```

`--maintenance-policy=TERMINATE` là **bắt buộc** với VM có GPU — GCP không live-migrate
được máy gắn GPU.

Format + mount data disk, sync dữ liệu, và viết `configs/env_config.yaml`:
xem `docs/cloud/GCP_L4_TRAINING_SETUP_GUIDE.md` §5–§6.

---

## 6. Chi phí (us-central1, giá on-demand)

| Hạng mục | Đơn giá | Ghi chú |
|---|---|---|
| `g2-standard-8` + 1× L4 | **~$0.85–1.00 / giờ** | chỉ tính khi VM đang RUNNING |
| `pd-balanced` | ~$0.10 / GB-tháng | **tính cả khi VM đã tắt** |
| 200 GB boot | ~$20 / tháng | |
| 800 GB data | ~$80 / tháng | |
| Egress GCS → VM | **$0** | cùng region `us-central1` — đừng đặt khác region |

**Smoke run: ~2–3 giờ ≈ $2–3.**

**Không dùng Spot VM.** Rẻ hơn ~60% nhưng bị preempt bất cứ lúc nào, mà `train_fine`
chỉ lưu adapter **sau epoch cuối** — bị cắt giữa chừng là mất sạch. Chỉ cân nhắc spot
sau khi đã thêm checkpoint giữa epoch.

> `gcloud compute instances stop meta-cxr-l4` ngay khi chạy xong. Disk vẫn tính tiền
> nhưng GPU thì không — đó là phần đắt.

---

## 7. Còn chưa verify

Ghi rõ ra đây để không nhầm là đã kiểm chứng:

1. **`transformers==4.53.2` có load nổi MedGemma 1.5 không** — pin từ giữa 2025, repo
   model cập nhật 2026-04-13. Đây là thứ smoke run cần trả lời **đầu tiên**; nếu hỏng
   thì biết trong ~15 phút và tắt VM ngay.
2. **VRAM thật của Stage-1** — §3.2 là ước lượng. Chưa có gì trong đợt rewrite Stage-1/2
   từng chạy trên GPU.
3. **Throughput thật** — không có số đo nào cho recipe hiện tại. Số cũ (2× T4, p10) đã
   lỗi thời hai lần: khác phần cứng, và `study_sampling` đổi định nghĩa epoch từ ~365k
   image row thành ~220k study.
4. **Đỉnh RAM khi dựng feature cache** — xem §3.2.

Mọi con số thời gian đưa ra trước đây đều là **ước lượng**, chưa đo.
