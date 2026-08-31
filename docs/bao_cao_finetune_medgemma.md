# Báo cáo: fine-tune MedGemma cho sinh phần FINDINGS (Stage 2)

**Trạng thái (cập nhật 2026-08-31 15:25):** attempt đầu của arm A chạy từ 08:46
đến 09:49 rồi dừng ở batch 1.323/88.156 với `torch.AcceleratorError: CUDA error:
the launch timed out and was terminated`. Kernel ghi `NVRM Xid 8` đúng thời điểm
đó; đây không phải OOM (run dùng 12.530 MiB VRAM) và không phải quá nhiệt. Chưa có
adapter cứu hộ. Raw log được giữ tại
`~/ft_only_full.failed-xid8-20260831-094940.log` trên host.

Arm A đã được chạy lại từ 15:21:12; arm B đã móc chuỗi tự khởi động sau nó. Mọi
con số dưới đây là **đo được**, kèm ngày đo và điều kiện; chỗ nào chưa chạy thì
nói rõ là chưa chạy. Checkpoint cứu hộ đầu tiên của attempt mới được xác minh lúc
15:33:41: batch 256, update 32, `status=in_progress`, adapter 119,3 MB và trainer
state 238,8 MB tại `~/ft_only_full/.../checkpoints/last/` trên host.

---

## 1. Mô hình

| | |
|---|---|
| model | `google/medgemma-1.5-4b-it` |
| commit sha | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` |
| kích thước | 2.520.025.456 tham số |
| tải về | 8,1 GB, 2026-08-30, 11m38s |

**Đây là bản MedGemma mới nhất hiện có.** Truy vấn Hub 2026-08-31, toàn bộ 5 model
của `google`, sắp theo `lastModified`:

| model | lastModified | downloads |
|---|---|---:|
| **`medgemma-1.5-4b-it`** ← dùng | **2026-04-13** | 248.386 |
| `medgemma-4b-it` | 2025-10-28 | 963.994 |
| `medgemma-4b-pt` | 2025-09-16 | 1.102 |
| `medgemma-27b-text-it` | 2025-09-16 | 24.163 |
| `medgemma-27b-it` | 2025-07-10 | 276.542 |

`medgemma-4b-it` có nhiều lượt tải hơn nhưng là **thế hệ cũ 1.0**, ra trước gần 6
tháng. Snapshot local trùng đúng `sha` của remote HEAD, tức không có bản cập nhật
nào bị bỏ lỡ.

**Bản 27B không dùng được trên phần cứng này**, và cũng không mới hơn: 54,9 GB
trọng số bf16 → khoảng 13,7 GB ở NF4, trên card 15,5 GB, trong khi bản 4B đã
chiếm 12,5 GB lúc huấn luyện. `27b-text-it` còn là text-only, không có tháp ảnh.

---

## 2. Cấu hình huấn luyện

Đọc trực tiếp từ `adapter_config.json` và `manifest.json` của lần smoke, không
phải từ tài liệu.

### 2.1 QLoRA

```
peft_type      LORA            task_type      CAUSAL_LM
r              16              lora_alpha     32
lora_dropout   0.05            bias           none
target_modules 106  (q_proj / k_proj / v_proj của language tower)
base           google/medgemma-1.5-4b-it, nạp 4-bit NF4 (bitsandbytes 0.50.2)
adapter        113,8 MB
```

| | tham số |
|---|---:|
| tổng | 2.520.025.456 |
| **huấn luyện** | **29.802.496 (1,1826%)** |
| tháp ảnh | 214.281.072 |
| **huấn luyện trong tháp ảnh** | **0** |

Base model đóng băng và lượng tử hoá 4-bit; chỉ 29,8M trọng số LoRA cắm vào
**language tower** là học. Tháp ảnh MedGemma **đóng băng hoàn toàn** —
`assert_vision_tower_frozen()` chặn chạy nếu con số đó khác 0, vì
`target_modules="all-linear"` sẽ vô tình adapt cả tháp ảnh và làm hỏng ablation.

### 2.2 Siêu tham số

```
batch_size 2      grad_accum 8      effective_batch_size 16
lora_lr 1e-4      max_length 768    epochs 1
section_mode findings_only          gradient_checkpointing bật
```

### 2.3 Dữ liệu

| split | bản ghi dùng được | bị loại (target không hợp lệ) |
|---|---:|---:|
| train | **176.312** | 46.446 |
| val | 1.415 | 393 |
| test | 2.984 | 285 |

`skipped_missing_image = 0` trên cả ba split. Manifest tự kiểm: *"no
subject/study/image overlap across splits"*.

### 2.4 Cái này KHÔNG phải cái gì

- **Không phải fine-tune Stage 1.** Log xác nhận: *"medgemma_direct: no Stage-1
  checkpoint, config, Q-Former or MHCAC is loaded; the image is the only clinical
  input"*.
- **Không phải huấn luyện Q-Former.** Đường đó đã đóng — xem
  `docs/negative_result_itc_qformer.md`.
- **Không phải huấn luyện encoder thị giác.** 0 tham số thị giác học.

⚠ `meta.json` có trường `img_token: <qformer_soft_token>`. Đó chỉ là hằng số mặc
định của `VariantLLM`, **không được dùng** ở chế độ native. Trường quyết định là
`image_mode: native`. Ghi lại vì nó trông y hệt bằng chứng rằng Q-Former đang
tham gia.

---

## 3. Thí nghiệm: hai arm khác đúng một biến

| | arm A | arm B |
|---|---|---|
| config | `experiment_native_anchor_only.yaml` | `experiment_native_anchor_guided.yaml` |
| `visual_mode` | `native_anchor_only` | `native_anchor_guided` |
| ảnh | tháp ảnh MedGemma | tháp ảnh MedGemma |
| cues P/N/U từ MHCAC | **không** | **có**, dạng bằng chứng phụ trợ có thể sai |

Hai file được sinh từ cùng một payload và **khác đúng một dòng không-comment**:

```
7c7
<   visual_mode: native_anchor_only
---
>   visual_mode: native_anchor_guided
```

Được chốt bằng `tests/test_stage2_prompts.py::test_the_two_experiment_arms_differ_in_one_line`.
Guard này là trọng tâm: nếu một setting thứ hai trôi giữa hai file thì so sánh
không còn cô lập một biến và **kết quả mất ý nghĩa một cách âm thầm**, vì cả hai
run vẫn chạy xong và vẫn ra metric. `prompt.config_hash` khác nhau giữa hai arm
và được ghi vào adapter, nên artifact không thể lẫn.

Hai guard phụ: không arm nào được dùng `visual_mode` thuộc họ qformer, và cả hai
phải nạp qua đúng `VisualMode` loader và phân giải về `image_mode = "native"`.

**Vì sao đây là ablation sạch hơn đường Q-Former:** cues đến từ đầu phân loại
MHCAC, không phụ thuộc ITC. Kết quả âm tính của ITC do đó không làm mất thí
nghiệm này.

---

## 4. Đo đạc trước khi cam kết GPU

Ba phép đo, làm trước khi khởi động run 70 giờ.

**Batch 4 → OOM.** Batch 2 là trần trên card 16 GB. Không có đòn bẩy tốc độ ở đây.

**Tiền xử lý ảnh chỉ chiếm 5,4% một step**, nên đổi sang processor nhanh không
đáng sửa code:

| | ms / batch-of-2 |
|---|---:|
| giải nén JPEG + RGB | 55,7 |
| `Gemma3ImageProcessor` (chậm) | 99,3 |
| `Gemma3ImageProcessorFast` | 45,0 |
| **một step GPU** | **2.850** |

Chuyển sang bản nhanh tiết kiệm ~54 ms = **1,9%**. Step bị chi phối bởi
forward/backward, không phải I/O.

**Tốc độ và chi phí:**

```
2,85–2,91 s/it       88.156 step/epoch       ETA của tqdm: 69h51m–70h52m
12.530 MiB / 16.311 MiB VRAM
```

| train | epochs | mỗi arm | cả hai arm |
|---:|---:|---:|---:|
| 10.000 | 2 | 7,9 h | 15,8 h |
| 20.000 | 2 | 15,8 h | 31,7 h |
| **176.312 (toàn bộ)** | **1** | **69,8 h** | **139,6 h** |

Đã chọn **toàn bộ dataset, 1 epoch** — 88.156 step, tức 11.020 bước cập nhật
optimizer ở accum 8, đủ nhiều cho LoRA.

---

## 5. Sàn zero-shot — mốc để so, KHÔNG phải kết quả

Trước khi fine-tune, đo `google/medgemma-1.5-4b-it` **nguyên bản** trên test split
để có mốc so sánh.

```
n = 300 study test        0 lỗi        26,7 phút        peak 10,65 GiB
độ dài sinh ra: median 66 từ   |   tham chiếu: median 57 từ
```

| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,3066 | 0,1752 | 0,1091 | 0,0704 | 0,2373 | 0,0556 | **0,0556** | **0,8112** |

⚠⚠ **Đây là SÀN của một mô hình chưa fine-tune. Tuyệt đối không trích như kết
quả Stage 2 của dự án.**

**Khoảng cách CIDEr 0,056 vs BERTScore 0,811 là thứ đáng đọc nhất ở đây.** CIDEr
gần như bằng không nghĩa là gần như không dùng lại n-gram của corpus; BERTScore
0,81 nghĩa là ngữ nghĩa vẫn gần. Đó chính là chữ ký của một mô hình sinh ra văn
trôi chảy, hợp lý lâm sàng, nhưng **không theo văn phong và quy ước của tập báo
cáo MIMIC-CXR**. Đó là khoảng trống mà fine-tune tồn tại để lấp.

⚠ Metric lâm sàng (CheXbert, RadGraph, RadCliQ) báo **unavailable**, không phải 0.
Repo cố ý không implement chúng — `training/evaluation/clinical.py` ném lỗi thay
vì trả điểm bịa.

**Smoke fine-tune** (200 mẫu, 1 epoch, để kiểm đường chạy chứ không phải để đọc
kết quả): `train_loss 1,81776 → val_loss 1,58047`, adapter và predictions JSONL
ghi thành công.

---

## 6. Hạ tầng phải xây thêm để chạy được

**`scripts/generate_stage2_reports.py`** — trước đó không có đường nào từ "một mô
hình đã huấn luyện" tới "một JSONL báo cáo có metric". Script dùng lại chính
`VariantLLM` của dự án ở chế độ native, nên prompt, chat template và tham số sinh
đúng bằng thứ huấn luyện dùng. Không có `--adapter` thì nó ghi
`mode = medgemma_direct_zeroshot` cùng một chuỗi cảnh báo tường minh vào summary.

**`--save-every-updates`** — `save_adapter` vốn chỉ chạy **sau khi vòng lặp batch
kết thúc**, nên một epoch trên toàn bộ dataset chạy ~70 giờ **không có một
checkpoint nào**, và một lần treo ở giờ thứ 69 mất sạch. Máy này có tiền sử treo
cứng không báo trước. Nay ghi adapter mỗi **32 bước cập nhật optimizer**. Với
gradient accumulation 8 và 2,85 giây mỗi batch, nhịp đo được là khoảng 12 phút:
`32 × 8 × 2,85 = 730 giây`.

⚠ Lệnh đầu tiên dùng `--save-every-updates 250` và báo cáo từng gọi đó là
"~13 phút". Sai: bộ đếm chỉ tăng sau `optimizer.step()`, nên 250 update thực tế
là khoảng **95 phút**. Xid 8 xảy ra ở phút 63, trước checkpoint đầu tiên. Attempt
relaunch ngắn với 250 đã được dừng chủ động và lưu riêng ngay khi phát hiện; cả
hai arm hiện dùng 32. Batch, accumulation, LR, loss và dữ liệu không đổi.

⚠ Nó là **artifact cứu hộ, không phải điểm resume**: `resume_state` đặt
`start_epoch = epoch + 1` và không có gì bỏ qua các batch đã chạy, nên resume vẫn
chạy lại epoch. Thứ nó giữ là **trọng số adapter**, phần đắt tiền. Đánh dấu
`status="in_progress"` để adapter dở không bị nhầm là run hoàn tất. Mặc định `0`
giữ nguyên hành vi cũ.

**`scripts/train_healthcheck.sh` + cron mỗi 3 giờ** — báo cáo chỉ-đọc, thoát với
mã 0 OK / 2 WARN / 3 ALERT / 4 IDLE, chỉ gọi Claude triage khi WARN/ALERT. Theo
dõi: tiện trình, GPU (util, VRAM, nhiệt, công suất, quạt, cờ throttle của
driver), nhiệt CPU và NVMe, thời điểm **ghi** checkpoint gần nhất, lỗi kernel,
dung lượng đĩa, trạng thái mount.

**`~/chain_arm_b.sh`** — khởi động arm B sau arm A, nhưng **kiểm artifact chứ
không tin exit code**: phải có `adapter_model.safetensors`, `manifest.json` phải
ghi `status == "complete"` (không phải `in_progress`), và log phải có dòng epoch
hoàn tất. Thiếu một điều kiện là không chạy arm B và ghi lý do. Lý do khắt khe:
chạy arm B **chồng lên** arm A, hoặc chạy sau khi arm A sập, đều tốn ~70 giờ.

---

## 7. Điều kiện vận hành đo được

Sau 47 phút ở 100% tải:

| | giá trị | ngưỡng |
|---|---:|---|
| GPU | 72–74 °C | throttle ~83–88 °C |
| công suất GPU | 164 W | giới hạn 180 W |
| quạt | 49% | — |
| CPU package | 55 °C | high 80, crit 100 |
| NVMe | 38 °C | high 89,8, crit 94,8 |
| cờ throttle của driver | `0x0`, cả ba `Not Active` | — |

**Throughput đi nhanh lên chứ không chậm đi: 3,06 s/it lúc nguội → 2,89 s/it sau
47 phút.** Card bị giới hạn bởi **công suất**, không phải nhiệt — clock ở 2737 của
3090 MHz (88%) trong khi công suất ở 91% giới hạn. Không có lỗi kernel nào từ đầu
boot. Ổ dữ liệu mount `ntfs3 ro` (đã remount từ `ntfs-3g`, thứ từng làm một run
Stage-1 tụt xuống 1,0–1,4 s/it kèm stall 10–14 phút).

---

## 8. Hạn chế

1. **Chưa có kết quả fine-tune.** Arm A đang chạy; báo cáo này ghi thiết kế, chi
   phí và sàn so sánh, không ghi kết quả.
2. **Một epoch duy nhất, không có điểm resume thật.** Treo ở giờ thứ 60 thì mất
   60 giờ tính toán, dù adapter cứu hộ vẫn dùng được.
3. **Không dùng Q-Former.** `lambda_itc/itm/lm = 0.0`, khớp bản gốc. Đường
   soft-token của Stage 2 đóng — xem `docs/negative_result_itc_qformer.md`.
4. **Metric lâm sàng không có.** Đánh giá chỉ dựa trên metric từ vựng và
   BERTScore, vốn **không phải** thước đo độ chính xác lâm sàng.
5. **~6% mục tiêu sinh là báo cáo một dòng.** `target_valid` cho qua findings từ
   5 token; trên val, 64 study (4,2%) có ≤10 token và 91 (6,0%) có ≤15. Điều này
   làm loãng cả loss lẫn mọi metric NLG. Ngưỡng nằm trong
   `preporcessing/preprocess_mimic_cxr.py`, sửa được nhưng phải dựng lại manifest.
6. **Sàn zero-shot đo trên n=300**, không phải toàn bộ 2.984 study test.

---

## 9. Truy vết

| | |
|---|---|
| nhánh | `feat/stage2-explainability` |
| commit liên quan | `db43c44` (bộ sinh), `e3c6e0a` (sửa guard tháp ảnh), `cfef8c6` (checkpoint cứu hộ), `6251958` (hai arm), `5def8ce` (sàn zero-shot), `4b05aea` `988b7dd` `7d269dd` (healthcheck) |
| artifact | `~/ft_only_full` (arm A), `~/ft_guided_full` (arm B), `~/gen_test` + `~/eval_final/stage2_test` (sàn zero-shot) |

⚠ Mọi artifact chứa dữ liệu dẫn xuất từ MIMIC-CXR và **ở nguyên trên máy train**.
