> Source: `scripts/check_itc_gate.py` (305 dòng)
> Status: 🔬 DIAGNOSTIC — read-only, không train, không ghi checkpoint
> Last verified against source: 2026-08-19

# `scripts/check_itc_gate.py`

## Purpose

Quyết định **có đáng bỏ ~33 giờ GPU để train khối vision-language hay không**,
bằng một phép đo ~7 phút: ITC đã thoát khỏi mức ngẫu nhiên chưa?

## Why it exists

`lambda_itc/itm/lm` đã bị tắt hai lần (2026-08-13 vì chi phí, 2026-08-16 vì
`loss_itc` nằm **đúng** ở `ln(queue)` — tức đầu ra hằng số, một head đã sụp chứ
không phải học chậm). Lần bật lại 2026-08-18 cần một tiêu chí **đăng ký trước**
để không lặp lại vòng đó. Script này là tiêu chí đó.

## Đo cái gì

Một InfoNCE hai chiều all-to-all trên **một subset cố định, không xáo trộn**,
ở `eval()`, không queue, không gradient:

```
delta = ln(N) - L_itc
```

| Trường JSON | Ý nghĩa |
|---|---|
| `delta_nats` | Số nat tách được so với chance. Tiêu chí: **≥ 0.10 VÀ lớn hơn lần đo trước** |
| `mean_rank_of_true_pair_i2t` / `_t2i` | Thứ hạng cặp đúng, chance = `(N-1)/2`. **Không phụ thuộc nhiệt độ** |
| `temperature`, `temp_learnable` | Chế độ nhiệt độ của phép đo |

⚠ **`delta_nats` tỉ lệ với `1/temperature`.** Hai phép đo chỉ so sánh được khi
`temperature` khớp nhau; đó là lý do `temp_learnable` có trong báo cáo. Khi so
sánh chéo hai chế độ, **chỉ đọc trường rank**.

Subset cố định và có thứ tự — so sánh hai lần đo là toàn bộ mục đích, một subset
khác làm hai con số vô nghĩa.

## Cách gọi

```bash
python scripts/check_itc_gate.py --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --split val --pairs 256 --batch-size 8 --device cuda:0 \
    [--checkpoint <ckpt.pth>] [--options KEY=VALUE ...] --output <out.json>
```

Bỏ `--checkpoint` để đo khởi tạo chưa train. `--options` (thêm 2026-08-19) nhận
override cùng cú pháp `pretraining/train.py`, chồng lên ba override một-process
mà script luôn đặt — để đo một biến thể **không cần sửa YAML đang được track**,
ví dụ `model.loss.itc_temp_learnable=False`.

Exit code **1 khi không đạt ngưỡng** — chạy trong script shell thì nhớ điều này.

## Kết quả đã đo (2026-08-19, val, 256 cặp, chance rank 127.5)

| | chưa train | sau ~525 update | ngưỡng |
|---|---|---|---|
| `delta_nats` | −1.3151 | **−24.5235** | ≥ +0.10 |
| rank i2t / t2i | 120.21 / 129.56 | **116.64 / 126.81** | — |
| `temperature` | 0.024888 | **0.00796** | — |

**Trượt.** Sau 525 optimizer update thứ hạng cặp đúng chỉ nhích 3.6 bậc trên 256.
Xem `CLAUDE.md` mục Q-Former để biết diễn giải đầy đủ và bước tiếp theo.

## Callers

Không module nào import. Gọi tay hoặc qua script shell trên máy train.

## Đọc thêm

- [`blip2_qformer.py`](../model/lavis/models/blip2_models/blip2_qformer.py.doc.md) — `itc_temp`, `itc_temp_learnable`, `itc_queue_size`
- `docs/handoff/PLAN-2026-08-19-itc-temp-probe.md` — probe đang chờ chạy
