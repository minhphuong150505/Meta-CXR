# `scripts/merge_qformer_refit.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-15 · ✅ (chạy trên máy train)
> Last verified against source: 2026-10-15

## Purpose
Ghi checkpoint Stage 1 đã xong (BASE, vd. `run_20261005_paper/phase1c`) với **phía META-Former
thay bằng bản refit của pha `phase1q`**. Bắt buộc: checkpoint của 1q KHÔNG chứa các block encoder
mà 1c đã mở khóa (không pha nào liệt kê chúng trong `trainable`, nên `checkpoint_keep` bỏ chúng);
nạp riêng nó thì encoder âm thầm về bản pretrained và MHCAC đổi theo. Stage 2, probe và chẩn đoán
đều đọc file gộp. Kế hoạch: `docs/handoff/PLAN-2026-10-15-qformer-refit.md`.

```bash
python scripts/merge_qformer_refit.py --base <1c checkpoint_best.pth> \
    --refit <phase1q checkpoint_best.pth> --output <ngoài repo>/merged.pth [--report merge.json]
```

## Quy tắc
- Tiền tố lấy từ refit = `trainable` + `init_from.skip` của pha ghi trong config đã lưu của checkpoint refit.
- **Từ chối** (không ghi gì) khi: một tensor chung ngoài các tiền tố đó khác dù 1 bit (MHCAC,
  adapter, fusion, projector, buffer BatchNorm) — đây là bằng chứng Stage 1 không đổi; refit thiếu
  tensor của BASE dưới các tiền tố; refit có tensor ngoài tiền tố mà BASE không có; output nằm trong
  repo hoặc đã tồn tại.
- In và ghi (`--report`) chỉ số đếm: số tensor thay, thêm (norm), kiểm trùng, giữ từ BASE, epoch của refit.

## Hàm
| Hàm | Vai trò |
|---|---|
| `refit_prefixes(config)` | tiền tố của pha refit từ config đã lưu |
| `merge_states(base, refit, prefixes)` | trả `(merged, summary)`, ném `MergeError` khi vi phạm |

## Calls / Called by
- Gọi: `pretraining.phases.matches_any`, `torch.load/save`.
- Được gọi bởi: người chạy (`~/qrefit_pipeline.sh` trên máy train).

## Tests
`tests/test_merge_qformer_refit.py` (CPU): lấy phía Q-Former từ refit, block encoder và MHCAC từ
BASE; từ chối khi MHCAC hoặc buffer lệch 1e-6, khi thiếu tensor Q-Former, khi có tensor lạ; tiền tố
đọc từ khối pha.
