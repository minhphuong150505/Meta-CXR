> Source: `scripts/evaluate_explanation.py`
> Status: ✅ ACTIVE entrypoint — implementation CPU-checked, chưa từng chạy checkpoint/GPU
> Last verified against source: 2026-08-14

# `evaluate_explanation.py`

## Purpose

Nạp Stage-1 checkpoint, chạy một split trong `model.eval()` nhưng **có grad**, tạo
Grad-CAM ở các stream MHCAC và chấm ba metric XAI. Không dựng optimizer, không
gọi backward cập nhật tham số.

Đây không phải extension của `evaluate_stage1.py`: script đó cố ý chỉ đọc `.npz`
và `RunnerBase.eval_epoch` bị `@torch.no_grad()`. CAM cần đạo hàm từ score tới
activation, nên phải có entrypoint model-driven riêng.

## CLI

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_explanation.py \
  --checkpoint <checkpoint_best.pth> \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --split test \
  --mask-cache-dir /mnt/drive1tb/datasets/explanation_masks \
  --ms-cxr-csv /mnt/drive1tb/datasets/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv \
  --output-dir /mnt/drive1tb/private-results/xai \
  --save-cams --export-figures 12 --device cuda
```

Flags bắt buộc: checkpoint, config, mask cache. `split` chỉ nhận val/test;
`--limit` smoke; `--export-figures 0` là mặc định; `--save-cams` opt-in. Nếu bỏ
`--output-dir`, mặc định là `outputs/explanation_evaluation`, đã bị Git ignore.

## Runtime flow

```text
project manifest → MIMIC_CXR_Dataset(split) → anchor image + label + mask
        ↓ model.eval(), grad enabled
_encode_image_streams → mhcac.capture_streams → logits + _last_cam_streams
        ↓ logit_difference_squared → mhcac.explanation.grad_cam
CAM từng stream → bilinear 112² → min-max [0,1]
        ↓ explanation_metrics.summarize (lung và bbox riêng)
metrics.json + tùy chọn cams.npz + PNG overlays
```

Grad-CAM dùng lại `logit_difference_squared` và `grad_cam`; script không cài lại
công thức. Multi-stream graph được giữ đủ lâu để chấm lần lượt, sau đó chỉ giữ
NumPy đã detach.

## Split and bbox invariant

MS-CXR CSV chỉ đọc cột ID, tọa độ và kích thước ảnh; cột `split` **không được
load**. Dataset/manifest project đã quyết định sample thuộc val hay test. Mỗi box
được đưa riêng qua `rasterize_bbox_union(one row)` và
`transform_mask_geometry`, tức tái sử dụng đúng Resize(512) → CenterCrop(448) →
112² đã kiểm chứng. Vì helper mask-builder validate schema có field `split` dù
hình học không đọc field đó, script chèn một marker tổng hợp
`"project_manifest"`; nó không sao chép hay diễn giải giá trị split của MS-CXR.

## Privacy boundary

- PNG và NPZ là dẫn xuất ảnh bệnh nhân, không được commit/push.
- Output bên trong repo bị từ chối nếu `git check-ignore` không xác nhận nó được
  ignore; vùng ignored vẫn phát cảnh báo lớn.
- Tên PNG chỉ là `sample_0001.png`; JSON và stdout không chứa identifier.
- `cams.npz` chỉ được ghi khi có `--save-cams`; chứa CAM, mask source, mask và vị
  trí tuần tự trong split, không chứa ID.

## Output schema

`metrics.json` có `settings`, aggregate counts, `streams.<name>.lung` và
`streams.<name>.bbox`. Không có mixed `overall`. Mỗi metric có `value`,
`available`, `num_samples`; coverage có thêm `num_boxes`. Lung coverage là
`null/unavailable`.

## Main functions

| Function | Doc | Role |
|---|---|---|
| `_assert_private_output_location` | [📄](evaluate_explanation.py.methods/_assert_private_output_location.md) | privacy gate |
| `_build_runtime` | [📄](evaluate_explanation.py.methods/_build_runtime.md) | lazy Stage-1 import/load |
| `_compute_batch_cams` | [📄](evaluate_explanation.py.methods/_compute_batch_cams.md) | pass có grad + Grad-CAM |
| `_export_figure` | [📄](evaluate_explanation.py.methods/_export_figure.md) | overlay identifier-free |
| `_evaluate` | [📄](evaluate_explanation.py.methods/_evaluate.md) | orchestration split/report |

## Validation status

- Đã compile, lint và import trên máy dev; import module không kéo torch/LAVIS.
- Lõi metric được test thuần NumPy.
- **Chưa từng chạy script end-to-end, checkpoint, dữ liệu hay GPU.** Cần smoke
  trên máy train trước khi dùng số/ảnh trong luận văn.

## Tests

`tests/test_explanation_metrics.py` kiểm lõi toán học. Script model-driven không
được chạy trong CPU suite vì máy dev không có torchvision/checkpoint/dataset.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
