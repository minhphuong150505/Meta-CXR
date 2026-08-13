> Source: `training/stage1/lavis_loader.py` (112 dòng)
> Status: 🟡 CONDITIONAL — ★ CỬA DUY NHẤT
> Last verified against source: 2026-08-12

# `training/stage1/lavis_loader.py`

## Purpose

**Cửa duy nhất** giữa Stage 2 và Stage 1. Mọi import LAVIS/mhcac/vision_encoders
trong `training/` phải qua đây.

## Why it exists

Xem [`stage1/_index.md`](_index.md). Tóm tắt: nếu `medgemma_direct` kéo theo LAVIS
thì (a) nó không chạy được ở môi trường Stage 2, và (b) so sánh giữa direct và
Q-Former không còn sạch.

`tests/test_native_independence.py` fail với thông điệp:
*"Move the import into training/stage1/lavis_loader.py and call it from inside the
branch that has already decided it needs Stage 1."*

## Status

```text
🟡 CONDITIONAL — chỉ chạy với --pipeline-mode meta_cxr_qformer*
```

## Main functions

| Hàm | Dòng | Vai trò |
|---|---|---|
| `default_stage1_config_path(run_name)` | 49 | ⚠ resolve vào `encoder_comparison/` |
| `build_cfg(context)` | 53 | YAML → LAVIS `Config` |
| `filter_state_dict_for_model(model, state_dict)` | 58 | Lọc key không khớp |
| `load_state_dict_materializing_meta(model, state_dict)` | 72 | ★ Vật chất hóa tham số meta-device |
| `build_stage1_model(context, checkpoint_root, device)` | 79 | ★ Dựng + nạp checkpoint |
| `make_stage1_loader(cfg, split, sample_limit, num_workers)` | 96 | `MIMIC_CXR_Dataset` → `DataLoader` |

## ⚠ Hai đường resolve config khác nhau

```python
PROJECT_DIR / "pretraining/configs/encoder_comparison" / f"{run_name}.yaml"   # :50
```

`--stage1-run 07_all_three` → tìm trong `encoder_comparison/`.
`--stage1-config` → trỏ thẳng vào file, mặc định `pretraining/configs/mimic_cxr_full.yaml`.

Hai đường này **không** trỏ cùng chỗ. Dễ nhầm.

## `load_state_dict_materializing_meta` — vì sao cần

Model dựng trên `meta` device (không cấp bộ nhớ thật) để tiết kiệm RAM lúc khởi
tạo. Khi nạp state dict, tham số phải được **vật chất hóa** sang device thật. Load
thẳng sẽ lỗi hoặc để lại tham số rỗng.

## Calls / Called by

Gọi: `model.lavis.*` (toàn bộ stack), `model.lavis.data.ReportDataset.MIMIC_CXR_Dataset` (`:31`),
`training.torch_io.load_torch_checkpoint` (`:32`), `torch.utils.data.DataLoader`.
Được gọi: `fig9:316` (**import lazy trong hàm**) → `build_stage1_records`;
`run_medgemma_qlora.build_stage1_records`.

## Side effects

Cấp phát model Stage 1 trên GPU · Đọc checkpoint từ đĩa · Đọc split CSV + ảnh

## Error / edge cases

Checkpoint không tồn tại → raise · State dict không khớp → `filter_state_dict_for_model`
bỏ key thừa ⚠ **lặng lẽ** — kiểm log nếu model có vẻ chưa được nạp

## Related tests

`tests/test_native_independence.py:63,214` — canh chính file này là ngoại lệ duy nhất

## Developer notes

1. ⚠ **Đừng bao giờ import file này ở module scope** từ nơi khác trong `training/`.
   Chỉ gọi từ trong nhánh đã kiểm `requires_stage1`.
2. `filter_state_dict_for_model` và `load_state_dict_materializing_meta`
   **không** được re-export từ tên cũ trong `fig9` (`docs/migration_guide.md:16-17`).
3. Cần thứ gì từ Stage 1 ở chỗ khác? Thêm hàm **vào đây**, đừng import ở đó.

← [`stage1/`](_index.md) · [HOME](../../../HOME.md)
