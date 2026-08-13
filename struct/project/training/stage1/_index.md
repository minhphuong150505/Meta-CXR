> Source: `training/stage1/`
> Status: 🟡 CONDITIONAL — ★ ranh giới kiến trúc
> Last verified against source: 2026-08-12

# `training/stage1/`

## Purpose

**Cửa duy nhất** giữa Stage 2 và Stage 1. Mọi import LAVIS, mhcac,
vision_encoders trong `training/` phải đi qua đây.

## ⚠ Đây là một ràng buộc kiến trúc, không phải quy ước

`tests/test_native_independence.py` quét toàn bộ `training/` và **fail** nếu tìm
thấy import Stage-1 ở bất kỳ file nào khác, kèm thông điệp:

> *"Move the import into training/stage1/lavis_loader.py and call it from inside
> the branch that has already decided it needs Stage 1."*

**Vì sao quan trọng:** nếu `medgemma_direct` kéo theo LAVIS, thì (a) nó không chạy
được trên máy chỉ có môi trường Stage 2, và (b) so sánh giữa `medgemma_direct` và
`meta_cxr_qformer` không còn là so sánh sạch.

## Role in project

```text
run_medgemma_qlora.py :: build_stage1_records()
        │  chỉ gọi khi mode.requires_stage1 == True
        ▼
stage1/lavis_loader.py     ★ CỬA
        ▼
model/lavis/, mhcac/, vision_encoders/, biovil_t/
```

## Parent

[`training/`](../_index.md)

## Children

| File | Doc | Vai trò |
|---|---|---|
| `lavis_loader.py` | [📄](lavis_loader.py.doc.md) | `build_cfg`, `build_stage1_model`, `make_stage1_loader`, `filter_state_dict_for_model`, `load_state_dict_materializing_meta` |
| `__init__.py` | — | Docstring cảnh báo: import `training.stage1.lavis_loader` **tường minh**, đừng dựa vào package import |

## Main responsibilities

1. Dựng `Config` từ YAML Stage 1 (`build_cfg`).
2. Dựng `Blip2Qformer` và nạp `checkpoint_best.pth` (`build_stage1_model`).
3. Lọc state dict cho khớp model hiện tại (`filter_state_dict_for_model`).
4. Vật chất hóa tham số meta-device khi load (`load_state_dict_materializing_meta`).
5. Dựng DataLoader từ `MIMIC_CXR_Dataset` (`make_stage1_loader`).

## Entry points

Không có. Thư viện, gọi từ trong nhánh đã quyết định cần Stage 1.

## Dependencies

`model/lavis/*` (toàn bộ stack), `training/torch_io.load_torch_checkpoint` (`:32`),
`model/lavis/data/ReportDataset.MIMIC_CXR_Dataset` (`:31`).

## Used by

| Ai | Khi nào |
|---|---|
| `training/run_medgemma_qlora.py` | Bên trong `build_stage1_records()`, chỉ khi mode cần Stage 1 |
| `training/train_eval_figure9_llm_variants_200.py:316` | `from training.stage1 import lavis_loader` — **import trễ, trong hàm** |

## Execution flow

```text
build_cfg(run_name | config_path)
   │  ⚠ resolve: PROJECT_DIR / "pretraining/configs/encoder_comparison" / f"{run_name}.yaml"  (:50)
   ▼
build_stage1_model(cfg, checkpoint_path)
   ├─ Blip2Qformer.from_config(cfg.model_cfg)
   ├─ load_torch_checkpoint(checkpoint_path)
   ├─ filter_state_dict_for_model(state, model)
   └─ load_state_dict_materializing_meta(model, state)
   ▼
make_stage1_loader(cfg, split)  →  MIMIC_CXR_Dataset  →  DataLoader
```

## Important configurations

| Flag | Ảnh hưởng |
|---|---|
| `--stage1-config` | Mặc định `pretraining/configs/mimic_cxr_full.yaml` |
| `--stage1-run` | Mặc định `mimic_cxr_full_blip2` |
| `--checkpoint-root` | Nơi tìm `<run>/checkpoint_best.pth` |

⚠ **Hai đường resolve khác nhau:** `build_cfg` theo `run_name` tìm trong
`encoder_comparison/`, còn `--stage1-config` trỏ thẳng vào file. Dễ nhầm.

## Status

```text
🟡 CONDITIONAL — chỉ chạy với --pipeline-mode meta_cxr_qformer*
```

## Notes

- **Đừng bao giờ thêm `import model.lavis` vào file khác trong `training/`.**
  Nếu bạn cần thứ gì từ Stage 1 ở nơi khác, thêm hàm vào file này rồi gọi nó từ
  trong nhánh đã kiểm tra `requires_stage1`.

- `training/torch_io.py` tồn tại **riêng** để `load_torch_checkpoint` dùng được ở
  cả hai phía mà không kéo theo LAVIS. Đó là lý do nó không nằm trong thư mục này.

- `filter_state_dict_for_model` và `load_state_dict_materializing_meta`
  **không được re-export** từ tên cũ trong `fig9` (xem `docs/migration_guide.md:16-17`).

## Related documentation

[CALL_GRAPH.md §3](../../_meta/CALL_GRAPH.md#3-stage-2--top-down) ·
[ARCHITECTURE.md §3.3](../../_meta/ARCHITECTURE.md#33-ranh-giới-độc-lập) ·
[`tests/_index.md`](../../tests/_index.md)

← [`training/`](../_index.md) · [HOME](../../../HOME.md)
