> Source: `model/lavis/` (34 file Python, 11.092 LOC)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-13

# `model/lavis/`

## Purpose

Fork **đã sửa** của Salesforce LAVIS. Đây là nơi model Stage 1 thực sự sống:
BLIP-2, Q-Former, dataset MIMIC-CXR, train loop, task hook.

> **Đây KHÔNG phải thư viện LAVIS cài qua pip.** Nó là bản vendor có sửa đổi
> đáng kể của Meta-CXR. Đừng thay bằng `pip install salesforce-lavis`.

## Role in project

```text
pretraining/train.py  ──► model/lavis/  ──► toàn bộ Stage 1
inference.py          ──► model/lavis/  ──► đường Vicuna (P9)
training/stage1/lavis_loader.py ──► model/lavis/  ──► Stage 2 mode Q-Former
```

## Parent

[`model/`](../_index.md)

## Children — 6 file cốt lõi (document sâu)

Quyết định phạm vi: [D-007](../../_meta/DECISIONS.md#d-007--độ-sâu-documentation-cho-fork-lavis)

| File | LOC | Doc | Vai trò |
|---|---|---|---|
| `models/blip2_models/blip2_qformer.py` | 1632 | [📄](models/blip2_models/blip2_qformer.py.doc.md) | ★ **Trung tâm Stage 1** — encoder, fusion, MHCAC, Q-Former, 12 loss, inference ablation |
| `models/blip2_models/Qformer.py` | 1221 | [📄](models/blip2_models/Qformer.py.doc.md) | Q-Former (BERT + cross-attention) |
| `runners/runner_base.py` | 1229 | [📄](runners/runner_base.py.doc.md) | ★ Train loop, epoch hook, checkpoint selection, early stopping, resume-in-place, freeze |
| `data/ReportDataset.py` | 1130 | [📄](data/ReportDataset.py.doc.md) | ★ `MIMIC_CXR_Dataset`, study sampling, mask, collate |
| `models/blip2_models/modeling_llama_imgemb.py` | 975 | [📄](models/blip2_models/modeling_llama_imgemb.py.doc.md) | Llama có inject image embedding (đường Vicuna) |
| `tasks/image_text_pretrain.py` | 320 | [📄](tasks/image_text_pretrain.py.doc.md) | Task hook, gọi evaluator Stage 1 |

## Children — 18 file còn lại (mô tả một dòng)

| File | LOC | Vai trò | Sửa đổi? |
|---|---|---|---|
| `common/config.py` | 468 | `Config` — merge **chỉ** `run`/`model`/`datasets` | ⚠ có |
| `common/registry.py` | 329 | Registry model/task/runner/builder | upstream |
| `common/utils.py` | 424 | `now()`, path helper, tải file | upstream |
| `common/optims.py` | 159 | `LinearWarmupCosineLRScheduler` — ⚠ giữ `lr_scale` từng group | ⚠ có |
| `common/dist_utils.py` | 138 | `init_distributed_mode`, `get_rank`, `is_main_process` | upstream |
| `common/logger.py` | 195 | `setup_logger`, `MetricLogger` | upstream |
| `common/gradcam.py` | — | GradCAM (không dùng trong đường chính) | upstream |
| `models/base_model.py` | 264 | `BaseModel`, `concat_all_gather` | upstream |
| `models/blip2_models/blip2.py` | 324 | `Blip2Base` — `init_Qformer`, `init_vision_encoder`, `disabled_train` | ⚠ có (BioViL-T) |
| `models/blip_models/blip_outputs.py` | 146 | [📄](models/blip_models/blip_outputs.py.doc.md) — `BlipOutput` có `loss_explanation` conditional | ⚠ có |
| `models/__init__.py` | 201 | Đăng ký model qua registry | ⚠ có |
| `data/mimic_cxr_utils.py` | 80 | `view_id()`, `build_study_index()` | ⚠ mới |
| `datasets/builders/base_dataset_builder.py` | 234 | Dựng dataset từ config | upstream |
| `datasets/datasets/base_dataset.py` | — | `BaseDataset`, `collater` mặc định | upstream |
| `datasets/datasets/caption_datasets.py` | 84 | Dataset caption | upstream |
| `datasets/datasets/dataloader_utils.py` | 162 | Prefetch loader | upstream |
| `datasets/data_utils.py` | 253 | Helper dữ liệu | upstream |
| `runners/runner_iter.py` | 344 | Runner lặp theo iteration — ⚠ **không được dùng** | upstream |
| `tasks/base_task.py` | 413 | `BaseTask`, `_train_inner_loop` — nơi accumulation sống | ⚠ có |
| `processors/base_processor.py` | — | Processor cơ sở | upstream |

Cộng: `configs/*.yaml` (default + BLIP-2 model config), `data/templates/vicuna.json`,
`defaults_report.yaml`, `LICENSE.txt` (BSD 3-Clause).

## Main responsibilities

1. **Registry** — model/task/runner tự đăng ký; `train.py` kích hoạt bằng star import.
2. **Config** — merge YAML thành `cfg.model_cfg` / `cfg.run_cfg` / `cfg.datasets_cfg`.
3. **Dataset** — `MIMIC_CXR_Dataset`, study sampling, mask, collate ragged aux view.
4. **Model** — `Blip2Qformer` là toàn bộ Stage 1 trong một class.
5. **Train loop** — `RunnerBase` + `BaseTask._train_inner_loop`.
6. **Eval hook** — `ImageTextPretrainTask.evaluation` gọi sang `training/evaluation/`.

## Entry points

Không có. Được `pretraining/train.py` và `inference.py` dùng.

## Dependencies

`torch`, `transformers`, `omegaconf`, `timm`, `torchvision`, `webdataset` ⚠ (cần
runtime verification cho danh sách chính xác — xem `requirements-stage1.txt`).

Ngoài repo: `biovil_t/`, `vision_encoders/`, `mhcac/`, và **`training/evaluation/`**
(import trễ, bên trong hàm, tại `tasks/image_text_pretrain.py:223,259`).

⚠ Import trễ đó tạo phụ thuộc **ngược chiều** `model/lavis/` → `training/`. Nó
được đặt trong hàm chứ không ở module scope, có chủ đích.

## Used by

`pretraining/train.py` (star import) · `inference.py` · `training/stage1/lavis_loader.py`
· `pretraining/precompute_features.py` · `tests/test_stage1_eval_hook.py`,
`tests/test_native_independence.py`, `tests/test_blip2_negative_sampling.py`

## Execution flow

Xem [CALL_GRAPH.md §1–2](../../_meta/CALL_GRAPH.md#1-stage-1--top-down).

## Important configurations

Toàn bộ `pretraining/configs/*.yaml`. ⚠ `Config` merge **chỉ** ba block gốc —
`data:` phải nằm trong `model:`.

## Status

```text
✅ ACTIVE — fork đã sửa, là lõi Stage 1
```

## Notes

- ⚠ **Bị loại khỏi ruff** (`pyproject.toml`) **có chủ đích.** Reformat sẽ làm mọi
  diff với upstream LAVIS sau này không đọc được. Đừng "dọn dẹp style" ở đây.

- ⚠ **`model/lavis/data/` nằm trong `.gitignore`**, nhưng `ReportDataset.py` đã
  được track từ trước. `git add .` **sẽ không** bắt thay đổi ở đó — dùng
  `git add -f model/lavis/data/ReportDataset.py`.

- ⚠ **`.pyc` bị git track** trong nhiều `__pycache__/` ở đây (13 file, Py3.7).
  Ghi nhận [I1](../../_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa).

- **`runner_iter.py` không được dùng** — `train.py` dựng `RunnerBase` trực tiếp và
  bỏ qua `get_runner_class`. Xem [trang đó](../../pretraining/train.py.methods/get_runner_class.md).

- **`ReportDataset.py` chứa ba dataset**: `MIMIC_CXR_Dataset` (✅ dùng),
  `CheXpertDataset` và `IU_Xray_Dataset` (không được `train.py` dựng).
  ⚠ `ReportDataset.py:897` trong `CheXpertDataset` có pattern
  `df[col].fillna(x, inplace=True)` — với pandas 3.0 Copy-on-Write thì **âm thầm
  không làm gì**. Không nằm trên đường MIMIC-CXR nên không ảnh hưởng training hôm nay.

- **`LICENSE.txt` là BSD 3-Clause** của Salesforce. Nó áp cho thư mục này, không
  tự động áp cho phần còn lại của repo.

## Related documentation

[ARCHITECTURE.md](../../_meta/ARCHITECTURE.md) · [DATA_FLOW.md](../../_meta/DATA_FLOW.md) ·
[CALL_GRAPH.md](../../_meta/CALL_GRAPH.md) · [`mhcac/_index.md`](../../mhcac/_index.md)

← [`model/`](../_index.md) · [HOME](../../../HOME.md)
