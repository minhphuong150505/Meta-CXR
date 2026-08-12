> Source: `pretraining/`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `pretraining/`

## Purpose

Chứa **entrypoint và cấu hình của Stage 1** — giai đoạn học biểu diễn thị giác và
huấn luyện bộ phân loại bất thường.

Thư mục này cố ý **mỏng**. Nó không chứa model, không chứa dataset, không chứa
train loop. Tất cả những thứ đó nằm trong `model/lavis/` (fork LAVIS). Vai trò của
`pretraining/` chỉ là: đọc config, dựng dataset, gọi `task.build_model`, giao cho
`RunnerBase`.

## Role in project

Điểm khởi đầu của một trong hai stage chính. Đầu ra của nó (`checkpoint_best.pth`)
là đầu vào tùy chọn của Stage 2 — chỉ khi `--pipeline-mode` là một mode Q-Former.

```text
pretraining/  ──►  checkpoint_best.pth  ──►  training/  (chỉ mode meta_cxr_qformer*)
                                        └──►  scripts/evaluate_stage1.py
```

## Parent

[`struct/project/`](../../HOME.md#source-code-tree) — repository root

## Children

### Files

| File | Doc | Status | Vai trò |
|---|---|---|---|
| `train.py` | [📄](train.py.doc.md) | ✅ ★ | Entrypoint Stage 1 |
| `precompute_features.py` | [📄](precompute_features.py.doc.md) | 🟡 | Tính trước feature encoder đóng băng |
| `__init__.py` | — | ✅ | Rỗng; làm `pretraining` thành package để `python -m` chạy được |

### Subdirectories

| Thư mục | Doc | Nội dung |
|---|---|---|
| `configs/` | [📁](configs/_index.md) | 6 YAML: production, DDP, demo, legacy, ablation |
| `outputs/` | — | **Generated artifact.** Chỉ có `.gitkeep`. `.gitignore` chặn `pretraining/outputs/`. |

## Main responsibilities

1. **Parse config** — `Config(parse_args())` từ `model/lavis/common/config.py`.
2. **Khởi tạo distributed** — `init_distributed_mode(cfg)` đọc `RANK`/`WORLD_SIZE`/
   `LOCAL_RANK` do `torchrun` đặt. Phải chạy qua `torch.distributed.run` kể cả với 1 GPU.
3. **Seed** — `seed = cfg.run_cfg.seed + get_rank()`, `cudnn.deterministic = True`.
4. **W&B** — chỉ rank 0 log thật; rank khác dùng `mode="disabled"` để chặn call lạc.
5. **Dựng dataset** — ba `MIMIC_CXR_Dataset` (train/val/test), có hỗ trợ `truncate`.
6. **Dựng model** — qua `task.build_model(cfg)` → registry → `Blip2Qformer.from_config`.
7. **Giao cho runner** — `RunnerBase(...).train(wandb_run)`.

## Entry points

```bash
# Production, 1 GPU
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --standalone --nproc_per_node=1 \
    -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_full_l4.yaml

# 2-GPU DDP
python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_2x3090.yaml

# Feature cache (tùy chọn)
python -m pretraining.precompute_features --cfg-path <yaml> --options model.encoders.biovil=true
```

Qua launcher: `source cloud/env.local.sh && cloud/run_stage1.sh`

⚠ **Phải dùng `torch.distributed.run`**, không chạy `python pretraining/train.py`
trực tiếp — `init_distributed_mode` cần các biến môi trường mà torchrun đặt.

## Dependencies

| Phụ thuộc | Dùng để làm gì |
|---|---|
| [`model/lavis/`](../model/lavis/_index.md) | Config, registry, dataset, model, runner, task — **gần như toàn bộ logic** |
| [`local_config.py`](../local_config.py.doc.md) | `WANDB_ENTITY`, `WANDB_PROJECT`, `VIS_ROOT` |
| [`mhcac/`](../mhcac/_index.md) | Gián tiếp, qua `blip2_qformer.py` |
| [`vision_encoders/`](../vision_encoders/_index.md), [`biovil_t/`](../biovil_t/_index.md) | Gián tiếp |
| `wandb`, `omegaconf`, `torch` | Trực tiếp |

## Used by

| Ai | Cách |
|---|---|
| `cloud/run_stage1.sh` | Gọi `python -m pretraining.train` với `--options` override output_dir/run_name |
| [`training/stage1/lavis_loader.py`](../training/stage1/_index.md) | **Không** gọi `train.py`, nhưng đọc cùng file config để dựng lại model từ checkpoint |
| `scripts/vm_preflight.py:152` | Kiểm tra sự tồn tại của `mimic_cxr_2x3090.yaml` và `mimic_cxr_full_l4.yaml` |
| `training/run_medgemma_qlora.py:59` | Mặc định `--stage1-config` trỏ vào `mimic_cxr_full_l4.yaml` |

## Execution flow

```text
torchrun
  ↓
train.py::main()
  ↓
Config(args)  →  init_distributed_mode  →  setup_seeds  →  wandb.init
  ↓
tasks.setup_task(cfg)              →  ImageTextPretrainTask
  ↓
MIMIC_CXR_Dataset × {train, val, test}
  ↓
task.build_model(cfg)              →  Blip2Qformer.from_config()
  ↓
RunnerBase(...).train(wandb_run)   →  vòng epoch, validate, checkpoint, early stop
  ↓
runner.evaluate(cur_epoch="best")  →  test chạy MỘT LẦN từ checkpoint_best
```

Chi tiết đầy đủ: [CALL_GRAPH.md §1](../_meta/CALL_GRAPH.md#1-stage-1--top-down)

## Important configurations

| Key | Ở đâu | Ảnh hưởng |
|---|---|---|
| `run.evaluate` | run YAML | `true` → bỏ qua train, chỉ eval trên `test_splits`/`valid_splits` |
| `run.test_splits` | run YAML | Rỗng → **không** dựng test dataset |
| `run.truncate_train/val/test` | run YAML | Cắt dataset cho smoke test |
| `run.seed` | run YAML | Cộng với `get_rank()` cho mỗi process |
| `run.wandb_run_id`, `run.wandb_resume` | run YAML | Nối lại một W&B run cũ khi resume |
| `run.resume_ckpt_path` | run YAML | Resume training |
| `paths.mimic_cxr_jpg_root` | `configs/env_config.yaml` | → `VIS_ROOT`; phải trỏ vào thư mục chứa trực tiếp `files/` |

## Status

```text
✅ ACTIVE
```

`precompute_features.py` là 🟡 CONDITIONAL — chỉ có tác dụng khi
`run.feature_cache_dir` được đặt.

## Notes

- **`train.py` chỉ hỗ trợ một dataset: `mimic_cxr`.** Dict `datasets` được dựng
  cứng với đúng key đó (`train.py:118`). `CheXpertDataset` và `IU_Xray_Dataset`
  tồn tại trong `ReportDataset.py` nhưng entrypoint này không dựng chúng.

- **Comment ở `train.py:39` trỏ vào config legacy** (`mimic_cxr_2gpu.yaml`). Đừng
  copy lệnh trong comment đó — dùng `mimic_cxr_full_l4.yaml`.

- **`registry.mapping['paths']['cache_root'] = '.'`** được đặt ở dòng đầu `main()`
  trước cả khi parse config. Điều này khiến mọi đường dẫn tương đối của LAVIS
  neo vào **thư mục làm việc hiện tại**, không phải repo root. Chạy từ thư mục
  khác sẽ đổi hành vi.

- **`pretraining/outputs/` và `pretraining/embs/`, `pretraining/cls/` đều bị
  `.gitignore` chặn.** Chúng là artifact dẫn xuất từ dữ liệu bệnh nhân.

- ⚠ `utils/split_emb.py` tham chiếu `pretraining/embs/…pkl` — thư mục đó không
  tồn tại trong working tree. Xem [LEGACY_AND_OPTIONAL.md](../_meta/LEGACY_AND_OPTIONAL.md#-potentially_unused--utils).

## Related documentation

- [PIPELINES.md → P1](../_meta/PIPELINES.md#p1--stage-1-pretraining)
- [ARCHITECTURE.md §2](../_meta/ARCHITECTURE.md#2-stage-1--chi-tiết-từng-khối)
- [DATA_FLOW.md §2](../_meta/DATA_FLOW.md#2-stage-1--csv--tensor)
- [`model/lavis/_index.md`](../model/lavis/_index.md)

← [Về HOME](../../HOME.md)
