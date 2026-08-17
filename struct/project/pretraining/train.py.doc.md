> Source: `pretraining/train.py` (174 dòng)
> Status: ✅ ACTIVE — ★ ENTRYPOINT
> Last verified against source: 2026-08-12

# `pretraining/train.py`

## Purpose

Entrypoint của **Stage 1**. Nó không huấn luyện gì cả — nó *lắp ráp*: đọc config,
dựng dataset, dựng model, rồi giao toàn bộ việc huấn luyện cho `RunnerBase`.

## Why it exists

Fork LAVIS được thiết kế quanh một registry: model, task, runner, dataset builder
đều tự đăng ký qua decorator. Nhưng registry chỉ biết một class **sau khi module
chứa nó đã được import**. File này tồn tại để:

1. Thực hiện các **star import kích hoạt registry** (dòng 31–35).
2. Nối `local_config.py` (đường dẫn của máy) với `Config` của LAVIS (siêu tham số
   của run) — hai nguồn cấu hình phân tầng.
3. Bắc cầu `cfg.gpu` / `cfg.distributed` từ namespace của `init_distributed_mode`
   sang OmegaConf để `runner_base` đọc được qua `cfg.run_cfg.gpu`.

Không có file này, `registry.get_model_class("blip2")` trả về `None`.

## Role in architecture

```text
người dùng (SSH vào phuong@phuong-b760m-pro-rs-d4-wifi)
        ↓
    train.py          ← BẠN ĐANG Ở ĐÂY
        ↓
Config + Dataset + Model
        ↓
    RunnerBase        ← toàn bộ train loop nằm ở đây
```

## Status

```text
✅ ACTIVE — PRIMARY entrypoint
```

## Used in

Training ✅ · Validation ✅ (qua runner) · Test ✅ (một lần, sau train) ·
Inference ❌ (dùng `inference.py` hoặc Stage 2)

## Entry point

Có. Chạy **plain**, không cần `torch.distributed.run` (1 GPU, `run.distributed: false`):

```bash
CUDA_VISIBLE_DEVICES=0 python \
    -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_full.yaml
```

Chạy `python pretraining/train.py` trực tiếp sẽ hỏng: `init_distributed_mode` cần
`RANK`/`WORLD_SIZE`/`LOCAL_RANK` mà chỉ torchrun đặt.

## Inputs

| Input | Nguồn | Bắt buộc |
|---|---|---|
| `--cfg-path` | CLI | ✅ |
| `--local_rank` | CLI (torchrun đặt) | mặc định `0` |
| `--options k=v …` | CLI | ❌ — override config |
| `configs/env_config.yaml` | qua `local_config.py` | ✅ — thiếu thì `FileNotFoundError` |
| Split CSV train/val/test | qua `MIMIC_CXR_Dataset` | ✅ |
| Ảnh MIMIC-CXR-JPG | qua `VIS_ROOT` | ✅ (trừ khi dùng feature cache) |
| `RANK`, `WORLD_SIZE`, `LOCAL_RANK` | môi trường (torchrun) | ✅ |
| `WANDB_API_KEY` | môi trường | ❌ — thiếu thì tự chuyển sang disabled |

## Outputs

| Output | Nơi ghi |
|---|---|
| `checkpoint_best.pth`, `checkpoint_last.pth`, `checkpoint_<epoch>.pth` | `<output_dir>/<run_name>/` |
| `val_predictions_epoch_best.npz`, `test_predictions*.npz` | `<output_dir>/<run_name>/result/` |
| Log W&B | rank 0 |
| Log stdout | qua `setup_logger()` |

## Important imports

```python
import model.lavis.tasks as tasks                      # setup_task
from model.lavis.common.config import Config           # merge run/model/datasets
from model.lavis.common.dist_utils import (            # DDP
    get_rank, is_main_process, init_distributed_mode)
from local_config import WANDB_ENTITY, WANDB_PROJECT, VIS_ROOT
from model.lavis.common.registry import registry

# ★ Star import — CHỈ để kích hoạt registry, không dùng tên nào trực tiếp
from model.lavis.datasets.builders import *
from model.lavis.models import *          # ← đăng ký Blip2Qformer
from model.lavis.processors import *
from model.lavis.runners import *         # ← đưa RunnerBase vào namespace
from model.lavis.tasks import *

from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset
```

⚠ **Đừng "dọn dẹp" các star import.** Chúng trông như import thừa (linter sẽ báo)
nhưng chúng là cơ chế đăng ký duy nhất. Xóa `from model.lavis.models import *` →
`build_model` không tìm thấy `blip2`.

`RunnerBase` được dùng ở dòng 163 mà **không có import tường minh** — nó vào
namespace qua `from model.lavis.runners import *`.

## Important configuration

| Key | Đọc ở | Ảnh hưởng |
|---|---|---|
| `run.evaluate` | `:117`, `:140` | `true` → bỏ qua train hoàn toàn |
| `run.test_splits` | `:133` | Rỗng → không dựng test dataset |
| `run.valid_splits` | `:143` | Fallback khi `evaluate=true` mà `test_splits` rỗng |
| `run.truncate_train/val/test` | `:114-116` | Cắt dataset (smoke test) |
| `run.seed` | `:60` | `+ get_rank()` cho mỗi process |
| `run.project_name`, `run.wandb_entity`, `run.run_name` | `:93-99` | Định danh W&B |
| `run.wandb_run_id`, `run.wandb_resume` | `:94-95` | Nối lại run cũ |
| `run.runner` | `:68` | Chọn runner class; mặc định `runner_base` |

## Main functions

| Hàm | Doc | Vai trò |
|---|---|---|
| `main()` | [📄](train.py.methods/main.md) | Lắp ráp và khởi động |
| `parse_args()` | [📄](train.py.methods/parse_args.md) | CLI |
| `setup_seeds(config)` | [📄](train.py.methods/setup_seeds.md) | Seed theo rank |
| `get_runner_class(cfg)` | [📄](train.py.methods/get_runner_class.md) | Tra runner từ registry |

⚠ `get_runner_class` **được định nghĩa nhưng không được gọi** trong file này —
`main()` dựng `RunnerBase` trực tiếp ở `:163`. Xem trang của nó.

## Execution flow

```text
main()
  │
  ├─ registry.mapping['paths']['cache_root'] = '.'   ← neo path vào CWD
  ├─ Config(parse_args())
  ├─ init_distributed_mode(cfg)
  ├─ OmegaConf.update(cfg.config, "run.gpu", cfg.gpu)          ← bắc cầu
  ├─ OmegaConf.update(cfg.config, "run.distributed", cfg.distributed)
  ├─ setup_seeds(cfg)
  ├─ setup_logger()
  ├─ wandb.init(...)          rank 0 thật / rank khác disabled
  ├─ cfg.pretty_print()
  ├─ task = tasks.setup_task(cfg)
  │
  ├─ IF không evaluate:  dựng dataset train + val (+ test nếu có test_splits)
  │  ELSE:               dựng dataset cho eval_splits
  │
  ├─ model = task.build_model(cfg)
  └─ RunnerBase(cfg, job_id, task, model, datasets).train(wandb_run)
```

## Calls

| Gọi tới | Ở đâu |
|---|---|
| `Config` | `model/lavis/common/config.py` |
| `init_distributed_mode`, `get_rank`, `is_main_process` | `model/lavis/common/dist_utils.py` |
| `setup_logger` | `model/lavis/common/logger.py` |
| `now()` | `model/lavis/common/utils.py` |
| `tasks.setup_task` | `model/lavis/tasks/__init__.py` |
| [`MIMIC_CXR_Dataset`](../model/lavis/data/ReportDataset.py.doc.md) | `model/lavis/data/ReportDataset.py` |
| `task.build_model` → [`Blip2Qformer.from_config`](../model/lavis/models/blip2_models/blip2_qformer.py.doc.md) | |
| [`RunnerBase.train`](../model/lavis/runners/runner_base.py.doc.md) | |
| `wandb.init` | thư viện ngoài |

## Called by

| Ai | Cách |
|---|---|
| Người dùng | `python -m pretraining.train` (plain, qua SSH vào máy train) |

Không module Python nào import file này — nó là lá của cây gọi hàm.

## Data flow

```text
--cfg-path YAML ────► Config ──────────────┐
configs/env_config.yaml ► local_config ────┤
                                            ├──► MIMIC_CXR_Dataset ──► batch
split CSV ──────────────────────────────────┘         ↓
                                              Blip2Qformer.forward
                                                      ↓
                                              BlipOutput(loss, logits)
                                                      ↓
                                              RunnerBase → checkpoint + .npz
```

Chi tiết: [DATA_FLOW.md](../_meta/DATA_FLOW.md)

## Side effects

| Tác dụng phụ | Chi tiết |
|---|---|
| **Global state** | `registry.mapping['paths']['cache_root'] = '.'` — thay đổi toàn cục, ảnh hưởng mọi resolve path của LAVIS |
| **RNG** | `random`, `numpy`, `torch` đều bị seed lại |
| **cuDNN** | `benchmark = False`, `deterministic = True` — chậm hơn, đổi lấy tái lập |
| **Distributed** | Khởi tạo process group |
| **Mạng** | W&B run; tải BLIP-2 pretrained weights từ URL trong config |
| **Đĩa** | Checkpoint, `.npz`, log |
| **GPU** | Cấp phát toàn bộ model |
| **Config mutation** | `OmegaConf.update` sửa `cfg.config` tại chỗ |

## Error / edge cases

| Tình huống | Hành vi |
|---|---|
| Thiếu `configs/env_config.yaml` | `FileNotFoundError` từ `local_config.py:6` — **lúc import**, trước cả `main()` |
| Không có `WANDB_API_KEY` | Bắt `wandb.errors.UsageError` → `wandb.init(mode="disabled")`, in cảnh báo, **tiếp tục chạy** |
| `evaluate=true` mà `test_splits` và `valid_splits` đều rỗng | `ValueError("evaluate=true requires test_splits or valid_splits")` (`:145`) |
| `test_splits` rỗng lúc train | Không dựng test dataset — im lặng, đúng ý đồ |
| Không chạy qua torchrun | `init_distributed_mode` không tìm thấy biến môi trường ⚠ hành vi cụ thể cần runtime verification |

## Related tests

Không có test trực tiếp cho `train.py` (nó cần GPU + dữ liệu). Gián tiếp:

- `tests/test_stage1_objectives.py` — teacher/student separation, shape student
- `tests/test_stage1_eval_hook.py` — eval hook ⚠ cần `model.lavis`, fail trên CPU
- `tests/test_mimic_data_pipeline.py` — study sampling
- `tests/test_view_fusion.py` — identity tại step 0
- `tests/test_multiview_losses.py` — MPC + view consistency

Xem [`tests/_index.md`](../tests/_index.md).

## Related documentation

- [`_index.md` thư mục cha](_index.md)
- [PIPELINES.md → P1](../_meta/PIPELINES.md#p1--stage-1-pretraining)
- [CALL_GRAPH.md §1](../_meta/CALL_GRAPH.md#1-stage-1--top-down)
- [`configs/_index.md`](configs/_index.md)

## Developer notes

1. **Đừng xóa star import.** Linter sẽ bảo chúng thừa. Chúng không thừa.

2. **`cache_root = '.'` neo vào CWD.** Chạy từ thư mục khác sẽ đổi nơi LAVIS
   resolve đường dẫn tương đối. Luôn chạy từ repo root.

3. **Chỉ hỗ trợ dataset `mimic_cxr`.** Dict `datasets` được dựng cứng
   (`:118`). Thêm dataset khác cần sửa file này, không chỉ sửa config.

4. **`get_runner_class` là code chết trong file này** — `main()` không gọi nó.
   Nếu bạn muốn dùng `runner_iter` thay `runner_base`, đặt `run.runner` **không
   đủ**; phải sửa `:163`.

5. **W&B init trước `cfg.pretty_print()`**, nên một config sai vẫn tạo ra một
   W&B run rỗng.

6. **Sửa file này ảnh hưởng tới:** mọi Stage 1 run; gián tiếp cả Stage 2 mode
   Q-Former (vì chúng dùng checkpoint do file này sinh ra) và
   `scripts/evaluate_stage1.py` (đọc `.npz` do file này sinh ra).

## Source relationships

- **Parent:** [`pretraining/_index.md`](_index.md)
- **Methods:** [`train.py.methods/`](train.py.methods/)
- **Related:** [`runner_base.py`](../model/lavis/runners/runner_base.py.doc.md) ·
  [`blip2_qformer.py`](../model/lavis/models/blip2_models/blip2_qformer.py.doc.md) ·
  [`ReportDataset.py`](../model/lavis/data/ReportDataset.py.doc.md) ·
  [`local_config.py`](../local_config.py.doc.md)

← [Về HOME](../../HOME.md)
