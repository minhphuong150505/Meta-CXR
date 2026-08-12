> Source: `pretraining/train.py:71-166`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `main()`

## Located in

[`pretraining/train.py`](../train.py.doc.md)

## Purpose

Lắp ráp toàn bộ Stage 1 rồi giao cho `RunnerBase`. Đây là hàm duy nhất trong file
làm việc thật; ba hàm còn lại là helper.

Nó **không** chứa train loop. Việc của nó là đưa bốn thứ — config, task, dataset,
model — về đúng trạng thái rồi gọi `runner.train()`.

## Signature

```python
def main() -> None
```

Không tham số (đọc từ `sys.argv` qua `parse_args()`), không giá trị trả về.

## Parameters

Không có. Đầu vào thực tế đến từ:

| Nguồn | Nội dung |
|---|---|
| `sys.argv` | `--cfg-path`, `--local_rank`, `--options` |
| Biến môi trường | `RANK`, `WORLD_SIZE`, `LOCAL_RANK` (torchrun), `WANDB_API_KEY` |
| `configs/env_config.yaml` | qua `local_config` — đã nạp lúc import module |

## Returns

`None`. Kết quả nằm ở tác dụng phụ: checkpoint và `.npz` trên đĩa.

## Local variables

| Biến | Ý nghĩa |
|---|---|
| `cfg` | `Config` — đã merge `run`/`model`/`datasets`. Bị **mutate** bởi `OmegaConf.update` |
| `job_id` | `now()` — timestamp, đưa vào `RunnerBase` |
| `wandb_run` | Run thật (rank 0) hoặc disabled run. Truyền xuống `runner.train()` |
| `task` | `ImageTextPretrainTask` |
| `datasets` | `{'mimic_cxr': {'train': …, 'val': …, 'test': …}}` — **hardcode một dataset** |
| `truncate_train/val/test` | Giới hạn số mẫu cho smoke test |
| `eval_splits` | Chỉ dùng khi `evaluate=true` |
| `model` | `Blip2Qformer` đã load pretrained weights |
| `runner` | `RunnerBase` |

## Execution flow

```text
Bước 1 — neo path
    registry.mapping['paths']['cache_root'] = '.'      ⚠ global, theo CWD

Bước 2 — config + distributed
    cfg = Config(parse_args())
    init_distributed_mode(cfg)
    OmegaConf.update(cfg.config, "run.gpu", cfg.gpu)             ← bắc cầu
    OmegaConf.update(cfg.config, "run.distributed", cfg.distributed)

Bước 3 — tái lập
    setup_seeds(cfg)        seed = run.seed + get_rank()
    setup_logger()

Bước 4 — logging
    IF is_main_process():  wandb.init(project, entity, name[, id, resume])
                           except UsageError → wandb.init(mode="disabled")
    ELSE:                  wandb.init(mode="disabled")

Bước 5 — task
    task = tasks.setup_task(cfg)

Bước 6 — dataset
                     ┌─ evaluate=false ─ train + val (+ test nếu test_splits)
    cfg.run_cfg ─────┤
                     └─ evaluate=true  ─ test_splits, fallback valid_splits
                                         rỗng cả hai → ValueError

Bước 7 — model
    model = task.build_model(cfg)     → registry → Blip2Qformer.from_config

Bước 8 — giao việc
    RunnerBase(cfg, job_id, task, model, datasets).train(wandb_run)
```

## Detailed logic

**Bước 2 — tại sao cần bắc cầu.** `init_distributed_mode` đặt `cfg.gpu` và
`cfg.distributed` như **thuộc tính Python thường** trên object `Config`. Nhưng
`runner_base` đọc chúng qua `cfg.run_cfg.gpu` — tức là qua OmegaConf. Hai đường
này không tự nối với nhau, nên `main()` phải copy thủ công bằng `OmegaConf.update`.
Bỏ hai dòng này thì runner không biết mình đang chạy trên GPU nào.

**Bước 4 — tại sao rank khác cũng phải `wandb.init`.** Nếu chỉ rank 0 init, một
lời gọi `wandb.log` lạc ở rank khác sẽ raise. Dùng `mode="disabled"` biến mọi
lời gọi thành no-op an toàn.

Việc thiếu API key **không** làm chết run — nó bị bắt và hạ xuống disabled. Đây
là chủ ý: một run dài không nên chết vì logging.

**Bước 6 — hai nhánh.** Nhánh `evaluate=true` cho phép chấm điểm lại một
checkpoint mà không train. Nó chọn split theo thứ tự `test_splits` → `valid_splits`
→ lỗi. `truncate` được tra theo tên split qua một dict.

## Data / Tensor flow

`main()` không đụng tensor. Nó chỉ dựng object. Tensor đầu tiên xuất hiện bên
trong `runner.train()`.

## Side effects

- ⚠ **Mutate global**: `registry.mapping['paths']['cache_root']`
- ⚠ **Mutate `cfg`**: hai lời gọi `OmegaConf.update`
- Seed lại `random`, `numpy`, `torch`; đặt `cudnn.deterministic = True`
- Khởi tạo distributed process group
- Tạo W&B run (mạng)
- Tải BLIP-2 pretrained weights từ URL (mạng, qua `from_config`)
- Cấp phát GPU
- Ghi checkpoint + `.npz` (qua runner)

## Error handling

| Lỗi | Xử lý |
|---|---|
| `wandb.errors.UsageError` | **Bắt** → disabled mode, in cảnh báo, tiếp tục |
| `evaluate=true` + không split | **Raise** `ValueError` |
| Thiếu `env_config.yaml` | Không bắt — đã raise lúc import module |
| Lỗi dataset / model | Không bắt — nổi lên và làm chết process |

## Config dependencies

`run.evaluate` · `run.test_splits` · `run.valid_splits` · `run.truncate_*` ·
`run.seed` · `run.project_name` · `run.wandb_entity` · `run.run_name` ·
`run.wandb_run_id` · `run.wandb_resume`

## Important conditions

```python
if not cfg.run_cfg.evaluate:        # :117  — nhánh TRAIN
if len(cfg.run_cfg.get("test_splits", [])) > 0:   # :133 — test là TÙY CHỌN
if is_main_process():               # :91   — chỉ rank 0 log thật
```

## Related methods

[`parse_args()`](parse_args.md) · [`setup_seeds()`](setup_seeds.md) ·
[`get_runner_class()`](get_runner_class.md) ⚠ không được gọi

## Tests

Không có unit test (cần GPU + dữ liệu). Xem [`tests/_index.md`](../../tests/_index.md).

## Modification risk

| Sửa gì | Ảnh hưởng |
|---|---|
| Xóa star import ở đầu file | `build_model` không tìm thấy `blip2` → **mọi run chết** |
| Bỏ `OmegaConf.update` | Runner không biết gpu/distributed |
| Đổi key `'mimic_cxr'` | Builder/task không khớp dataset |
| Đổi công thức seed | Mất tái lập; DDP các rank có thể trùng seed |
| Dựng runner khác `RunnerBase` | Đổi checkpoint cadence, early stopping, freeze logic |

← [`train.py`](../train.py.doc.md) · [`pretraining/`](../_index.md) · [HOME](../../../HOME.md)
