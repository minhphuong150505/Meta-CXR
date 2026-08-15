> Source: `model/lavis/runners/runner_base.py` (1.229 dòng)
> Status: ✅ ACTIVE — ★
> Last verified against source: 2026-08-15

# `runner_base.py`

## Purpose

Train loop của Stage 1, cộng mọi chính sách quanh nó: chọn checkpoint, early
stopping, cadence lưu, đóng băng tham số, resume, và gọi eval.

## Why it exists

`pretraining/train.py` chỉ lắp ráp. Toàn bộ **chính sách huấn luyện** sống ở đây.
Đây là bản LAVIS đã sửa nhiều: `selection_metric`, `early_stop_patience`,
`eval_start_epoch`, `save_freq`, freeze-list là bổ sung của Meta-CXR.

### ⚠ `eval_start_epoch` — cửa sổ warm-up không chấm điểm

`run.eval_start_epoch` (mặc định `0`) bỏ qua validation cho các epoch có index nhỏ
hơn nó. Đếm theo **index**, khớp log train: `epoch: [5]` là epoch thứ sáu.

Hai hành vi đi kèm, cả hai đều quan trọng:

1. **Không epoch nào chưa chấm có thể được chọn.** `checkpoint_best` chỉ được ghi
   *bên trong* nhánh evaluation, nên bỏ eval cũng đồng thời loại epoch đó khỏi
   diện tuyển. Cùng một knob, cố ý.
2. **Patience chỉ đếm epoch đã chấm.** `best_epoch` khởi tạo bằng `0` trong khi
   epoch chấm đầu tiên là `eval_start_epoch`, nên điều kiện dừng được kẹp thành
   `cur_epoch - max(best_epoch, eval_start_epoch) >= early_stop_patience`. Nếu đo
   từ `best_epoch` trần, toàn bộ ngân sách patience bị tiêu bởi các epoch chưa
   từng eval và run **chết ngay ở epoch chấm đầu tiên** mà chưa lưu best nào —
   log ghi "early stopping", trông y hệt hội tụ. Với `eval_start_epoch: 0` biểu
   thức này bằng đúng hành vi cũ.

⚠ **Số học của config hiện tại:** `eval_start_epoch` 5 + `patience` 5 = 10, mà
index cuối của run 10 epoch là `[9]`. **Early stopping không thể kích hoạt.** Vô
hại (run kết thúc ở [9] dù thế nào) nhưng giá trị đang bất động; muốn nó sống thì
đặt patience ≤ 4. Có test khẳng định điều này để việc đổi `max_epoch` không âm thầm
làm sai.

## Role in architecture

```text
train.py ──► RunnerBase.train() ──► BaseTask.train_epoch ──► Blip2Qformer.forward
                    │
                    ├─► validate()  ──► ImageTextPretrainTask.evaluation
                    └─► _save_checkpoint()
```

## Status

```text
✅ ACTIVE
```

## Used in

Training ✅ · Validation ✅ · Test ✅ (một lần, sau train)

## Entry point

Không.

## Inputs

`cfg` (LAVIS Config), `job_id`, `task`, `model`, `datasets` dict.

## Outputs

`checkpoint_best.pth`, `checkpoint_last.pth`, `checkpoint_<epoch>.pth`,
log W&B, và (qua task) `.npz` prediction.

## Important configuration

| Property | Config key | Mặc định prod |
|---|---|---|
| `max_epoch` | `run.max_epoch` | **10** |
| `eval_start_epoch` | `run.eval_start_epoch` | **5** — epoch index đầu tiên được chấm; [0]–[4] chỉ train |
| `selection_metric` | `run.selection_metric` | **`loss`** (YAML `:366`) — đã đổi từ `macro_auprc` |
| `selection_mode` | `run.selection_mode` | **vắng mặt trong YAML một cách có chủ ý** → suy ra `min` vì tên metric chứa `loss` (`:501`) |
| `early_stop_patience` | `run.early_stop_patience` | 5 — ⚠ **hiện không thể kích hoạt**, xem ghi chú dưới |
| `early_stop_min_delta` | `run.early_stop_min_delta` | 1.0e-4 |
| `save_freq` | `run.save_freq` | 5 (`0` → chỉ best + last) |
| `accum_grad_iters` | `run.accum_grad_iters` | 8 |
| `max_grad_norm` | `run.max_grad_norm` | 1.0 |
| `amp_dtype` | `run.amp_dtype` | `bfloat16` |
| `resume_ckpt_path` | `run.resume_ckpt_path` | — |
| `use_dist_eval_sampler` | `run.use_dist_eval_sampler` | false |

## Main classes

`RunnerBase` (`:89`) — không đăng ký registry ở file này; `train.py` dựng trực tiếp.

## Main methods

| Method | Doc | Vai trò |
|---|---|---|
| `setup_output_dir` (`:499`) | [📄](runner_base.py.methods/setup_output_dir.md) | ★ Resume local vào đúng thư mục checkpoint cũ |
| `train` (`:656`) | [📄](runner_base.py.methods/train.md) | ★ Vòng epoch, early stop, checkpoint |
| `validate` (`:553`) | [📄](runner_base.py.methods/validate.md) | ★ Eval + so sánh best |
| `_save_checkpoint` (`:939`) | [📄](runner_base.py.methods/_save_checkpoint.md) | ★ Ghi checkpoint, lọc param đóng băng |
| `_load_checkpoint` (`:1010`) | [📄](runner_base.py.methods/_load_checkpoint.md) | Resume |
| `eval_epoch` (`:806`) | — | Gọi `task.evaluation` cho một split |
| `evaluate` (`:776`) | — | Reload best rồi eval test |
| `train_epoch` (`:840`) | [📄](runner_base.py.methods/train_epoch.md) | Truyền epoch cho model rồi ủy quyền task |
| `_metric_improved` (`:488`) | — | So sánh có `min_delta`, theo `selection_mode` |
| `_reduce_eval_stats` (`:520`) | — | Gộp stats qua các rank |
| `_reload_best_model` (`:982`) | — | Nạp lại `checkpoint_best` |
| `create_loaders` (`:848`) | — | Dựng DataLoader, sampler DDP |
| `_delete_local_resume_checkpoint_after_load` (`:1129`) | — | Dọn checkpoint resume cục bộ |

### Property (lazy, cached)

`model` (`:151`), `optimizer` (`:174`), `scaler` (`:230`), `amp_dtype` (`:246`),
`lr_scheduler` (`:262`), `dataloaders` (`:294`), `device`, `use_distributed`,
và ~15 property đọc thẳng từ `run_cfg`.

## Execution flow

Xem [CALL_GRAPH.md §1](../../../_meta/CALL_GRAPH.md#vòng-train).

## Calls

`task.train_epoch` → `BaseTask._train_inner_loop` · `task.evaluation` →
`ImageTextPretrainTask.evaluation` · `LinearWarmupCosineLRScheduler`
(`common/optims.py`) · `torch.save` / `torch.load`

## Called by

`pretraining/train.py:163`

## Side effects

Ghi checkpoint và log · W&B · Cấp phát optimizer/scaler · cập nhật epoch state
của model nếu có `set_epoch` · Thay đổi
`requires_grad` của tham số (freeze) · Có thể **xóa** checkpoint resume cục bộ sau
khi nạp (`:1129`)

## Error / edge cases

| Tình huống | Hành vi |
|---|---|
| `selection_metric` không có trong eval stats | `raise KeyError` kèm danh sách metric có sẵn (`:675`) — hỏng **ồn ào**, không âm thầm |
| `best_agg_metric` resume vào sai mode (`inf` dưới mode `max`) | ⚠ **không raise** — `checkpoint_best` lặng lẽ không bao giờ được ghi, xem [`_load_checkpoint`](runner_base.py.methods/_load_checkpoint.md) |
| Checkpoint resume không tồn tại | Raise từ `torch.load` |
| Early stop đạt patience | `break` khỏi vòng epoch |
| Resume từ file local | `output_dir` là thư mục cha của checkpoint; không tạo run timestamp mới |

## ⚠ Freeze-list mồ côi

```python
for token in ("mhcac", "aggregator", "cls_loss_fn")   # :189
```

`Blip2Qformer` **không còn tạo** attribute `aggregator`
([D-003](../../../_meta/DECISIONS.md#d-003--mhcac-variants-và-encoder-trùng-lặp-là-legacy)).
Danh sách này trỏ vào thứ không tồn tại.

**Hệ quả:** ai thêm lại `self.aggregator` sẽ thấy nó **tự động bị đóng băng** mà
không có thay đổi code nào ở chỗ khác. Ghi nhận là
[I2](../../../_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa).

`:1050` và `:1073` cũng có logic đóng băng theo module (`self.model.mhcac.parameters()`;
dòng `aggregator` đã comment).

## Related tests

`tests/test_training_core.py` (scheduler giữ `lr_scale`, đuôi accumulation) ·
`tests/test_stage1_eval_hook.py` ⚠ cần `model.lavis` ·
logic resume-directory được bảo vệ bởi hành vi `setup_output_dir` và cần giữ khi sửa resume

## Related documentation

[PIPELINES.md → P1](../../../_meta/PIPELINES.md#p1--stage-1-pretraining) ·
[`image_text_pretrain.py`](../tasks/image_text_pretrain.py.doc.md)

## Developer notes

1. **`selection_metric` thực tế là `loss`** (YAML `:366`, xác minh 2026-08-15).
   Ghi chú cũ ở đây nói `macro_auprc` và kết luận `CLAUDE.md`/`README.md` sai —
   ghi chú đó **nay đã lỗi thời**, config đã đổi từ lúc nó được viết. Nguyên tắc
   vẫn đúng: đọc YAML, đừng tin doc.
   ⚠ `loss` có thiên lệch đã biết (bị các nhãn phổ biến chi phối) — xem `CLAUDE.md`.
2. **Đổi `selection_metric` khi resume sẽ vô hiệu hoá `checkpoint_best` trong im
   lặng** nếu không trung hoà `best_agg_metric` trong file checkpoint. Đây là bẫy
   nguy hiểm nhất của module này —
   [`_load_checkpoint`](runner_base.py.methods/_load_checkpoint.md) mô tả đầy đủ
   cơ chế và cách vá.
3. **Test split không tham gia chọn checkpoint** ở bất kỳ bước nào. Nó chỉ chạy
   một lần trong `evaluate(cur_epoch="best")`.
4. **`lr_scheduler.step()` gọi mỗi optimizer update**, không phải mỗi microbatch —
   nên `warmup_steps` đếm bằng update.
5. **`save_freq: 0`** giữ **chỉ** best + last. Một run 20 epoch với `save_freq: 5`
   để lại 4 checkpoint theo epoch + 2.
5. Sửa file này ảnh hưởng mọi Stage 1 run và gián tiếp Stage 2 mode Q-Former.
6. `train_epoch` gọi `set_epoch` trên `_model` gốc, không phụ thuộc DDP wrapper;
   hook có `hasattr` để giữ tương thích với model khác.

## Source relationships

- **Parent:** [`model/lavis/_index.md`](../_index.md)
- **Methods:** [`runner_base.py.methods/`](runner_base.py.methods/)
- **Related:** [`blip2_qformer.py`](../models/blip2_models/blip2_qformer.py.doc.md) · [`train.py`](../../../pretraining/train.py.doc.md)

← [HOME](../../../../HOME.md)
