> Source: `pretraining/train.py:68-70`
> Status: ⚠ ĐƯỢC ĐỊNH NGHĨA NHƯNG KHÔNG ĐƯỢC GỌI
> Last verified against source: 2026-08-12

# `get_runner_class(cfg)`

## Located in

[`pretraining/train.py`](../train.py.doc.md)

## ⚠ Trạng thái — đọc trước

Hàm này **không được gọi ở bất kỳ đâu**. `main()` dựng `RunnerBase` trực tiếp:

```python
runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
# train.py:163 — KHÔNG đi qua get_runner_class
```

Kiểm chứng:
```bash
grep -n "get_runner_class" pretraining/train.py
# → chỉ dòng định nghĩa :68
```

**Hệ quả thực tế:** đặt `run.runner: runner_iter` trong YAML **không có tác dụng
gì**. Config key đó bị đọc ở đây, nhưng kết quả bị vứt bỏ.

## Purpose (ý đồ ban đầu)

Cho phép chọn runner class qua config thay vì hardcode. LAVIS có ít nhất hai
runner: `runner_base` (lặp theo epoch) và `runner_iter` (lặp theo iteration,
`model/lavis/runners/runner_iter.py`, 344 dòng).

## Signature

```python
def get_runner_class(cfg) -> type
```

## Parameters

### `cfg`
- Type: `Config` · Đọc `cfg.run_cfg.get("runner", "runner_base")`

## Returns

- Type: `type` — class runner từ registry
- Mặc định `RunnerBase` khi `run.runner` không được đặt

## Execution flow

```text
cfg.run_cfg.get("runner", "runner_base")
   ↓
registry.get_runner_class(name)
   ↓
return class (KHÔNG khởi tạo)
```

## Dependencies

`model/lavis/common/registry.py::registry.get_runner_class`

## Called by

**Không ai.** Đây là code chết trong phạm vi file này.

## Side effects

Không (nếu được gọi). Chỉ tra registry.

## Error handling

Không có. Tên runner không tồn tại → `registry.get_runner_class` trả `None` hoặc
raise ⚠ (cần kiểm tra `registry.py` để chắc).

## Config dependencies

`run.runner` — **hiện không có hiệu lực**.

## Modification risk

Muốn kích hoạt lại: đổi `train.py:163` thành

```python
runner_cls = get_runner_class(cfg)
runner = runner_cls(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
```

⚠ Trước khi làm: `RunnerBase` chứa **logic riêng của Meta-CXR** —
`selection_metric`, `early_stop_patience`, `save_freq`, và freeze-list
(`:189`). `runner_iter` chưa chắc có những thứ đó. Đổi runner có thể âm thầm mất
early stopping và checkpoint selection.

## Related methods

[`main()`](main.md)

← [`train.py`](../train.py.doc.md) · [LEGACY_AND_OPTIONAL](../../_meta/LEGACY_AND_OPTIONAL.md) · [HOME](../../../HOME.md)
