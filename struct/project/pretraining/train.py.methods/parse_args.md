> Source: `pretraining/train.py:41-56`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `parse_args()`

## Located in

[`pretraining/train.py`](../train.py.doc.md)

## Purpose

Đọc CLI. Ba tham số, không hơn — mọi thứ khác đi qua file YAML.

## Signature

```python
def parse_args() -> argparse.Namespace
```

## Parameters

Không có (đọc `sys.argv`).

## Returns

`argparse.Namespace` với ba thuộc tính:

### `cfg_path`
- Type: `str` · Required: **✅**
- Đường dẫn tới run YAML, ví dụ `pretraining/configs/mimic_cxr_full.yaml`
- Được `Config` đọc

### `local_rank`
- Type: `int` · Default: `0`
- Do `torchrun` truyền vào. `init_distributed_mode` thực tế đọc **biến môi trường**
  `LOCAL_RANK`, nên tham số này chủ yếu để tương thích ngược.

### `options`
- Type: `list[str]` · Default: `None` · `nargs="+"`
- Cặp `key=value` merge đè lên config, ví dụ `run.output_dir=/tmp/x run.run_name=y`
- Help text tự đánh dấu là **deprecate**, đề nghị dùng `--cfg-options` — nhưng
  `--cfg-options` **không tồn tại** trong parser này. ⚠ Help text sai.

## Local variables

`parser` — `ArgumentParser(description="Training")`.

## Execution flow

```text
ArgumentParser → add_argument × 3 → parse_args() → return
```

Không validate gì. `--cfg-path` không tồn tại sẽ nổ ở `Config`, không ở đây.

## Dependencies

`argparse` (stdlib).

## Called by

[`main()`](main.md) `:73` — `Config(parse_args())`

## Side effects

Không, ngoài việc `argparse` có thể `sys.exit(2)` khi thiếu `--cfg-path`.

## Error handling

Giao cho `argparse`: thiếu tham số bắt buộc → in usage → exit 2.

## Modification risk

Thêm tham số ở đây thì phải xử lý ở `main()` **và** kiểm tra `Config.__init__`
có bỏ qua thuộc tính lạ không. Lệnh train truyền `--cfg-path` và
`--options`; đổi tên chúng sẽ làm mọi lệnh train và script gọi nó hỏng.

← [`train.py`](../train.py.doc.md) · [`main()`](main.md) · [HOME](../../../HOME.md)
