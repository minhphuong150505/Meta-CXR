> Source: `medgemma_inference/progress.py:55-102`
> Status: ✅ ACTIVE

# `ProgressFile`

## Responsibility

Sidecar JSON nhỏ mang immutable `RunIdentity`, cumulative elapsed time và trạng
thái finished. Prediction JSONL vẫn là source of truth cho sample đã xong.

## Lifecycle

- `open(output_dir, identity)`: mkdir; nếu file đã có thì `assert_matches` trước
  khi cho append run mới.
- `read()`: JSON hỏng do crash trả `None`, vì có thể phục hồi từ predictions.
- `prior_elapsed_seconds()`: đưa runtime lần trước vào `BudgetState`.
- `write(...)`: ghi `.json.tmp` rồi atomic replace để không để half JSON.

## Modification risk

Không bỏ identity check hoặc `prior_elapsed_seconds`: lỗi đầu trộn hai experiment,
lỗi sau reset budget qua mỗi lần resume.

← [`progress.py`](../../progress.py.doc.md) · [HOME](../../../../HOME.md)
