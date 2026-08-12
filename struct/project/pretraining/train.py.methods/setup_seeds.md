> Source: `pretraining/train.py:59-66`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `setup_seeds(config)`

## Located in

[`pretraining/train.py`](../train.py.doc.md)

## Purpose

Làm run tái lập được, và làm **mỗi rank DDP có seed khác nhau**.

Điểm thứ hai quan trọng hơn nó có vẻ: nếu mọi rank dùng chung một seed, chúng sẽ
sinh ra cùng chuỗi số ngẫu nhiên — cùng thứ tự augmentation, cùng dropout mask —
làm mất phần lớn lợi ích của việc chạy nhiều process.

## Signature

```python
def setup_seeds(config) -> None
```

## Parameters

### `config`
- Type: `Config` (LAVIS) · Required: ✅
- Chỉ đọc **một** trường: `config.run_cfg.seed`
- Production: `seed: 42`

## Returns

`None`.

## Local variables

### `seed`
```python
seed = config.run_cfg.seed + get_rank()
```
Rank 0 → 42, rank 1 → 43, … Chênh lệch nhỏ nhưng đủ để các luồng RNG phân kỳ.

## Execution flow

```text
seed = run.seed + get_rank()
   ↓
random.seed(seed)          ← Python stdlib
numpy.random.seed(seed)    ← NumPy
torch.manual_seed(seed)    ← PyTorch (CPU và CUDA)
   ↓
cudnn.benchmark = False
cudnn.deterministic = True
```

## Detailed logic

**Ba RNG riêng biệt.** Python, NumPy và PyTorch mỗi cái một luồng số ngẫu nhiên.
Bỏ sót một cái là mất tái lập ở đúng chỗ đó. `MIMIC_CXR_Dataset` dùng `random`
(shuffle view) và torchvision transform dùng RNG của PyTorch.

**`cudnn.benchmark = False`** tắt việc cuDNN tự dò thuật toán conv nhanh nhất.
Autotuning cho kết quả khác nhau giữa các lần chạy → mất tái lập. Đổi lại: chậm hơn.

**`cudnn.deterministic = True`** ép chọn kernel xác định. Cũng chậm hơn.

Cả hai là **đánh đổi có chủ đích: tái lập > tốc độ**.

⚠ Vẫn còn nguồn bất định chưa bị chặn: `torch.use_deterministic_algorithms` không
được bật, và DataLoader `worker_init_fn` không được đặt (`num_workers: 4`). Nên
run **không** tái lập bit-for-bit tuyệt đối. ⚠ Cần runtime verification để biết
mức độ lệch.

## Data / Tensor flow

Không đụng tensor.

## Side effects

⚠ **Toàn cục theo process.** Thay đổi trạng thái RNG của cả ba thư viện và cấu
hình cuDNN. Mọi code chạy sau đó trong cùng process đều bị ảnh hưởng.

## Error handling

Không có. `config.run_cfg.seed` thiếu → `AttributeError`/`KeyError` không bắt.

## Config dependencies

`run.seed` (production: `42`).

## Related methods

[`main()`](main.md) — gọi nó ở `:88`, ngay sau `init_distributed_mode` (bắt buộc
theo thứ tự này, vì `get_rank()` cần process group đã khởi tạo).

## Tests

Không có unit test trực tiếp.

## Modification risk

- Bỏ `+ get_rank()` → mọi rank DDP đồng bộ RNG, giảm hiệu quả đa process.
- Bật `cudnn.benchmark = True` → nhanh hơn, **mất tái lập**.
- Gọi trước `init_distributed_mode` → `get_rank()` sai.

← [`train.py`](../train.py.doc.md) · [`main()`](main.md) · [HOME](../../../HOME.md)
