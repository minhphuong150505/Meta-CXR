> Source: `local_config.py` (29 dòng)
> Status: ✅ ACTIVE — ★
> Last verified against source: 2026-08-12

# `local_config.py`

## Purpose
Nạp `configs/env_config.yaml` và phơi ra thành hằng số Python.

## Why it exists
Tách **đường dẫn của máy** khỏi **siêu tham số của run**. Nhờ vậy cùng một YAML
run chạy được trên nhiều máy.

## ★ Fail nhanh và rõ
```python
if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"env_config.yaml not found at {_CONFIG_PATH}. "
        "Copy configs/env_config.yaml.example to configs/env_config.yaml ...")
```
Raise **lúc import module**, trước cả khi `main()` chạy, và thông điệp nói đúng
việc cần làm.

## Hằng số xuất ra
`PATH_TO_MIMIC_CXR` · `VIS_ROOT` ★ · `SPLIT_CSV` · `REPORTS_CSV` · `CHEXPERT_CSV` ·
`METADATA_CSV` · `PROCESSED_TRAIN_CSV` · `PROCESSED_VAL_CSV` · `PROCESSED_TEST_CSV` ·
`OUTPUT_DIR` · `JAVA_HOME` · `JAVA_PATH` · `WANDB_ENTITY` · `WANDB_PROJECT`

## Calls / Called by
Gọi: `omegaconf.OmegaConf` (`resolve=True` → biến nội suy được giải).
Được gọi: `pretraining/train.py:22`, `inference.py:9`,
`training/dataio/validate_manifest.py`, `medgemma_inference/run_pretrained_findings.py`,
`scripts/vm_preflight.py`.

## Side effects
Đọc YAML lúc import module.

## Error / edge cases
Thiếu file → `FileNotFoundError` · Thiếu key → `KeyError` **không bắt** (fail rõ hơn
là trả `None` rồi hỏng ở nơi khác)

## Related documentation
[`configs/env_config.yaml.example`](configs/env_config.yaml.example.doc.md) ·
[ARCHITECTURE.md §6](_meta/ARCHITECTURE.md#6-cấu-hình-phân-tầng)

## Developer notes
⚠ `VIS_ROOT` phải trỏ vào thư mục chứa **trực tiếp** `files/` (mirror bucket root),
không phải chính `files/`. Đây là lỗi cấu hình hay gặp nhất.

← [HOME](../HOME.md)
