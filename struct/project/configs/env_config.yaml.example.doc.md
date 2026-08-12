> Source: `configs/env_config.yaml.example` (config)
> Status: ✅ ACTIVE — ★ bắt buộc
> Last verified against source: 2026-08-12

# `configs/env_config.yaml.example`

## Purpose
Mẫu cho `configs/env_config.yaml` — đường dẫn **của máy**, tách khỏi siêu tham số
của run.

## ★ Bắt buộc trước mọi thứ
```bash
cp configs/env_config.yaml.example configs/env_config.yaml
```
Thiếu → `local_config.py:6` raise `FileNotFoundError` **lúc import**, trước cả `main()`.

## Các key và ai đọc
| Key | Biến trong `local_config` | Consumer |
|---|---|---|
| `paths.data_root` | `PATH_TO_MIMIC_CXR` | |
| `paths.mimic_cxr_jpg_root` | `VIS_ROOT` | ★ `ReportDataset._row_visual` — **phải trỏ vào thư mục chứa TRỰC TIẾP `files/`** |
| `paths.split_csv` | `SPLIT_CSV` | |
| `paths.reports_csv` | `REPORTS_CSV` | |
| `paths.chexpert_csv` | `CHEXPERT_CSV` | |
| `paths.metadata_csv` | `METADATA_CSV` | |
| `paths.processed_train_csv` | `PROCESSED_TRAIN_CSV` | Stage 1 + Stage 2 |
| `paths.processed_val_csv` | `PROCESSED_VAL_CSV` | |
| `paths.processed_test_csv` | `PROCESSED_TEST_CSV` | |
| `paths.output_dir` | `OUTPUT_DIR` | |
| `java.home` / `java.path` | `JAVA_HOME` / `JAVA_PATH` | `inference.py` — CheXpert labeler |
| `wandb.entity` / `wandb.project` | `WANDB_ENTITY` / `WANDB_PROJECT` | `pretraining/train.py` |

## ⚠ Git
`configs/env_config.yaml` **được git-ignore ở checkout này**. Sửa file `.example`
cho thứ dùng chung; đừng commit path hay credential của máy.

## Related documentation
[`local_config.py`](../local_config.py.doc.md) · [ARCHITECTURE.md §6](../_meta/ARCHITECTURE.md#6-cấu-hình-phân-tầng)

## Developer notes
`mimic_cxr_jpg_root` sai là lỗi hay gặp nhất: nó phải là **mirror của bucket root**,
tức thư mục chứa trực tiếp `files/`, không phải chính `files/`.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
