> Source: `training/trainer/checkpointing.py` (120 dòng)
> Status: ❓ UNKNOWN — chỉ test import
> Last verified against source: 2026-08-12

# `training/trainer/checkpointing.py`

## Purpose

`CheckpointManager` — lưu/nạp trạng thái huấn luyện đầy đủ (model state qua
optimizer/scheduler + `TrainingState` + RNG) một cách **atomic**.

## ⚠ Không có caller production

Chỉ `tests/test_trainer_resume.py`. Stage 1 dùng `RunnerBase._save_checkpoint`;
Stage 2 dùng logic riêng trong `stage2_utils.py`.
[D-001](../../_meta/DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline).

## Status

```text
❓ UNKNOWN
```

## Main items

| Tên | Dòng | Vai trò |
|---|---|---|
| `atomic_save(payload, path)` | 29 | ★ Ghi tạm rồi rename — crash không để lại file hỏng |
| `CheckpointManager` | 37 | |
| `.state_path(subdir="")` | 48 | Đường dẫn `trainer_state.pt` |
| `.save(state, optimizer, scheduler, data_generator, subdir)` | 52 | Chụp RNG rồi ghi |
| `.load(...)` | 75 | Khôi phục; `FileNotFoundError` nếu không có |
| `.is_resumable(subdir)` | 110 | Kiểm nhanh |
| `.verify_resumable(required_files, subdir)` | 113 | Kiểm kỹ, nêu file thiếu |

`TRAINER_STATE_FILE = "trainer_state.pt"` (`:26`)

⚠ **Trùng tên với thứ Stage 2 kiểm** (`run_medgemma_qlora.py:183`, `:204`) — nhưng
file đó do `stage2_utils` ghi, **không phải** module này. Trùng tên, khác đường code.

## Calls / Called by

Gọi: `torch`, `training.torch_io.load_torch_checkpoint` (`:23`), `trainer.state` (`:24`).
Được gọi: **chỉ** `tests/test_trainer_resume.py`.

## Side effects

Ghi file (atomic). Không mutate model.

## Error / edge cases

Không có state để resume → `FileNotFoundError` (`:86`) · `verify_resumable` nêu
file thiếu · Không truyền optimizer → vẫn lưu được (có test)

## Related tests

`tests/test_trainer_resume.py` (309 dòng) — resume giữa epoch, subdir last/best,
thiếu optimizer, provenance

## Developer notes

Nếu thêm resume cho Stage 2, **dùng module này** thay vì viết mới — nó đã có test
đầy đủ và có `atomic_save`.

## Source relationships

- **Parent:** [`training/trainer/`](_index.md)
- **Related:** [`state.py`](state.py.doc.md)

← [HOME](../../../HOME.md)
