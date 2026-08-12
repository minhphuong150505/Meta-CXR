> Source: `training/trainer/state.py` (174 dòng)
> Status: ❓ UNKNOWN — chỉ test import
> Last verified against source: 2026-08-12

# `training/trainer/state.py`

## Purpose

`TrainingState` + `RngSnapshot` — trạng thái huấn luyện tuần tự hóa được, gồm cả
RNG để resume **tái lập bit-for-bit**.

## ⚠ Không có caller production

Xem [D-001](../../_meta/DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline).

## Status

```text
❓ UNKNOWN
```

## Main items

### `RngSnapshot` (`:32`)

| Method | Dòng | Vai trò |
|---|---|---|
| `.capture(data_generator=None)` | 42 | ★ Chụp RNG Python/NumPy/torch (+ generator của DataLoader) |
| `.restore(data_generator=None)` | 51 | Khôi phục |
| `.to_dict()` / `.from_dict(payload)` | 64, 74 | Tuần tự hóa |

**Đây là thứ hai hệ checkpoint kia không có.** Không chụp RNG thì resume cho kết
quả khác với run liền mạch, và không ai truy được vì sao.

### `TrainingState` (`:91`)

Trường: `epoch`, `micro_step`, `global_step`, `best_score`, `lower_is_better`,
`rng`, `provenance`.

| Method | Dòng | Vai trò |
|---|---|---|
| `.is_improvement(score)` | 104 | Theo `lower_is_better` |
| `.record_score(score)` | 107 | Cập nhật best, trả bool |
| `.should_stop(patience)` | 116 | Early stopping |
| `.to_dict()` / `.from_dict()` | 119, 133 | ⚠ `from_dict` **raise** với payload sai (có test) |

### `git_sha(default="unknown")` (`:78`) và `build_provenance(...)` (`:154`)

Gắn commit hash vào checkpoint. Sau này biết checkpoint sinh từ code nào.

## Calls / Called by

Gọi: `torch`, `random`, `numpy`, `subprocess` (git).
Được gọi: `trainer/checkpointing.py:24`; `tests/test_trainer_resume.py`.

## Side effects

`.restore()` **mutate RNG toàn cục**. `git_sha()` chạy subprocess.

## Error / edge cases

`from_dict` với payload thiếu/sai version → raise (`tests/test_trainer_resume.py:228-231`) ·
Ngoài git repo → `git_sha` trả `"unknown"`

## Related tests

`tests/test_trainer_resume.py` — `RngSnapshot.capture`, `lower_is_better`,
`from_dict` raise, provenance

## Developer notes

`micro_step` **và** `global_step` là hai thứ khác nhau — cái đầu đếm microbatch,
cái sau đếm optimizer update. Resume giữa epoch cần cả hai.

## Source relationships

- **Parent:** [`training/trainer/`](_index.md)
- **Related:** [`checkpointing.py`](checkpointing.py.doc.md)

← [HOME](../../../HOME.md)
