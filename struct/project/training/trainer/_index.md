> Source: `training/trainer/` (178 LOC)
> Status: ❓ UNKNOWN — chưa có caller production
> Last verified against source: 2026-08-12

# `training/trainer/`

## ⚠ Trạng thái — đọc trước

**Không có caller production nào.** `CheckpointManager`, `TrainingState`,
`RngSnapshot` chỉ được import bởi `tests/test_trainer_resume.py`.

```bash
grep -rn "CheckpointManager\|TrainingState\|RngSnapshot" --include='*.py' . \
  | grep -v '^./training/trainer/'
# → chỉ tests/test_trainer_resume.py
```

Trong khi đó:
- Stage 1 dùng `RunnerBase._save_checkpoint` (`model/lavis/runners/runner_base.py:931`)
- Stage 2 dùng logic riêng trong `training/stage2_utils.py`

Không gắn nhãn LEGACY vì code chất lượng cao, test kỹ, và làm được thứ hai bên
kia **không** làm: snapshot trạng thái RNG để resume tái lập bit-for-bit.

Quyết định: [D-001](../../_meta/DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline)
— user chưa xác nhận được đây là hạ tầng chuẩn bị hay thiết kế bị bỏ dở.

## Purpose (theo thiết kế)

Quản lý checkpoint có khả năng resume **chính xác**: không chỉ trọng số, mà cả
optimizer, scheduler, AMP scaler, số bước, điểm tốt nhất, và trạng thái RNG.

## Role in project

Chưa nằm trong pipeline nào.

## Parent

[`training/`](../_index.md)

## Children

| File | Doc | Nội dung |
|---|---|---|
| `checkpointing.py` | [📄](checkpointing.py.doc.md) | `CheckpointManager` — `save`, `load`, `is_resumable`, `subdir` last/best |
| `state.py` | [📄](state.py.doc.md) | `TrainingState` (epoch, micro_step, global_step, best_score, lower_is_better, provenance), `RngSnapshot.capture/from_dict` |

## Main responsibilities

1. Lưu/nạp `trainer_state.pt` cùng optimizer + scheduler.
2. Chụp và khôi phục RNG (`RngSnapshot.capture(data_generator)`).
3. Phân biệt checkpoint `last` và `best` qua `subdir`.
4. Mang `provenance` trong state.

## Entry points

Không có.

## Dependencies

`torch` · `training/torch_io.load_torch_checkpoint` (`:23`) ·
`training/trainer/state` (`:24`)

## Used by

**Chỉ** `tests/test_trainer_resume.py` (309 dòng — test rất kỹ: resume giữa epoch,
lower_is_better, thiếu optimizer, thiếu file, subdir last/best, provenance).

## Execution flow (nếu được dùng)

```text
CheckpointManager(dir).save(state, optimizer, scheduler, data_generator, subdir=…)
        ↓
state.rng = RngSnapshot.capture(data_generator)
        ↓
torch.save({"state": state.to_dict(), "optimizer": …, "scheduler": …})
        ↓
… crash / dừng …
        ↓
CheckpointManager(dir).load()  →  TrainingState.from_dict(payload["state"])
        ↓
FileNotFoundError nếu không có gì để resume
```

## Important configurations

Không đọc config nào — API thuần Python.

`TRAINER_STATE_FILE = "trainer_state.pt"` (`checkpointing.py:26`).

⚠ Tên file này **trùng** với thứ Stage 2 kiểm tra:
`run_medgemma_qlora.py:183` yêu cầu `trainer_state.pt` tồn tại để coi adapter là
resume được. Nhưng file đó do `stage2_utils.py` ghi, **không phải** module này.
Trùng tên, khác đường code.

## Status

```text
❓ UNKNOWN
```

## Notes

- Nếu bạn định thêm resume cho Stage 2, **hãy dùng module này** thay vì viết mới
  — nó đã có test đầy đủ.
- Nếu bạn xác nhận nó không cần nữa, cập nhật [D-001](../../_meta/DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline)
  và chuyển sang [LEGACY_AND_OPTIONAL.md](../../_meta/LEGACY_AND_OPTIONAL.md).

## Related documentation

[LEGACY_AND_OPTIONAL.md §U1](../../_meta/LEGACY_AND_OPTIONAL.md#u1--trainingtrainer) ·
[CALL_GRAPH.md §5](../../_meta/CALL_GRAPH.md#5-bottom-up--ai-gọi-cái-này)

← [`training/`](../_index.md) · [HOME](../../../HOME.md)
