> Source: `model/`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `model/`

## Purpose

`lavis/` — fork LAVIS đã sửa, nơi model Stage 1 thực sự sống. (`pretrained_medgemma/`,
loader MedGemma bên thứ ba cho baseline P8, đã xóa 2026-11-27 theo D-029.)

## Role in project

```text
model/lavis/               → Stage 1 (P1)
```

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| Thư mục | Doc | Status | Nội dung |
|---|---|---|---|
| `lavis/` | [📁](lavis/_index.md) | ✅ | 24 file, 10.862 LOC — fork Salesforce LAVIS |
| `__init__.py` | — | ✅ | Rỗng |

## Main responsibilities

Không có — đây là thư mục gom nhóm. Trách nhiệm nằm ở hai thư mục con.

## Entry points

Không có trực tiếp.

## Dependencies

Xem `_index.md` của từng thư mục con.

## Used by

`pretraining/train.py`, `training/stage1/lavis_loader.py` (→ `lavis/`).

## Status

```text
✅ ACTIVE
```

## Notes

- ⚠ `tests/conftest.py` đăng ký `model` và `model.lavis` là **package
  path-only**, để import một submodule không chạy `model/lavis/__init__.py` —
  file đó kéo theo cả stack GPU. **Không có conftest này thì test suite không
  collect nổi trên máy CPU.**

- `model/lavis/data/` nằm trong `.gitignore` nhưng `ReportDataset.py` **đã được
  track từ trước**, nên `git add .` sẽ không bắt thay đổi ở đó. Dùng
  `git add -f <path>`.

## Related documentation

[`lavis/_index.md`](lavis/_index.md) ·
[ARCHITECTURE.md](../_meta/ARCHITECTURE.md)

← [Về HOME](../../HOME.md)
