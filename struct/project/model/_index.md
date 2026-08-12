> Source: `model/`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `model/`

## Purpose

Hai thứ **hoàn toàn không liên quan nhau**, chỉ tình cờ cùng thư mục cha:

1. `lavis/` — fork LAVIS đã sửa, nơi model Stage 1 thực sự sống.
2. `pretrained_medgemma/` — loader cho checkpoint MedGemma **của bên thứ ba**,
   dùng cho baseline inference (P8).

Chúng không import lẫn nhau và thuộc hai pipeline khác nhau.

## Role in project

```text
model/lavis/               → Stage 1 (P1) và demo Vicuna (P9)
model/pretrained_medgemma/ → baseline external MedGemma (P8)
```

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| Thư mục | Doc | Status | Nội dung |
|---|---|---|---|
| `lavis/` | [📁](lavis/_index.md) | ✅ | 24 file, 10.862 LOC — fork Salesforce LAVIS |
| `pretrained_medgemma/` | [📁](pretrained_medgemma/_index.md) | ✅ | 6 file, 526 LOC — checkpoint ngoài |
| `__init__.py` | — | ✅ | Rỗng |

## Main responsibilities

Không có — đây là thư mục gom nhóm. Trách nhiệm nằm ở hai thư mục con.

## Entry points

Không có trực tiếp.

## Dependencies

Xem `_index.md` của từng thư mục con.

## Used by

`pretraining/train.py`, `inference.py`, `training/stage1/lavis_loader.py`
(→ `lavis/`); `medgemma_inference/` (→ `pretrained_medgemma/`).

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

[`lavis/_index.md`](lavis/_index.md) · [`pretrained_medgemma/_index.md`](pretrained_medgemma/_index.md) ·
[ARCHITECTURE.md](../_meta/ARCHITECTURE.md)

← [Về HOME](../../HOME.md)
