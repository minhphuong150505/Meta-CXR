> Source: `preporcessing/` (2 file)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `preporcessing/`

> ⚠ **Tên thư mục sai chính tả** (`preporcessing`, không phải `preprocessing`).
> Đây là tên thật trong tree. **Giữ nguyên** — đổi tên sẽ làm hỏng mọi tài liệu,
> script và đường dẫn đang tham chiếu tới nó.

## Purpose

Từ CSV thô + report `.txt` dựng ra ba split CSV mà **cả Stage 1 và Stage 2 đều
tiêu thụ**. CPU-only, không đọc ảnh.

## Role in project

```text
mimic-cxr-*.csv.gz + mimic-cxr-reports/files/
        ↓
preprocess_mimic_cxr.py  (+ mimic_report_parser.py)
        ↓
train.csv / val.csv / test.csv     ← patient- VÀ study-disjoint
        ↓
   ┌────┴────────────────────┐
MIMIC_CXR_Dataset      dataio/manifest.py
   (Stage 1)              (Stage 2, P8)
```

Đây là **thượng nguồn của mọi thứ**. Một lỗi ở đây lan vào cả hai stage.

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| File | LOC | Doc | Vai trò |
|---|---|---|---|
| `preprocess_mimic_cxr.py` | 420 | [📄](preprocess_mimic_cxr.py.doc.md) | ★ Dựng split |
| `mimic_report_parser.py` | — | [📄](mimic_report_parser.py.doc.md) | Trích FINDINGS / IMPRESSION |

Hai notebook trong thư mục này (`preprocessing_walkthrough.ipynb`,
`kltn-data-preprocessing.ipynb`) bị **git-ignore** vì outputs của chúng mang ID
thật. Chúng không có trong working tree đã kiểm tra.

## `mimic_report_parser.py` — vì sao quan trọng

Một tỉ lệ đáng kể report **không có tag FINDINGS**. Hành vi cũ là rơi về nguyên
văn cả báo cáo — tạo ra target rác (lẫn cả IMPRESSION, INDICATION, TECHNIQUE).

Parser hiện tại **khôi phục phần thân tường thuật** thay vì fallback mù.

## Main responsibilities

1. Ghép metadata + CheXpert labels + report text.
2. Trích FINDINGS và IMPRESSION riêng biệt.
3. Chia split **disjoint theo cả patient và study**.
4. Ghi `image_path` dạng **tương đối**.
5. Đặt cờ `target_valid` / `classification_valid` → thành mask lúc train.

## Entry points

```bash
python preporcessing/preprocess_mimic_cxr.py \
    --raw-dir ~/data/mimic-cxr-raw \
    --reports-root ../Report/mimic-cxr-reports/files \
    --output-dir ~/data/mimic-cxr-processed/full_allviews

# --views frontal      chỉ PA/AP
# --limit-studies N    smoke run
```

Kiểm tra sau khi dựng:
```bash
python -m training.dataio.validate_manifest --section-mode findings_and_impression
```

## Dependencies

`pandas`, stdlib. **Không** torch, không LAVIS.

## Used by

Không module nào import nó — nó là script chạy một lần. Nhưng **mọi thứ** phụ
thuộc vào đầu ra của nó.

## Important configurations

| Cột đầu ra | Ai đọc |
|---|---|
| `image_path` (**tương đối**) | `ReportDataset._row_visual` (nối với `vis_root`) |
| `ViewPosition` | anchor selection, `_view_id` |
| `findings`, `impression_clean` | `text_output`, Stage 2 records |
| `target_valid` | → `generation_mask` |
| `classification_valid` | → `classification_mask` |
| 14 cột CheXpert | `classification_labels` |

## Status

```text
✅ ACTIVE
```

## Notes

- ⚠ **`image_path` phải giữ dạng tương đối.** `_row_visual` (`ReportDataset.py:637-644`)
  chủ động **raise** nếu gặp đường dẫn tuyệt đối hoặc path traversal. Đây là chốt
  bảo mật, không phải kiểm tra hình thức.

- ⚠ **Manifest dựng trước 2026-07-21 thiếu `impression_clean` / `impression_valid` /
  `impression_token_count`** và không phục vụ được `--section-mode findings_and_impression`
  (mặc định). Phải chạy lại script này rồi upload lại.

- **Split phải disjoint theo cả patient và study.** `assert_no_leakage` trong
  `training/dataio/manifest.py` kiểm tra lại điều này ở phía tiêu thụ.

- ⚠ Đầu ra là **dữ liệu PhysioNet dẫn xuất**. `.gitignore` chặn `train.csv`,
  `val.csv`, `test.csv`, `*_split.csv`, `splits/`. **Không commit.**

## Related documentation

[DATA_FLOW.md §1](../_meta/DATA_FLOW.md#1-preprocessing--split-csv) ·
[PIPELINES.md → P3](../_meta/PIPELINES.md#p3--preprocessing--dựng-split) ·
[`training/dataio/_index.md`](../training/dataio/_index.md)

← [Về HOME](../../HOME.md)
