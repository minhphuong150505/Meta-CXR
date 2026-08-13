> Source: `preporcessing/` (3 file Python)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-13

# `preporcessing/`

> ⚠ **Tên thư mục sai chính tả** (`preporcessing`, không phải `preprocessing`).
> Đây là tên thật trong tree. **Giữ nguyên** — đổi tên sẽ làm hỏng mọi tài liệu,
> script và đường dẫn đang tham chiếu tới nó.

## Purpose

Từ CSV thô + report `.txt` dựng ra ba split CSV mà **cả Stage 1 và Stage 2 đều
tiêu thụ**; đồng thời dựng cache mask giải thích riêng cho recipe Stage 1
explanation-aware. Cả hai đường đều CPU-only và không đọc pixel ảnh gốc.

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

CheXmask RLE + MS-CXR bbox + ba split CSV
        ↓
build_explanation_masks.py
        ↓
masks_<split>.npy + index_<split>.json
        ↓
MIMIC_CXR_Dataset → explanation loss
```

Đây là **thượng nguồn của mọi thứ**. Một lỗi ở đây lan vào cả hai stage.

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| File | LOC | Doc | Vai trò |
|---|---|---|---|
| `preprocess_mimic_cxr.py` | 420 | [📄](preprocess_mimic_cxr.py.doc.md) | ★ Dựng split |
| `mimic_report_parser.py` | — | [📄](mimic_report_parser.py.doc.md) | Trích FINDINGS / IMPRESSION |
| `build_explanation_masks.py` | 786 | [📄](build_explanation_masks.py.doc.md) | 🟡 Cache CheXmask/MS-CXR hai tầng |

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
6. Từ split project, dựng mask phổi/bbox 112² với cùng geometry ảnh train.

## Entry points

```bash
python preporcessing/preprocess_mimic_cxr.py \
    --raw-dir ~/data/mimic-cxr-raw \
    --reports-root ../Report/mimic-cxr-reports/files \
    --output-dir ~/data/mimic-cxr-processed/full_allviews

# --views frontal      chỉ PA/AP
# --limit-studies N    smoke run

# Xác nhận schema và khoảng Dice trước khi quét file 13 GB
python preporcessing/build_explanation_masks.py --inspect

# Smoke cache một split trên private storage
python preporcessing/build_explanation_masks.py \
    --split val --limit 200 --output-dir <private-cache-dir>
```

Kiểm tra sau khi dựng:
```bash
python -m training.dataio.validate_manifest --section-mode findings_and_impression
```

## Dependencies

`pandas`, `numpy`, `Pillow`, stdlib. **Không** torch/torchvision, không LAVIS.

## Used by

Không module nào import các script — chúng chạy một lần. Nhưng đường train phụ
thuộc vào split output; explanation-aware train còn phụ thuộc mask cache.

## Important configurations

| Cột đầu ra | Ai đọc |
|---|---|
| `image_path` (**tương đối**) | `ReportDataset._row_visual` (nối với `vis_root`) |
| `ViewPosition` | anchor selection, `_view_id` |
| `findings`, `impression_clean` | `text_output`, Stage 2 records |
| `target_valid` | → `generation_mask` |
| `classification_valid` | → `classification_mask` |
| 14 cột CheXpert | `classification_labels` |
| `masks_<split>.npy` + `index_<split>.json` | `explanation_mask`, validity, source |

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

- ⚠ Cache mask cũng là dẫn xuất MIMIC-CXR. Script không log identifier và output
  chỉ được đặt trên private storage.

- ⚠ Split trong MS-CXR không phải split project. Builder luôn map qua ba manifest
  và in số dòng lệch để leakage không xảy ra âm thầm.

## Related documentation

[DATA_FLOW.md §1](../_meta/DATA_FLOW.md#1-preprocessing--split-csv) ·
[PIPELINES.md → P3](../_meta/PIPELINES.md#p3--preprocessing--dựng-split) ·
[`training/dataio/_index.md`](../training/dataio/_index.md)

← [Về HOME](../../HOME.md)
