> Source: `preporcessing/preprocess_mimic_cxr.py` (420 dòng)
> Status: ✅ ACTIVE — ★ thượng nguồn của mọi thứ
> Last verified against source: 2026-08-12

# `preporcessing/preprocess_mimic_cxr.py`

## Purpose
Dựng ba split CSV mà **cả Stage 1 và Stage 2** tiêu thụ. CPU-only, không đọc ảnh.

## Entry point
```bash
python preporcessing/preprocess_mimic_cxr.py \
    --raw-dir <mimic-cxr-raw> --reports-root <reports/files> \
    --output-dir <processed/full_allviews>
# --views frontal | --limit-studies N
```

## Main functions
| Hàm | Dòng | Vai trò |
|---|---|---|
| `main()` | 258 | ★ |
| `parse_args()` | 55 | |
| `clean_chexpert(df)` | 76 | 14 nhãn; trả cả danh sách cột |
| `clean_metadata(df, views)` | 111 | Lọc view, giữ `ViewPosition` |
| `clean_split(df)` | 150 | Split gốc PhysioNet |
| `build_study_text(studies, reports_root, workers)` | 194 | ★ Đọc + parse report song song |
| `report_path(reports_root, subject_id, study_id)` | 189 | |
| `build_image_path(subject_id, study_id, dicom_id)` | 252 | ★ **Đường dẫn TƯƠNG ĐỐI** |

## ★ `build_image_path` — tương đối, không tuyệt đối
Docstring `:253`: *"Relative path; ReportDataset joins it onto vis_root."*
`ReportDataset._row_visual:637` **raise** nếu gặp đường dẫn tuyệt đối. Đây là chốt
bảo mật, không phải quy ước.

## Outputs
`train.csv` / `val.csv` / `test.csv` — patient- **và** study-disjoint, kèm
`target_valid`, `classification_valid`, `impression_clean`, `ViewPosition`, 14 cột CheXpert.

## Side effects
Đọc `.csv.gz` + 227.835 file `.txt`; ghi 3 CSV. Không GPU.

## Error / edge cases
Report thiếu → `target_valid=False` · Thiếu nhãn CheXpert → `classification_valid=False`
Cả hai vẫn giữ dòng lại, chỉ mask ở loss tương ứng.

## Related tests
`tests/test_mimic_data_pipeline.py` (gián tiếp, qua study index)

## Developer notes
1. ⚠ Đầu ra là **dữ liệu PhysioNet dẫn xuất**. `.gitignore` chặn `train.csv`,
   `val.csv`, `test.csv`, `*_split.csv`, `splits/`. **Không commit.**
2. Chạy `python -m training.dataio.validate_manifest` ngay sau khi dựng.
3. Manifest cũ thiếu cột impression → phải chạy lại script này.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
