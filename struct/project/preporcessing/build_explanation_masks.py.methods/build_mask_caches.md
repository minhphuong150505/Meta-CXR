> Source: `preporcessing/build_explanation_masks.py::build_mask_caches`
> Status: 🟡 CONDITIONAL

# `build_mask_caches(...)`

## Flow

```text
đọc ba manifest → chọn anchor study + map split project
  → đọc MS-CXR → đếm split mismatch → rasterize bbox target
  → stream CheXmask theo chunk → Dice gate → decode union phổi
    (`--limit` dừng sớm sau khi gặp đủ ID của smoke subset)
  → bbox override lung
  → compact valid rows → atomic replace .npy + JSON
```

## Safety invariants

- Không dùng cột split MS-CXR để định tuyến dữ liệu.
- Zero match giữa CheXmask và manifest → fail, không ghi join rỗng.
- Duplicate identifier, split overlap hoặc bbox dimension bất nhất → fail.
- Input thiếu → báo số file thiếu và nhắc kiểm tra private volume mount trước
  khi tạo output.
- Không log identifier/path/report.
- Temp file theo PID được dọn trong `finally`.

## Return / logging

Trả dict thống kê theo split. Stdout chỉ có số đếm/tỉ lệ lung, bbox, no-mask;
số dòng MS-CXR lệch split; và số dòng MS-CXR bị bỏ vì không có trong manifest.

← [`functions`](./_index.md)
