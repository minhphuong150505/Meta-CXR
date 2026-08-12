> Source: `threshold.json` (config)
> Status: ⚠ POTENTIALLY_UNUSED — historical threshold artifact
> Last verified against source: 2026-08-12

# `threshold.json`

## Purpose
Threshold phân loại per-abnormality cũ. File có đúng schema mà
`fig9.load_thresholds(path)` và `inference.classify_abnormalities(..., thresholds)`
chấp nhận, nhưng **không có caller hiện tại nào truyền chính file này**.

## ⚠ Không bao giờ được load ngầm
Comment source tại `fig9:192-194` nói rõ các giá trị lịch sử không mang provenance
Stage-1 checkpoint/validation và không được load tự động. CLI chỉ đọc file do
người chạy chọn bằng `--threshold-path`; mặc định là argmax.

Đây là chủ ý: một threshold sai lặng lẽ áp vào sẽ đổi mọi con số P/N/U mà không ai
biết.

## Quan hệ với threshold calibrate được
| | |
|---|---|
| `threshold.json` (file này) | Artifact lịch sử, provenance chưa được ghi trong file |
| `<run>/result/f1_thresholds.json` | Do `scripts/calibrate_thresholds.py` sinh, **chỉ từ validation** |

Hai thứ khác nhau. Đừng lẫn.

## Consumer
Không có direct caller của path `threshold.json`. Các API có thể nhận **một path
do caller chọn**: `inference.classify_abnormalities(..., thresholds=...)` và CLI
Stage 2 `--threshold-path` → `fig9.load_thresholds` →
`Stage1Context.threshold_for(abnormality)`.

## Developer notes
Threshold dùng để báo cáo kết quả **phải** đến từ calibrate trên validation và
gắn đúng checkpoint, không mặc định dùng file tĩnh này. Status chỉ là provisional;
chưa có user confirmation để gọi nó LEGACY hay UNUSED.

← [HOME](../HOME.md)
