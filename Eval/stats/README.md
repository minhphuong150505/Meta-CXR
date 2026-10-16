# Thống kê ba split — `processed/full_allviews_v2`

Sinh ngày 2026-08-16 bằng `split_stats.py`, chạy trên host có mount dữ liệu riêng.
Nguồn: `{train,val,test}.csv` + `mimic-cxr-2.0.0-chexpert.csv.gz` +
`/home/phuong/explanation_masks_v2/`.

**Chỉ chứa số liệu tổng hợp.** Không có `subject_id`, `study_id`, `dicom_id`,
đường dẫn ảnh hay nội dung báo cáo. `Test/` nằm trong `.gitignore` (dòng 131).

| File | Nội dung |
|---|---|
| `split_stats.py` | Script sinh ra toàn bộ, có tham số `--split` |
| `{train,val,test}_split_stats.json` | Số liệu máy đọc được |
| `{train,val,test}_label_distribution.csv` | Ma trận nhãn theo từng bệnh |

**Đối chứng:** script được chạy lại trên `test` và tái tạo **đúng từng con số**
của bản thống kê test làm ngày 2026-08-16 13:43 (CSV nhãn giống hệt, JSON không
lệch khoá nào). Chỉ sau đó mới dùng cho train/val.

---

## Quy mô

| | train | val | test |
|---|---:|---:|---:|
| ảnh | 368,960 | 2,991 | 5,159 |
| study | 222,758 | 1,808 | 3,269 |
| bệnh nhân | 64,586 | 500 | 293 |
| **study / bệnh nhân** | **3.45** | **3.62** | **11.16** |
| study có >1 ảnh | 55.1% | 53.8% | 46.7% |

⚠ **Test bị gom cụm theo bệnh nhân gấp ~3 lần train và val.** 3,269 study chỉ
đến từ 293 người. Bootstrap CI trong `evaluation_report.md` lấy mẫu theo
**study**, nên trên test khoảng tin cậy **hẹp hơn thực tế** — và hẹp sai nhiều
hơn hẳn so với val, nơi tỉ lệ chỉ 3.62. Cần nêu giới hạn này khi công bố, hoặc
chuyển sang bootstrap theo bệnh nhân.

Số ảnh mỗi study — train lên tới **11**, nhưng `max_aux_views: 1` nên mọi study
đều chỉ dùng anchor + 1 view phụ. Phần còn lại bị bỏ:

```
train   1 ảnh 100,098 | 2 ảnh 101,501 | 3 ảnh 18,974 | 4+ 2,185
val     1 ảnh    835  | 2 ảnh    782  | 3 ảnh    172 | 4    19
test    1 ảnh  1,742  | 2 ảnh  1,198  | 3 ảnh    296 | 4+    33
```

## View

| View | train | val | test |
|---|---:|---:|---:|
| AP | 143,556 | 1,212 | 2,405 |
| PA | 94,416 | 747 | 998 |
| LATERAL | 81,317 | 633 | 903 |
| LL | 34,330 | 275 | 528 |
| UNKNOWN | 15,320 | 124 | 325 |

Riêng train còn **21 ảnh mang view hiếm** không có ở val/test: `PA LLD` (4),
`LAO` (3), `RAO` (3), `AP AXIAL`, `AP LLD`, `AP RLD`, `XTABLE LATERAL` (mỗi loại 2),
`SWIMMERS`, `PA RLD`, `LPO` (mỗi loại 1). `anchor_priority: [PA, AP, lateral]`
không khớp chuỗi nào trong số này. 0.006% của split — không đáng lo, nhưng nếu
sau này thêm luật chọn anchor thì đây là các trường hợp rơi xuống nhánh mặc định.

## Văn bản — test **không** cùng phân bố với train

| | train | val | test |
|---|---:|---:|---:|
| FINDINGS dùng được | 79.2% | 78.3% | **91.3%** |
| IMPRESSION dùng được | 82.5% | 83.2% | **66.6%** |

Đây là đảo chiều, không phải dao động. Nguồn trích xuất cho thấy vì sao:

| Nguồn | train | val | test |
|---|---:|---:|---:|
| `FINDINGS_TAG` | 65.7% | 63.1% | 65.6% |
| `IMPRESSION_ONLY` | **19.9%** | 20.9% | **6.8%** |
| `NARRATIVE_BODY` | **14.3%** | 15.9% | **27.5%** |

Báo cáo trong test được viết theo kiểu khác: ít bản chỉ có IMPRESSION hơn 3 lần,
và nhiều bản phải khôi phục thân tường thuật hơn 2 lần. Hệ quả trực tiếp cho
Stage 2: `--section-mode findings_and_impression` được huấn luyện trên một tập
mà IMPRESSION phủ 82.5% rồi chấm trên tập chỉ phủ 66.6%.

Độ dài (token, chỉ tính study thực sự có phần đó):

| | train | val | test |
|---|---|---|---|
| FINDINGS | median 52 (p10 26, p90 90, max 420) | 52 (24, 89, 273) | 63 (31, 105, 378) |
| IMPRESSION | median 20 (p10 5, p90 66, max 269) | 22 (5, 69, 156) | 23 (5, 72, 193) |

FINDINGS trong test **dài hơn ~20%** so với train.

## Ma trận nhãn

| | train | val | test |
|---|---:|---:|---:|
| ô | 3,118,612 | 25,312 | 45,766 |
| Positive | 412,459 | 3,428 | 7,178 |
| Negative | 155,282 | 1,230 | 2,102 |
| Uncertain | 72,859 | 643 | 1,516 |
| Trống | 2,478,012 | 20,011 | 34,970 |
| **% trống** | **79.46** | **79.06** | **76.41** |
| nhãn/study | 2.88 | 2.93 | **3.30** |

Train 79.46% khớp con số 79.5% ghi trong `CLAUDE.md`. Test được gán nhãn **dày
hơn** (3.30 nhãn/study so với 2.88), tức nó không phải mẫu đại diện của train ở
mặt này.

Tỉ lệ Uncertain trong số ô được nhắc: train **11.4%**, val 12.1%, test **14.0%**.
Với `uncertain_policy: ignore_uncertain`, test mất nhiều dữ liệu chấm điểm hơn
train mất dữ liệu huấn luyện.

Chi tiết từng bệnh nằm ở `*_label_distribution.csv`. Vài điểm đáng chú ý:

**`No Finding` có 0 mẫu Negative ở cả ba split** — 74,305 / 582 / 568 Positive và
không một Negative nào. Đây là cấu trúc của labeler, không phải thiếu dữ liệu, và
đúng như `CLAUDE.md` mô tả.

**Nhãn gần như không có mẫu âm** (Negative):

| Nhãn | train | val | test |
|---|---:|---:|---:|
| No Finding | 0 | 0 | 0 |
| Pleural Other | 126 | **0** | **0** |
| Fracture | 877 | 6 | 3 |
| Atelectasis | 1,502 | 9 | 20 |
| Support Devices | 3,422 | 30 | 34 |

Vấn đề này **không chỉ có ở test**. `Pleural Other` cũng có 0 Negative trên val,
nên nó vô nghĩa ở cả bước quét ngưỡng lẫn bước chấm điểm.

**Đảo chiều mất cân bằng.** Sau khi che ô trống, Positive là đa số ở 12/14 nhãn
trên train — `Atelectasis` 44,718 Positive so với 1,502 Negative (khớp `CLAUDE.md`).
Nhãn duy nhất Negative áp đảo là `Pneumothorax` (10,171 / 41,271, prevalence
0.198). Trên test prevalence của nó tụt còn 0.133 — mức lệch phân bố train→test
lớn nhất trong 14 nhãn.

## ⚠ Val quá mỏng cho hai nhãn, và điều đó ảnh hưởng tới việc quét ngưỡng

`scripts/calibrate_thresholds.py` chạy với `--min-positive 20`. Số Positive trên val:

```
Pleural Other   15   ← dưới ngưỡng
Fracture        18   ← dưới ngưỡng
Lung Lesion     70
Enlarged Card.  60
```

**Hai nhãn không đủ mẫu để quét ngưỡng** và sẽ rơi xuống giá trị mặc định, rồi
vẫn được chấm trên test. Đây chính là lý do `selection_metric` đã đổi từ
`macro_auprc` sang `loss` (ghi trong `CLAUDE.md`) — nhưng nó cũng có nghĩa là
ngưỡng của hai nhãn này không được hiệu chỉnh, chứ không phải được hiệu chỉnh kém.
Nên nêu rõ khi báo cáo thay vì để lẫn vào macro.

## Mask giải thích

| | train | val | test |
|---|---:|---:|---:|
| study có mask | 208,987 (93.8%) | 1,690 (93.5%) | 2,959 (90.5%) |
| phổi CheXmask | 208,118 | 1,685 | 2,795 |
| box MS-CXR | 869 | **5** | 164 |
| cặp (study, bệnh) có box | 928 | **7** | 188 |

Phủ mask phổi đồng đều ~90–94% ở cả ba split, nên **term weak** (`lambda_explanation`)
có nền dữ liệu vững.

**Term strong thì không.** Val chỉ có **5 study / 7 cặp** mang box chuyên gia —
không đủ để chọn checkpoint, quét ngưỡng, hay early-stop theo chất lượng định vị.
Mọi kết luận "model nhìn đúng vào vùng tổn thương" chỉ dựa được vào **164 study
test**, và nó phải được chấm một lần duy nhất ở cuối. Train có 869 study, tức
term strong kích hoạt trên 0.39% số study mỗi epoch.

---

## Cách sinh lại

Chạy trên host có mount `/mnt/drive1tb` (không chạy được ở máy dev):

```bash
scp Test/stats/split_stats.py phuong@minhphuong:/home/phuong/
ssh phuong@minhphuong '~/.venvs/meta-cxr-stage1-311/bin/python /home/phuong/split_stats.py \
    --split train --output-dir /home/phuong/split_stats_out'
```
