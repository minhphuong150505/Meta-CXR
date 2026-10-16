# Toàn bộ độ đo theo giao thức bài báo META-CXR

Ngày tính: 2026-10-05, trên máy train, code `8809995` (không sửa đổi cách tính). Khoảng tin cậy 95% = bootstrap percentile 1.000 lần theo nghiên cứu (seed 42).

## Giai đoạn 1 — phân loại 14 bất thường, ba lớp (Âm / Dương / Không chắc chắn)

Checkpoint: `run_20260930_3class`, phase 1c, `checkpoint_best`. Tập test, n = 3,269 nghiên cứu.

Cách tính như bài báo: argmax trên softmax ba lớp cho từng bệnh; precision / recall / F1 kiểu sklearn `average='weighted'`, `zero_division=1` cho từng bệnh rồi lấy trung bình 14 bệnh (Fig. 10); F1 trung bình 5 bệnh (Bảng 5, 7); AUROC một-lớp-so-với-phần-còn-lại cho từng lớp (Fig. 5). Ô trống được coi là âm tính, như mục III-A của bài báo.

### Bảng 1. Chỉ số tổng hợp

| Chỉ số | Bài báo | Mô hình của đề tài | KTC 95% |
|---|---:|---:|---|
| Precision (weighted, TB 14 bệnh) — Fig. 10 | 0.87 | 0.8379 | [0.8344, 0.8452] |
| Recall (weighted, TB 14 bệnh) — Fig. 10 | 0.78 | 0.7685 | [0.7641, 0.7728] |
| F1 (weighted, TB 14 bệnh) — Fig. 10 | 0.73 | 0.7658 | [0.7612, 0.7705] |
| F1 trung bình 5 bệnh — Bảng 5, 7 | 0.701 | 0.6438 | [0.6352, 0.6525] |
| AUROC lớp Dương (TB 14 bệnh) — Fig. 5 | không báo cáo số tổng hợp | 0.7755 | [0.7671, 0.7839] |
| AUROC lớp Âm (TB 14 bệnh) — Fig. 5 | không báo cáo số tổng hợp | 0.7691 | [0.7611, 0.7773] |
| AUROC lớp Không chắc (TB bệnh có lớp này) — Fig. 5 | không báo cáo số tổng hợp | 0.5737 | [0.5382, 0.6067] |

Ghi chú: weighted recall bằng đúng accuracy (cùng một đại lượng). Code tham chiếu của bài báo lấy trung bình theo từng batch rồi trung bình các batch; ở đây mỗi bệnh được tính trên toàn bộ tập test — đúng đại lượng mà trung bình theo batch xấp xỉ, nên có thể lệch nhỏ.

### Bảng 2. Từng bệnh — precision / recall / F1 weighted (Fig. 10) và AUROC theo lớp (Fig. 5)

| Bệnh | P (w) | R (w) | F1 (w) | AUROC Âm | AUROC Dương | AUROC Không chắc | n Âm | n Dương | n Không chắc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No Finding | 0.8333 | 0.8522 | 0.8300 | 0.8215 | 0.8221 | – | 2680 | 568 | 0 |
| Enlarged Cardiomediastinum | 0.8113 | 0.8738 | 0.8375 | 0.6359 | 0.6331 | 0.5934 | 2893 | 151 | 204 |
| Cardiomegaly * | 0.7712 | 0.4055 | 0.3629 | 0.7314 | 0.7428 | 0.5033 | 2267 | 866 | 115 |
| Lung Opacity | 0.7008 | 0.6127 | 0.6164 | 0.7183 | 0.7218 | 0.5066 | 2131 | 1038 | 79 |
| Lung Lesion | 0.9293 | 0.9135 | 0.9184 | 0.7224 | 0.7272 | 0.5788 | 3107 | 123 | 18 |
| Edema * | 0.7446 | 0.6539 | 0.6767 | 0.8380 | 0.8429 | 0.6716 | 2276 | 683 | 289 |
| Consolidation * | 0.8858 | 0.7746 | 0.8054 | 0.7180 | 0.7562 | 0.4550 | 2931 | 220 | 97 |
| Pneumonia | 0.8084 | 0.6918 | 0.6884 | 0.7311 | 0.7140 | 0.6812 | 2563 | 335 | 350 |
| Atelectasis * | 0.7171 | 0.6170 | 0.6272 | 0.7703 | 0.7652 | 0.6017 | 2326 | 731 | 191 |
| Pneumothorax | 0.9510 | 0.8996 | 0.9185 | 0.8229 | 0.8423 | 0.4762 | 3117 | 112 | 19 |
| Pleural Effusion * | 0.8083 | 0.7555 | 0.7468 | 0.8905 | 0.8919 | 0.5520 | 2044 | 1074 | 130 |
| Pleural Other | 0.9594 | 0.9667 | 0.9605 | 0.8067 | 0.8374 | 0.5859 | 3169 | 63 | 16 |
| Fracture | 0.9725 | 0.9717 | 0.9580 | 0.6643 | 0.6632 | 0.6350 | 3155 | 89 | 4 |
| Support Devices | 0.8374 | 0.7700 | 0.7746 | 0.8964 | 0.8976 | 0.6175 | 2119 | 1125 | 4 |

\* = một trong 5 bệnh của Bảng 5/7. "–" = lớp không có mẫu nào trong tập test nên AUROC không xác định (ví dụ No Finding không có nhãn Không chắc).

### Bảng 3. Precision / recall / F1 theo từng lớp (bổ sung để đọc Bảng 2, bài báo không báo cáo)

| Bệnh | P Âm | R Âm | F1 Âm | P Dương | R Dương | F1 Dương | P Không chắc | R Không chắc | F1 Không chắc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No Finding | 0.8696 | 0.9657 | 0.9151 | 0.6618 | 0.3169 | 0.4286 | 1.0000 | 1.0000 | 1.0000 |
| Enlarged Cardiomediastinum | 0.8940 | 0.9768 | 0.9336 | 0.1538 | 0.0397 | 0.0632 | 0.1250 | 0.0294 | 0.0476 |
| Cardiomegaly | 0.9366 | 0.2086 | 0.3413 | 0.3077 | 0.9746 | 0.4677 | 1.0000 | 0.0000 | 0.0000 |
| Lung Opacity | 0.8141 | 0.5631 | 0.6657 | 0.4453 | 0.7611 | 0.5619 | 1.0000 | 0.0000 | 0.0000 |
| Lung Lesion | 0.9612 | 0.9482 | 0.9546 | 0.1148 | 0.1707 | 0.1373 | 1.0000 | 0.0000 | 0.0000 |
| Edema | 0.9066 | 0.6696 | 0.7703 | 0.4448 | 0.7848 | 0.5678 | 0.1768 | 0.2215 | 0.1966 |
| Consolidation | 0.9356 | 0.8182 | 0.8730 | 0.1723 | 0.5364 | 0.2608 | 1.0000 | 0.0000 | 0.0000 |
| Pneumonia | 0.8602 | 0.8069 | 0.8327 | 0.2121 | 0.5343 | 0.3036 | 1.0000 | 0.0000 | 0.0000 |
| Atelectasis | 0.8879 | 0.6028 | 0.7181 | 0.3609 | 0.8235 | 0.5019 | 0.0000 | 0.0000 | 0.0000 |
| Pneumothorax | 0.9781 | 0.9179 | 0.9470 | 0.1889 | 0.5446 | 0.2805 | 1.0000 | 0.0000 | 0.0000 |
| Pleural Effusion | 0.9056 | 0.7324 | 0.8098 | 0.6000 | 0.8911 | 0.7171 | 1.0000 | 0.0000 | 0.0000 |
| Pleural Other | 0.9767 | 0.9899 | 0.9832 | 0.0833 | 0.0476 | 0.0606 | 1.0000 | 0.0000 | 0.0000 |
| Fracture | 0.9717 | 1.0000 | 0.9856 | 1.0000 | 0.0112 | 0.0222 | 1.0000 | 0.0000 | 0.0000 |
| Support Devices | 0.9586 | 0.6777 | 0.7940 | 0.6086 | 0.9467 | 0.7409 | 1.0000 | 0.0000 | 0.0000 |

Lưu ý: với `zero_division=1`, một lớp mà mô hình không bao giờ dự đoán sẽ có precision = 1.0 (quy ước của code bài báo, không phải mô hình dự đoán đúng).

## Giai đoạn 2 — sinh phần FINDINGS (Bảng 3 của bài báo)

Báo cáo đã sinh ngày 2026-09-13 trên **cùng 2.772 nghiên cứu của tập test** cho cả ba cấu hình (khớp theo `sample_key`), greedy, tối đa 160 token, stop token đã sửa `[1, 106]`, không chặn lặp n-gram. Chấm lại hôm nay bằng `scripts/evaluate_stage2.py`.

BERTScore tính theo cài đặt dùng để so với bài báo (`--paper-bertscore`: `distilroberta-base`, `rescale_with_baseline=True`). Bài báo không ghi cài đặt; đây là cài đặt CXR-Report-Metric của công trình RadCliQ mà bài báo trích dẫn.

### Bảng 4. Độ đo ngôn ngữ

| Chỉ số | Bài báo | Arm A | Arm C | Zero-shot |
|---|---:|---:|---:|---:|
| BLEU-1 | 0.390 | 0.2057 | 0.1348 | 0.3112 |
| BLEU-2 | 0.255 | 0.1246 | 0.0818 | 0.1771 |
| BLEU-3 | 0.175 | 0.0832 | 0.0556 | 0.1105 |
| BLEU-4 | 0.102 | 0.0583 | 0.0395 | 0.0715 |
| METEOR | 0.173 | 0.2143 | 0.1937 | 0.2613 |
| ROUGE-L | 0.280 | 0.2356 | 0.2234 | 0.2370 |
| CIDEr | 0.291 | 0.1194 | 0.1602 | 0.0599 |
| BERTScore-F1 (rescaled) | 0.426 | 0.2289 | 0.1280 | 0.2792 |

### Bảng 5. Khoảng tin cậy 95%

| Chỉ số | Arm A | Arm C | Zero-shot |
|---|---|---|---|
| bleu_1 | – | – | – |
| bleu_2 | – | – | – |
| bleu_3 | – | – | – |
| bleu_4 | – | – | – |
| meteor | [0.2079, 0.2203] | [0.1865, 0.2011] | [0.2578, 0.2645] |
| rouge_l | [0.2299, 0.2409] | [0.2170, 0.2306] | [0.2342, 0.2396] |
| cider | [0.0958, 0.1442] | [0.1287, 0.1972] | [0.0552, 0.0653] |
| bertscore_f1 | [0.2163, 0.2403] | [0.1140, 0.1426] | [0.2719, 0.2865] |

BLEU được tính ở mức corpus nên script không bootstrap nó ("–").

Cấu hình:

- **armA**: Arm A — MedGemma tinh chỉnh, ảnh gốc, không cue (`medgemma_direct`)
- **armC**: Arm C — MedGemma tinh chỉnh + 32 soft token Q-Former + cue P/N/U (`meta_cxr_native_qformer_guided`)
- **zeroshot**: Zero-shot — MedGemma gốc, không tinh chỉnh

## Các chỉ số của bài báo KHÔNG tính được (ghi là không có, không phải 0)

| Chỉ số | Bài báo | Lý do |
|---|---:|---|
| Clinical Efficacy P / R / F1 — Bảng 3 | 0.411 / 0.465 / 0.428 | Repo không có CheXpert labeler / CheXbert để gán nhãn báo cáo sinh ra; chỉ có phần chấm điểm (`evaluate_clinical_efficacy.py`). |
| RadGraph-F1, RadCliQ — Bảng 3 | 0.224 / 1.211 | Repo không cài RadGraph / RadCliQ (giấy phép riêng). |
| CheXpert cross-domain F1 — Bảng 4 | 0.699 | Repo không có bộ nạp tập CheXpert-val cho Giai đoạn 1; chỉ có phần chấm (`evaluate_chexpert_crossdomain.py`). |

## Lưu ý khi so với bài báo

- Hai arm Giai đoạn 2 được huấn luyện trên checkpoint Giai đoạn 1 cũ (`run_20260820_ft`), không phải `run_20260930_3class` ở trên. Arm A chỉ chạy 0,86 epoch và không có chọn checkpoint theo validation.
- Bài báo dùng Vicuna-7B; đề tài dùng MedGemma-1.5-4B. Bài báo không công bố tập test cụ thể; Giai đoạn 1 ở đây dùng 3.269 nghiên cứu test chính thức của MIMIC-CXR, Giai đoạn 2 dùng 2.772 nghiên cứu (nghiên cứu có FINDINGS hợp lệ, ảnh chính diện).
- Các độ đo ngôn ngữ chỉ đo trùng lặp bề mặt, không phải độ chính xác lâm sàng.

## Tệp gốc

Trên máy train: `~/paper_metrics_20261005/` (script `~/paper_metrics.sh`, log `status.log`). Bản sao số liệu tổng hợp: `raw/` cạnh file này (không có văn bản báo cáo hay mã bệnh nhân).
