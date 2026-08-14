> Source: `training/evaluation/` (17 file Python)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-14

# `training/evaluation/`

## Purpose

Toàn bộ evaluator cho **cả hai Stage**. Nguyên tắc thiết kế trung tâm:

> **Classification/generation đọc file kết quả đã lưu — không cần model, không
> cần GPU, không cần dataset.** XAI là ngoại lệ có chủ đích: module metric vẫn
> thuần NumPy, nhưng `scripts/evaluate_explanation.py` phải tạo CAM từ đồ thị
> autograd sống.

Đổi threshold hay đổi uncertain policy không được tốn một GPU-hour nào.

⚠ **Không có thư mục top-level `evaluation/`.** Tài liệu cũ nào nhắc tới
`evaluation/eval_final_200.py` đều đã lỗi thời.

## Role in project

```text
Stage 1 → .npz ──► calibrate_thresholds.py → evaluate_stage1.py ──► metrics
Stage 2 → .jsonl ─────────────────────────► evaluate_stage2.py ──► metrics
Stage 1 checkpoint + split ───────────────► evaluate_explanation.py ──► XAI
```

## Parent

[`training/`](../_index.md)

## Children

### Lõi — chỉ cần numpy

| Module | LOC | Doc | Vai trò |
|---|---|---|---|
| `schemas.py` | 334 | [📄](schemas.py.doc.md) | `ClassificationPredictions`, `load_generation_records`, `CLASS_NAMES` |
| `classification_metrics.py` | 662 | [📄](classification_metrics.py.doc.md) | P/R/F1 macro, per-pathology, AUROC, AUPRC |
| `uncertain_policy.py` | — | [📄](uncertain_policy.py.doc.md) | `POLICIES`, `binarize_labels`, `DEFAULT_POLICY` |
| `threshold_calibration.py` | 345 | [📄](threshold_calibration.py.doc.md) | Calibrate — **chỉ validation** |
| `baselines.py` | 204 | [📄](baselines.py.doc.md) | All-negative và các baseline |
| `bootstrap.py` | 221 | [📄](bootstrap.py.doc.md) | Khoảng tin cậy |
| `generation_metrics.py` | 437 | [📄](generation_metrics.py.doc.md) | BLEU, ROUGE-L (tự implement), `normalize`, `tokenize` |
| `error_analysis.py` | 356 | [📄](error_analysis.py.doc.md) | Per-sample; → `safety/claims.py` |
| `subgroup_analysis.py` | 215 | [📄](subgroup_analysis.py.doc.md) | Phân tích theo nhóm |
| `report_writer.py` | 423 | [📄](report_writer.py.doc.md) | Xuất markdown + json |
| `clinical.py` | 185 | [📄](clinical.py.doc.md) | ⚠ Luôn báo unavailable — xem dưới |
| `explanation_metrics.py` | — | [📄](explanation_metrics.py.doc.md) | Eq. (7)–(9), thuần NumPy; tách lung/bbox bắt buộc |

### Có điều kiện / chưa nối

| Module | Status | Lý do |
|---|---|---|
| `visualization.py` (336) | 🟡 | Import trễ ở `evaluate_stage1.py:294`; cần extra `eval-plots` |
| `config.py` (238) | ❓ | Chỉ test import — [D-001](../../_meta/DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline) |
| `counterfactual.py` (268) | ❓ | Chỉ test |
| `perturbations.py` | ❓ | Chỉ `counterfactual.py` + test |

## ⚠ Chỉ số lâm sàng — quy tắc không thương lượng

CheXbert, RadGraph, RadCliQ, RadFact **cố ý không cài được** như extras: chúng là
research code sau license riêng, không phải pin tái lập được.

`clinical.py` raise:
- `MissingOptionalDependency` — **nêu tên package** thiếu
- `NotImplementedError` — package có nhưng adapter chưa validate với điểm tham
  chiếu công bố

> **Chỉ số lâm sàng thiếu được báo là "unavailable", KHÔNG BAO GIỜ là điểm 0.**
> Và BLEU/ROUGE **không được** trình bày như độ chính xác lâm sàng — chúng đo
> trùng lặp từ ngữ, không đo đúng/sai y khoa.

## Main responsibilities

1. Đọc `.npz` / `.jsonl` thành schema có kiểu.
2. Tính chỉ số phân loại và sinh ngôn ngữ.
3. Calibrate threshold **chỉ trên validation**.
4. Bootstrap khoảng tin cậy.
5. Xuất báo cáo markdown + json.
6. **Từ chối** bịa chỉ số lâm sàng.
7. Tính XAI metric theo từng `mask_source`, không tạo aggregate lung+bbox.

## Entry points

Không có. Được `scripts/evaluate_stage*.py` và `calibrate_thresholds.py` gọi.

## Dependencies

| Nhóm | Cần gì |
|---|---|
| Lõi | chỉ `numpy` |
| `visualization.py` | extra `eval-plots` (matplotlib) |
| METEOR / CIDEr / BERTScore | extra `eval-generation` |
| `clinical.py` | **cố ý không có** |
| `error_analysis.py` | `safety/claims.py` |

## Used by

`scripts/evaluate_stage1.py`, `scripts/evaluate_stage2.py`,
`scripts/evaluate_explanation.py`, `scripts/calibrate_thresholds.py`, và ⚠
`model/lavis/tasks/image_text_pretrain.py:223,259`
(**import trễ trong hàm** — tạo phụ thuộc ngược `model/lavis/` → `training/`).

## Execution flow

Xem [CALL_GRAPH.md §4](../../_meta/CALL_GRAPH.md#4-evaluation--top-down).

## Important configurations

| Tham số | Nguồn | Ghi chú |
|---|---|---|
| `--objective f1` | CLI | Mục tiêu calibrate |
| `--uncertain-policy` | CLI | prod: `ignore_uncertain` |
| `--min-positive 20` | CLI | Bệnh lý <20 positive giữ threshold 0.5 |
| `--metrics bleu,rouge,…` | CLI | Chọn chỉ số Stage 2 |
| `--skip-clinical-metrics` | CLI | Bỏ qua adapter lâm sàng |
| `--bootstrap-samples` | CLI | Số lần lấy mẫu lại |

## Status

```text
✅ ACTIVE — 12 module lõi
🟡 CONDITIONAL — visualization.py
❓ UNKNOWN — config.py, counterfactual.py, perturbations.py
```

## Notes

- **Threshold chỉ calibrate trên validation**, rồi mới áp lên test. Làm ngược lại
  là rò rỉ test set.

- XAI report không có trường `overall`: `mask_source=0` chỉ chứng minh saliency
  ở trong phổi; chỉ `mask_source=1` với bbox MS-CXR mới là bằng chứng định vị ổ
  bệnh. Annotation coverage của nhóm lung là `null/unavailable`, không phải 0.

- `--min-positive 20`: calibrate trên quá ít mẫu positive chỉ là overfit vào nhiễu.

- `error_analysis.py` là **nơi duy nhất ngoài test** dùng `safety/` — cụ thể là
  `safety/claims.py`. Phần còn lại của `safety/` chưa được nối.

- ⚠ `docs/evaluator_audit.md` và `docs/evaluator_validation.md` là **biên bản tại
  một thời điểm**. Kiểm lại với code trước khi trích dẫn.

## Related documentation

[PIPELINES.md → P4, P5](../../_meta/PIPELINES.md#p4--evaluation-stage-1) ·
[PROJECT_OVERVIEW.md §5](../../_meta/PROJECT_OVERVIEW.md#5-đánh-giá) ·
[`scripts/_index.md`](../../scripts/_index.md) · [`safety/_index.md`](../../safety/_index.md)

← [`training/`](../_index.md) · [HOME](../../../HOME.md)
