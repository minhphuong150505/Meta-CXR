> Source: `model/pretrained_medgemma/` (6 file, 526 LOC)
> Status: ✅ ACTIVE — baseline chính thức
> Last verified against source: 2026-08-12

# `model/pretrained_medgemma/`

## Purpose

Loader và reporter cho checkpoint MedGemma **của bên thứ ba**, dùng làm baseline
đối chứng (pipeline [P8](../../_meta/PIPELINES.md#p8--external-medgemma-inference-baseline)).

## ⚠ Provenance — phải nêu ở mọi nơi báo cáo kết quả này

`erjui/medgemma-4b-srrg-findings` được fine-tune từ `google/medgemma-4b-it` trên
csrrg_ift (dẫn xuất MIMIC-CXR + CheXpert+) **bởi bên thứ ba**, không phải project
này, và **không** trên split của repository này.

Docstring của package nói thẳng: project *"does not fine-tune them and must not
describe them as its own models"*. Đây là nghĩa vụ học thuật, không phải chi tiết
kỹ thuật.

Record đầu ra mang cờ provenance (`external_checkpoint`,
`fine_tuned_by_this_project`) để một file kết quả **không thể** bị nhầm là output
của model project này train.

## Role in project

```text
medgemma_inference/  ──►  model/pretrained_medgemma/  ──►  JSONL predictions
                                      │
                                 runtime/device.py
```

Hoàn toàn tách khỏi Stage 1 → Stage 2.

## Parent

[`model/`](../_index.md)

## Children

| File | LOC | Doc | Status |
|---|---|---|---|
| `findings_loader.py` | 197 | [📄](findings_loader.py.doc.md) | ✅ Nạp checkpoint, fail-closed nếu không đa phương thức |
| `findings_reporter.py` | 116 | [📄](findings_reporter.py.doc.md) | ✅ Sinh FINDINGS; **bỏ** IMPRESSION nếu model tự thêm |
| `output_schema.py` | 74 | [📄](output_schema.py.doc.md) | ✅ Record + hậu xử lý |
| `errors.py` | 34 | [📄](errors.py.doc.md) | ✅ Các failure mode |
| `impression_reporter.py` | 58 | [📄](impression_reporter.py.doc.md) | ⛔ **DISABLED có chủ đích** |
| `__init__.py` | 47 | — | ✅ Docstring provenance |

## Nguyên tắc thiết kế: fail-closed, không bao giờ hạ cấp

`errors.py` nêu rõ: mọi lỗi đều **raise** thay vì hạ xuống pipeline yếu hơn.

> *"A loader that cannot produce a multimodal model must not fall back to a
> text-only one: the resulting reports would look plausible while never having
> seen a pixel, which is the single most expensive failure this project can make."*

## `impression_reporter.py` — trơ có chủ đích

Import nó **không được** tải checkpoint, không dựng processor, không cấp VRAM,
không import transformers. Nó tồn tại để interface Phase 2 được chốt và review,
**không phải để chạy**. Phase 2 chưa được duyệt ngân sách.

## Main responsibilities

1. Nạp checkpoint ngoài, xác nhận đa phương thức, fail-closed nếu không.
2. Sinh FINDINGS; loại bỏ IMPRESSION nếu model tự thêm (kèm cảnh báo).
3. Đóng gói record có provenance, **không** mang định danh MIMIC.

## Entry points

Không có. Gọi từ `medgemma_inference/`.

## Dependencies

`transformers`, `torch` · `runtime/device.plan_device` (`findings_loader.py:24`) ·
`training/dataio/manifest.split_generated_report` (`output_schema.py:15`)

## Used by

`medgemma_inference/runner.py` · `tests/test_pretrained_findings.py` (553 dòng) ·
`tests/test_inference_only_invariants.py`

## Execution flow

```text
run_pretrained_findings.py
   ↓  guard Impression chạy TRƯỚC mọi thứ  (fail trong mili-giây)
runner.py
   ↓  lazy — chỉ dựng model khi còn việc chưa xong
findings_loader.load()  →  plan_device()  →  kiểm tra multimodal → raise nếu không
   ↓
findings_reporter.generate()  →  bỏ IMPRESSION nếu có
   ↓
output_schema  →  record có provenance, KHÔNG có subject_id/study_id
   ↓
prediction_writer  →  JSONL (flush + fsync từng dòng)
```

## Important configurations

`configs/experiments/pretrained_medgemma_findings_first.yaml` — validate bởi
`medgemma_inference/config.py`.

⚠ Validator đó **chỉ** chạy trên `configs/experiments/*.yaml`. Config Stage 1
dưới `pretraining/configs/` là namespace riêng, không bị parse ở đây.

## Status

```text
✅ ACTIVE — baseline chính thức (D-005)
⛔ impression_reporter.py — DISABLED
```

## Notes

- **Inference-only.** Không dựng optimizer, không tính gradient, không gọi
  `model.train()`. Enforce bởi `tests/test_inference_only_invariants.py`.
- **Không chạy được qua CLI fine-tuning** — `resolve_pipeline_modes` chủ động raise.
- `sample_key` là digest có salt, **không** phải `study_id`. Record an toàn để lưu.

## Related documentation

[PIPELINES.md → P8](../../_meta/PIPELINES.md#p8--external-medgemma-inference-baseline) ·
[D-005](../../_meta/DECISIONS.md#d-005--track-inference-checkpoint-ngoài-là-baseline-chính-thức) ·
[`medgemma_inference/_index.md`](../../medgemma_inference/_index.md) ·
[`runtime/_index.md`](../../runtime/_index.md)

← [`model/`](../_index.md) · [HOME](../../../HOME.md)
