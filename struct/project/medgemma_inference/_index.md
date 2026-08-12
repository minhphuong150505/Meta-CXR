> Source: `medgemma_inference/` (6 file, 939 LOC)
> Status: ✅ ACTIVE — baseline chính thức
> Last verified against source: 2026-08-12

# `medgemma_inference/`

## Purpose

Chạy inference FINDINGS trên checkpoint MedGemma **của bên thứ ba**
([P8](../_meta/PIPELINES.md#p8--external-medgemma-inference-baseline)). Đây là
số đối chứng chính thức ([D-005](../_meta/DECISIONS.md#d-005--track-inference-checkpoint-ngoài-là-baseline-chính-thức)).

> **Package này chỉ chạy inference.** Docstring `__init__.py` nói rõ: không dựng
> optimizer, không tính gradient, không bao giờ gọi `model.train()`.
> Enforce bởi `tests/test_inference_only_invariants.py`.

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| File | LOC | Doc | Vai trò |
|---|---|---|---|
| `run_pretrained_findings.py` | 241 | [📄](run_pretrained_findings.py.doc.md) | ★ Entrypoint CLI |
| `runner.py` | 240 | [📄](runner.py.doc.md) | Điều phối; guard Impression chạy trước, model dựng lazy |
| `config.py` | 241 | [📄](config.py.doc.md) | Validate `configs/experiments/*.yaml` |
| `prediction_writer.py` | 109 | [📄](prediction_writer.py.doc.md) | JSONL append-only, flush + fsync từng dòng |
| `progress.py` | 102 | [📄](progress.py.doc.md) | Run identity — chặn trộn hai cấu hình vào một file |
| `__init__.py` | 6 | — | Docstring inference-only |

## Ba quyết định thiết kế đáng học

### 1. Thứ tự thao tác (`runner.py` docstring)
Guard Impression chạy **trước khi nạp bất cứ thứ gì** → cấu hình sai fail trong
mili-giây, thay vì sau khi tải xong checkpoint 4B. Model dựng **lazy**, chỉ khi
còn việc chưa xong → run đã resume hoàn toàn tốn **0 GPU time**.

### 2. Crash-safe resume (`prediction_writer.py`)
Mỗi record được flush + fsync ngay. Process bị kill để lại file mà **mọi dòng
hoàn chỉnh đều hợp lệ**; dòng dở dang bị cắt khi resume.

### 3. Run identity (`progress.py`)
Một run được resume phải là **cùng một run**. Nếu model, revision, generation
setting, split hay dataset fingerprint đổi, append vào file cũ sẽ **âm thầm trộn
output của hai cấu hình** — và không metric hạ nguồn nào phát hiện được. Nên nó
**từ chối** thay vì đoán.

## Main responsibilities

1. Validate config experiment.
2. Chặn Phase 2 Impression (chưa duyệt ngân sách).
3. Chạy inference có resume, có ngân sách wall-clock.
4. Ghi JSONL an toàn, có provenance.

## Entry points

```bash
python -m medgemma_inference.run_pretrained_findings \
    --config configs/experiments/pretrained_medgemma_findings_first.yaml \
    --split validation --max-samples 100
```

⚠ **Không** gọi được qua `training/run_medgemma_qlora.py` —
`resolve_pipeline_modes` chủ động raise với thông điệp chỉ đúng lệnh cần dùng.

## Dependencies

`model/pretrained_medgemma/` · `runtime/budget.py` (`runner.py:23`) ·
`training/dataio/manifest` (`:33`) · `training/stage2_utils.stable_fingerprint` (`:39`) ·
`transformers`, `torch`, `yaml`

## Used by

Người dùng trực tiếp. `tests/test_pretrained_findings.py` (553 dòng).

## Important configurations

`configs/experiments/pretrained_medgemma_findings_first.yaml` — gồm `runtime.hourly_cost_usd`,
`runtime.budget_limit_usd` (`config.py:230`).

⚠ Validator này **chỉ** áp cho `configs/experiments/*.yaml`. Config Stage 1 dưới
`pretraining/configs/` là namespace riêng — `learning_rate`, `optimizer`,
`warmup_steps` ở đó vẫn hợp lệ và không bị parse ở đây.

## Status

```text
✅ ACTIVE — baseline chính thức
```

## Notes

- **Ngân sách tính theo wall-clock**: GPU treo tốn đúng bằng GPU bận.
  `prior_elapsed_seconds` khiến resume **không** reset trần chi phí.
  `runtime/budget.py` **chỉ dừng run** — không bao giờ tự hạ cấp model.
- Record đầu ra **không mang** `subject_id`/`study_id`/path/reference text.
- Mọi báo cáo dùng số từ đây **phải nêu provenance**: checkpoint bên thứ ba,
  không train trên split của repo này.

## Related documentation

[PIPELINES.md → P8](../_meta/PIPELINES.md#p8--external-medgemma-inference-baseline) ·
[`model/pretrained_medgemma/_index.md`](../model/pretrained_medgemma/_index.md) ·
[`runtime/_index.md`](../runtime/_index.md)

← [Về HOME](../../HOME.md)
