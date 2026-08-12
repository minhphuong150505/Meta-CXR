> Source: `scripts/audit_temporal_targets.py` (78 dòng)
> Status: 🧪 ABLATION
> Last verified against source: 2026-08-12

# `scripts/audit_temporal_targets.py`

## Purpose
Đo tần suất target FINDINGS chứa ngôn ngữ so sánh thời gian **trong khi input
không có phim trước**.

## ★ Vì sao cần đo
Docstring `:3`: *"a target that says \"unchanged / compared to prior\" while the
model has no prior trains confident temporal hallucination."*

Script đếm mức độ để `temporal_target_policy` được chọn **bằng bằng chứng thay vì
đoán**. Mặc định hiện tại vẫn là `keep`.

## Entry point
```bash
python scripts/audit_temporal_targets.py
```

## Main functions
`main()` (`:31`)

## Calls / Called by
Gọi: `stage2.prompts.contains_temporal_language` (`:28`), `training.dataio.manifest`.

## Side effects
Ghi báo cáo.

## Related tests
`tests/test_stage2_prompts.py` (phần temporal policy)

## Developer notes
Kết quả script này là đầu vào cho quyết định `temporal_target_policy`. Xem
`docs/stage2_temporal_target_audit.md` (biên bản cũ — kiểm lại với dữ liệu hiện tại).

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
