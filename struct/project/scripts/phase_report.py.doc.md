> Source: `scripts/phase_report.py`
> Status: ✅ CPU, chỉ đọc
> Last verified against source: 2026-09-24

# `scripts/phase_report.py`

← [scripts](_index.md) · [D-020](../_meta/DECISIONS.md#d-020--meta-former-3-pha-theo-bài-báo-medclip-swin-mhcac-một-nhánh)

`python scripts/phase_report.py --root <ROOT>` in markdown: gate ITC mỗi epoch
(1a, 1c), phân loại val (`sp_*`, 12/13/14 nhãn) của từng epoch 1c kèm delta so
với epoch cuối 1b (= `checkpoint_phase1b`), và tóm tắt `grad_interference.jsonl`
(cosine trung bình/min/max, tỉ lệ chuẩn align/cls). Chỉ số, không có ID hay text.
