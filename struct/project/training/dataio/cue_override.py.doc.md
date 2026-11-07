# training/dataio/cue_override.py

numpy + stdlib, opt-in 2026-11-07/08 (user-approved). Cue P/N/U của study PL từ OFF + R1 lateral, tính trên logit của CHÍNH
bản ghi giai đoạn 2 (bản ghi train mã hoá từ ảnh có augmentation nên không dùng được xác suất OFF sạch).

- `load_override(path)`: npz từ [`scripts/r1_stage2_cues.py`](../../scripts/r1_stage2_cues.py.doc.md) — `keys`
  (`blake2b(dicom, 12)`), `pl_row`, `names`, `l_probs [k, n_pl, F, 3]`, `A [k, F]`, `B [k, F, 3]`, `off_cut`, `r1_cut`.
- `record_key(record)` = `sample_key_for(image_path)`.
- `r1_probs(p, l, A, B)`: TB theo seed của `softmax(log p + A log l + B)` (giống `residual_stack.apply_r1`).
- `apply_r1_cues(records, override, findings)`: mỗi bản ghi phải có khoá; bản ghi ngoài PL không đụng; bản ghi PL: kiểm tra cue
  hiện có = cutpoint OFF trên p (≥ `MIN_AGREEMENT` 0,99 ô, nếu không SystemExit), rồi thay bằng cue R1. Trả số đếm.

Caller: `training/run_medgemma_qlora.py` (`attach_prompt_inputs`, trước oracle_mix), `scripts/generate_stage2_reports.py`
(`apply_cue_override`). Test: `tests/test_cue_override.py`.
