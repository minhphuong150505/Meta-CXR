# scripts/r1_stage2_cues.py

CPU + annotation của dataset (host), opt-in 2026-11-07, đổi định dạng 2026-11-08 (bước B). Ghi `<prefix>_val.npz` và
`<prefix>_train.npz`: CÔNG THỨC R1 cho cue giai đoạn 2 (định dạng: [`cue_override`](../training/dataio/cue_override.py.doc.md)).

- A, B mỗi seed: `residual_stack.r1_trainfit` trên train PL (chuyên gia OOF), dùng chung cho hai split.
- `l_probs`: val = TB 2 mô hình fold mỗi seed; train = OOF mỗi seed.
- `fit_r1_cutpoints`: cutpoint (weighted F1) trên val PL dưới OFF + R1 (TB seed); `off_cut` = file cutpoint của OFF.
- `split_keys`: khoá `gen_key(dicom)` theo thứ tự dataset; view map dựng lại phải trùng thứ tự khoá của file dự đoán.
- `<prefix>.json`: số ô cue val đổi trên xác suất OFF (bước A: 268), cutpoint R1, A trung bình. Từ chối ghi đè.

Callee: [`residual_stack`](residual_stack.py.doc.md), [`residual_pooled_calibration`](residual_pooled_calibration.py.doc.md)
(`load_split`), `threshold_calibration`, `view_keys`, `training.stage1.lavis_loader`. Test: `tests/test_r1_stage2_cues.py`.
