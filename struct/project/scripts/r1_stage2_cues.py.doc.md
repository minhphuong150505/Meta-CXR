# scripts/r1_stage2_cues.py

CPU + dataset annotation (host), opt-in 2026-11-07 (user-approved). Tạo cue P/N/U giai đoạn 2 từ OFF + R1 cho các study PL
của một split (val), cho `generate_stage2_reports.py --cue-override`.

- R1 = cấu hình B2a (`residual_stack.r1_trainfit`, L 3 seed, fit trên train PL), xác suất cue = trung bình 3 seed.
- `pl_cues`: `off_codes` = cutpoint của OFF trên xác suất OFF (mọi study); `new_codes` = như off ngoài PL, trong PL dùng
  cutpoint fit trên PL dưới OFF + R1.
- `gen_key(dicom_id)` = `blake2b(dicom_id, 12)` = `generate_stage2_reports.sample_key_for`. dicom id đọc từ annotation của
  dataset trong bộ nhớ; npz chỉ chứa digest, mã cue và xác suất (vẫn là dẫn xuất MIMIC-CXR, giữ trên host).
- Guard: view map dựng lại phải trùng `--view-map`; khoá duy nhất. Ghi `<out>.json` (số ô/study đổi, cutpoint R1).

Callee: [`residual_stack`](residual_stack.py.doc.md), [`residual_pooled_calibration`](residual_pooled_calibration.py.doc.md)
(`load_split`), `threshold_calibration`, `view_keys`, `training.stage1.lavis_loader`. Test: `tests/test_r1_stage2_cues.py`.
Quy tắc: handoff `PLAN-2026-11-04-residual-lateral.md`, mục "R1 vào giai đoạn 2 — bước A".
