# scripts/residual_pooled_calibration.py

CPU, opt-in (2026-11-07, user-approved). Chẩn đoán và sửa việc R1 (`residual_stack.py`) làm giảm chỉ số xếp hạng GỘP
toàn split (AUROC Pos/Unc, AP present), dù trong nhóm PL các chỉ số đó tăng. Cấu hình B2a: R1 (chỉ L) fit trên train PL
(`rs.r1_trainfit`), ngoài PL = OFF.

- Kênh: `channel_scores` (pos, unc, present = p_pos/(p_pos+p_neg)).
- Chẩn đoán A: `rank_swap` / `swap_inside` — cánh `scale` (thứ hạng OFF + giá trị R1 trong PL), `rank` (ngược lại).
- Sửa B: `platt_fit` / `platt_apply` (`sigmoid(a·logit(x)+b)`, a ≥ 1e-3), `fit_group_calibration` /
  `apply_group_calibration` theo (nhóm, kênh), fit trên train; `within_group_max_change` assert số trong nhóm không đổi.
- `run` → arm OFF, OFF_cal, R1, scale, rank, R1_cal; hiệu `R1_cal-OFF` quyết định (bootstrap theo study):
  TANG / HET GIAM / VAN GIAM.

CLI: `--off --view-map --expert L=… --train-off --train-expert L=… --out`. Chỉ ghi số tổng hợp, từ chối ghi đè.
Callee: [`residual_stack`](residual_stack.py.doc.md), `classification_metrics`, `view_keys`, scipy. Test:
`tests/test_residual_pooled_calibration.py`. Quy tắc: handoff `PLAN-2026-11-04-residual-lateral.md`, mục "Hiệu chỉnh theo nhóm".
