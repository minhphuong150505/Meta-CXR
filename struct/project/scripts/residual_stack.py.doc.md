# `scripts/residual_stack.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-11-04 · 🔬 chẩn đoán (CPU, chạy trên máy train)

## Purpose
Residual **R1 ở lớp đầu ra** trên OFF đóng băng, chỉ cho study PL, và reading rule của
`docs/handoff/PLAN-2026-11-04-residual-lateral.md` (mục 2 và 7). Với mỗi finding f:

`z_ON = log p_OFF + Σ_E a[f,E] · log p_E + b[f]`, `a ∈ [0, 3]`, `b[f,neg] = 0`,

fit bằng CE ba lớp có mask + `λ(Σa² + ‖b‖²)` (λ = 1e-2), L-BFGS-B từ `a = b = 0` (= OFF).
Fit trên **val** bằng cross-fit hai nửa (PL sắp theo key, `default_rng(16).permutation(n) % 2`).
Study ngoài PL, và ô PL có tham số bằng 0, giữ **đúng từng bit** xác suất của OFF.

Arm, mỗi seed chuyên gia: `OFF+C`, `OFF+L`, `OFF+C+L`, `OFF+Lperm` (hàng của L xáo trong cùng
fold), cùng `L_alone` / `C_alone` (mô tả). Hiệu: D1 = (OFF+L) − OFF, D2 = (OFF+C+L) − (OFF+C),
D3 = (OFF+C) − OFF, Dperm.

## Reading rule (chốt 2026-11-04)
- T = 5 finding của F1-5 (`PAPER_FIVE_FINDINGS`).
- Chỉ số chính: trung bình trên T của AP one-vs-rest lớp Positive (Positive vs Negative ∪
  Uncertain, ô có nhãn, điểm `p_pos`) trên val PL; CI bootstrap ghép cặp theo study 2.000 lần,
  seed 16, của hiệu trung bình theo seed.
- NB = 2·sqrt(σ_A² + σ_B²), σ = SD giữa seed (OFF: 0); so trung bình k seed với NB/√k.
- QUA D1: cận dưới > 0 và Δ > NB/√k; rào chắn: Δ AP_pos 9 finding ngoài T ≥ −NB/√k;
  D2: cận dưới > 0. Kết luận: `QUA` / `PARTIAL` (D1 qua, D2 không) / `KHONG QUA`.
- Phụ: AUROC Pos/Unc, AP present, wF1 / F1-5 / macro recall theo cutpoint cross-fit trên PL,
  toàn val (ngoài PL dùng cutpoint của OFF), từng finding + Holm trên T, `a[f]`.

## Đầu vào / đầu ra
`--off` (npz val của OFF), `--view-map` (`view_map.npz` từ lượt `--views lateral` trên val),
`--off-cutpoints`, `--expert L=…,…,…` `[--expert C=…]`, `--expect-counts` (mặc định
`PL=797,F1=930,other=81`, lệch thì dừng). Ghi một JSON tổng hợp; từ chối ghi đè.
Mọi key của OFF phải khớp map 100%, nhãn chuyên gia phải trùng nhãn OFF trên PL.

## Bẫy
- `fit_severity_cutpoints` coi **-1** là thiếu nhãn: script chuẩn hoá mọi nhãn âm về -1 trước
  khi fit cutpoint (đặc trưng probe ghi -100).
- Lưới cutpoint 121 điểm là O(lưới²) mỗi finding: vài phút cho ~13 arm-seed; test hạ
  `CUTPOINT_GRID`.

Callee: `classification_metrics`, `threshold_calibration`, [`view_keys`](../training/evaluation/view_keys.py.doc.md),
scipy (`L-BFGS-B`). Test: `tests/test_residual_lateral.py`.

## B2a (2026-11-05, opt-in): fit R1 trên train
`--train-off <OFF train npz>` + `--train-expert NAME=<oof_f0>+<oof_f1>,...` (cùng tên, cùng thứ tự seed với `--expert`):
`r1_trainfit` fit tham số trên toàn bộ study train PL (logit OFF trên train, dự đoán out-of-fold của chuyên gia) rồi
áp lên mọi study val PL; `run(..., train=...)` thay `r1_crossfit` bằng nó, mọi phần sau giữ nguyên. Ở `--expert`,
`a+b` = trung bình các mô hình fold trên val; ở `--train-expert`, `a+b` = nối các phần out-of-fold
(`load_expert_parts(spec, "mean" | "concat")`, từ chối phần trùng/khác study). JSON thêm `fit_on`, `n_train_pl`.
