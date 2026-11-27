# PLAN 2026-11-27 — số chính cho pipeline OFF + residual lateral R1 (D-029)

User duyệt 2026-11-27 ("làm cả 2 đi"): (1) bảng Stage 1 đầy đủ của OFF + R1, (2) Stage 2 đầy đủ trên Stage 1 của
pipeline chính rồi chấm test một lần. Quy tắc đọc dưới đây chốt TRƯỚC khi có số.

## 1. Bảng Stage 1 (CPU, không đọc test thêm lần nào)

Chấm lại lần đọc test DUY NHẤT của Bước 3 (2026-11-05, `~/reslat_step3_20261105/`): cùng ứng viên (OFF + R1, chuyên gia
L_lora 3 seed, R1 fit trên toàn bộ val PL, cutpoint PL fit trên val), cùng lệnh, thêm `residual_stack.py --save-arms`,
rồi `scripts/r1_stage1_table.py`. Không chọn gì.

- **G1:** JSON chạy lại phải trùng `step3_test.json` từng byte (cờ mới chỉ thêm file).
- **Báo cáo:** weighted P/R/F1, F1-5, macro recall, AUROC N/P/U và trung bình, OFF và OFF + R1, hiệu có CI (bootstrap
  cặp 2.000, seed 16), toàn test và nhóm PL; headline = cutpoint (không phải giao thức bài báo), argmax ghi bên cạnh.
- OFF ở đây dùng cutpoint PL của chính nó (cùng quy trình với R1); OFF theo file cutpoint gốc ghi riêng (`off_reported`).

## 2. Stage 2 đầy đủ trên OFF + R1 (GPU ~2 ngày)

Công thức = P2 đã báo cáo (`meta_cxr_qformer_with_mhcac_prompt`, `--prompt-style paper_context`, `--train-cues
oracle_mix` 0,2, `--lora-lr 3e-4`, 2 epoch, batch 2 × 8, seed 16, chọn epoch theo val loss), chỉ đổi Stage 1:

- Stage 1 = `run_20261024_nomv` phase1c best với `--stage1-options model.multi_view=false model.loss.lambda_mpc=0.0
  model.loss.lambda_view_consistency=0.0`, cue `--cue-rule cutpoints --threshold-path configs/stage1_cutpoints/run_20261024_nomv.json`.
  File này = `~/nomv_probe_20261024/cutpoints_nomv.json` (thư mục đã mất trên host): fit lại 2026-11-27 bằng
  `calibrate_thresholds.py --rule cutpoints` trên npz val của OFF, trùng `off_cut` trong `r1_val.npz` của bước B từng bit.
- Cue của study PL = R1: `--cue-override-train/val` = công thức của bước B (`~/r1s2b_20261108/r1_{train,val}.npz`),
  test = `r1_stage2_cues.py --test-only --recipe r1_val.npz` (A, B, cutpoint chép từng bit; chuyên gia L test của Bước 3).
- Manifest v3 (`env_config` của `~/Meta-CXR-v3`) như P2; checkout mới của nhánh ở commit ghi trong log.
- Sinh test: beam 4 / lp 2 (số chính) và greedy, `--limit 0`, `--max-new-tokens 256`.

**Kiểm (dừng nếu sai):** GPU rảnh và không job nào chạy; Stage-1 load `unexpected=0`; log có "train R1 override" và
"val R1 override"; test có "[cues] override"; adapter có `adapter_model.safetensors` + `img_proj.pt`;
`n_failed == 0`.

**Quy tắc đọc (chốt trước):** đây là số chính của pipeline, không phải lựa chọn giữa các ứng viên, nên không có luật
"qua/không qua". Báo cáo: giao thức bài báo (BLEU-1/4, METEOR, ROUGE-L, CIDEr, BERTScore distilroberta rescale) và
CheXbert micro/macro F1-14, F1-5, trên toàn test. So với P2 đã báo cáo (Stage 1 = `run_20261005_paper`, fusion ON) chỉ
MÔ TẢ, ghép cặp theo `sample_key` (bootstrap 2.000): hai lượt khác Stage 1, cohort train oracle_mix và một seed mỗi
bên, nên một hiệu nằm trong NB Stage 2 (BERTScore 0,0054, CheXbert micro F1-14 0,0192) không được đọc là khác biệt.
Luôn ghi "không phải điều kiện của bài báo" (mô hình đọc indication + FINDINGS lần trước). Test đọc một lần.

## Execution report — phần 1, bảng Stage 1 (CPU 2026-11-27, `~/r1table_20261127/`)

Checkout `~/Meta-CXR-d029` + 3 file của `9367c52`. **G1:** `residual_stack` chạy lại trùng `step3_test.json` ở mọi con số
(chỉ thêm hai khoá `gate: none`, `rule_gate: null` của R1H, có sau 05/11): D1 +0,0222 [+0,0134, +0,0304], `whole_test`
wF1 OFF+L 0,8203 / OFF 0,8189. File cutpoint của OFF đã mất trên host; bản fit lại (`configs/stage1_cutpoints/
run_20261024_nomv.json`) trùng `off_cut` của bước B từng bit. Test n = 3.269, trong đó PL 1.196; R1 = TB 3 seed;
bootstrap cặp 2.000.

**Toàn test, headline (cutpoint fit trên val — không phải giao thức bài báo):**

| chỉ số | OFF | OFF + R1 | hiệu [CI 95%] |
|---|---|---|---|
| weighted precision | 0,8186 | 0,8157 | −0,0029 [−0,0041, −0,0005] |
| weighted recall | 0,8338 | 0,8367 | +0,0029 [+0,0020, +0,0038] |
| weighted F1 | 0,8189 | **0,8203** [0,8161, 0,8248] | +0,0015 [+0,0006, +0,0023] |
| F1-5 | 0,7496 | **0,7517** [0,7436, 0,7592] | +0,0021 [+0,0005, +0,0037] |
| macro recall | 0,4433 | 0,4423 | −0,0011 [−0,0031, +0,0009] |
| AUROC N / P / U (gộp, thô) | 0,7772 / 0,7861 / 0,6446 | 0,7687 / 0,7767 / 0,6656 | −0,0085* / −0,0093* / +0,0210 (CI qua 0) |
| AUROC TB | 0,7382 | 0,7387 | +0,0006 [−0,0096, +0,0094] |

**Toàn test, argmax (giao thức bài báo):** OFF wP/wR/wF1 0,8463 / 0,7805 / 0,7738, F1-5 0,6558; OFF + R1 0,8375 / 0,8007 /
0,7928, F1-5 0,6923 (wF1 +0,0190 [+0,0171, +0,0209], F1-5 +0,0365 [+0,0328, +0,0405], macro recall −0,0006 CI qua 0).

**Nhóm PL (n = 1.196), headline:** wF1 0,8476 → 0,8522 (+0,0046 [+0,0024, +0,0069]), F1-5 0,8080 → 0,8139 (+0,0059
[+0,0017, +0,0102]), AUROC P 0,8009 → 0,8079 (+0,0070*), U +0,0473*, TB +0,0211*; argmax wF1 +0,0387*, F1-5 +0,0720*,
macro recall −0,0166*.

**Đọc:** trên toàn test OFF + R1 hơn OFF một chút ở wF1 / F1-5 (CI loại 0, nhưng hiệu nhỏ: +0,0015 / +0,0021), đổi
precision lấy recall. AUROC gộp thô giảm vì lệch thang PL / non-PL (đã biết; hiệu chỉnh theo nhóm cho +0,0060
[+0,0040, +0,0079], xem PLAN-2026-11-04 "Hiệu chỉnh theo nhóm trên TEST"). Dưới argmax hiệu lớn hơn nhiều, chủ yếu ở PL —
là dịch điểm vận hành (R1 làm xác suất P/U lớn hơn), không phải bằng chứng xếp hạng tốt hơn. Kết luận của Bước 3 giữ
nguyên: PARTIAL, "lợi ích chủ yếu do ghép thêm một mô hình". `off_reported` (file cutpoint trên mọi study, như số đã báo
cáo của OFF): wF1 0,8174, F1-5 0,7484, macro recall 0,4403.
