# Báo cáo Stage 1 — chấm theo argmax (giao thức bài báo), 2026-11-27

## Quyết định của người dùng (2026-11-27, D-030)

- **Chấm điểm mô hình Stage 1 dùng argmax** trên ba lớp Negative / Positive / Uncertain, đúng giao thức của bài báo
  META-CXR. Đây là số chính để so sánh các mô hình Stage 1 với nhau và với bài báo.
- **Ngưỡng (cutpoint) vẫn được fit trên val và đưa vào Stage 2** làm cue P/N/U (`--cue-rule cutpoints`). Số Stage 1 theo
  cutpoint chỉ còn là số bổ sung (điểm vận hành của cue), không còn là số chính.
- Lượt Stage 2 đầy đủ của pipeline chính (bắt đầu 2026-11-27 17:38, `~/s2main_20261127/`) giữ nguyên cue cutpoint,
  không phải chạy lại.

Thay cho quyết định 2026-10-10 ("headline = hai cutpoint trên val, không phải argmax").

## Bảng so sánh, test MIMIC-CXR n = 3.269, ba lớp, argmax

| Chỉ số (TB 14 finding) | Bài báo META-CXR | ON (fusion, `run_20261005_paper`) | OFF (`run_20261024_nomv`) | **OFF + R1** (pipeline chính) |
|---|---|---|---|---|
| weighted precision | 0,87 | 0,8392 [0,8351; 0,8435] | 0,8463 [0,8428; 0,8500] | 0,8375 [0,8337; 0,8425] |
| weighted recall | 0,78 | 0,7814 [0,7766; 0,7857] | 0,7805 [0,7763; 0,7846] | 0,8007 [0,7966; 0,8048] |
| weighted F1 | 0,73 | 0,7795 [0,7748; 0,7842] | 0,7738 [0,7695; 0,7783] | **0,7928** [0,7886; 0,7973] |
| F1-5 (Atel., Cardio., Consol., Edema, PE) | 0,701 | 0,6687 [0,6596; 0,6775] | 0,6558 [0,6470; 0,6645] | **0,6923** [0,6838; 0,7010] |
| macro recall 3 lớp (không có trong bài báo) | — | 0,4609 | 0,4548 | 0,4543 |
| AUROC Neg / Pos / Unc (TB) | — | 0,7780 / 0,7836 / 0,6633 | 0,7772 / 0,7861 / 0,6446 | 0,7687 / 0,7767 / 0,6656 |

Khoảng tin cậy 95%: bootstrap theo study (ON 1.000 mẫu; OFF và OFF + R1 2.000 mẫu, seed 16). AUROC không phụ thuộc luật
quyết định.

**OFF + R1 − OFF (bootstrap cặp):** weighted P −0,0088 [−0,0112; −0,0053], weighted R +0,0202 [+0,0182; +0,0221],
weighted F1 **+0,0190 [+0,0171; +0,0209]**, F1-5 **+0,0365 [+0,0328; +0,0405]**, macro recall −0,0006 (CI chứa 0).
Riêng nhóm có ảnh frontal + lateral (n = 1.196): weighted F1 +0,0387, F1-5 +0,0720, macro recall −0,0166 (đều có ý nghĩa).

## Đọc kết quả

- OFF + R1 có weighted F1 và F1-5 cao nhất trong ba mô hình, và weighted F1 vượt bài báo (0,793 so với 0,73); F1-5 vẫn
  dưới bài báo (0,692 so với 0,701).
- Mức tăng so với OFF là **đổi precision lấy recall**: R1 đẩy xác suất Positive / Uncertain lên ở study có lateral, nên
  argmax gọi nhiều ô dương hơn. AUROC gộp không tăng (Pos −0,0093 thô; sau hiệu chỉnh theo nhóm +0,0060), nên đây là
  dịch điểm vận hành nhiều hơn là xếp hạng tốt hơn. Kết luận của Bước 3 giữ nguyên: lợi ích chủ yếu do ghép thêm một
  mô hình, không nói là lợi ích riêng của ảnh lateral.
- ON và OFF + R1 mới chỉ đặt cạnh nhau, CHƯA bootstrap cặp; không kết luận bên nào hơn.
- Một mô hình, một seed mỗi dòng; bài báo không nói ngưỡng/split nào nên so sánh với bài báo chỉ là tham khảo.

## Số bổ sung theo cutpoint (điểm vận hành của cue Stage 2)

Toàn test: OFF weighted F1 0,8189, F1-5 0,7496; OFF + R1 0,8203 / 0,7517 (hiệu +0,0015 / +0,0021, CI loại 0). ON (file
`configs/stage1_cutpoints/run_20261005_paper.json`): 0,820 / 0,749. Cutpoint của OFF:
`configs/stage1_cutpoints/run_20261024_nomv.json`.

## Nguồn

- Plan và quy tắc: `docs/handoff/PLAN-2026-11-27-main-pipeline-eval.md` (execution report phần 1 + bảng argmax).
- Trên máy train: `~/r1table_20261127/table_test.json`, `~/argmax_on_20261127/metrics.json` (chỉ số tổng hợp).
- Cách chấm: `scripts/evaluate_stage1.py` (không `--cutpoints` = argmax); OFF + R1:
  `scripts/residual_stack.py --save-arms` + `scripts/r1_stage1_table.py`.
