# Kết quả multi-view (View Fusion): tổng hợp, kết quả âm

Cập nhật 2026-10-27. Tài liệu này gom mọi thí nghiệm nhằm trả lời một câu hỏi:
**dùng thêm ảnh thứ hai của study (thường là ảnh lateral) có giúp Stage 1 phân
loại tốt hơn so với chỉ dùng ảnh chính hay không?** Câu hỏi đặt ra từ 2026-10-25,
khi user muốn View Fusion là tính mới của luận văn.

Số liệu ở đây là số tổng hợp, lấy từ các plan trong `docs/handoff/`; nguồn ghi ở
từng mục. Không có ID bệnh nhân hay văn bản báo cáo trong file này. File `.npz`,
checkpoint và log nằm trên máy train.

## Kết luận ngắn

**Không.** Ở mọi cách đã thử, mô hình dùng ảnh thứ hai không thắng mô hình chỉ
dùng ảnh chính (OFF) một cách vượt nhiễu seed:

- Run đầy đủ: ON hơn OFF theo argmax, nhưng phần hơn bằng nhau ở study một ảnh
  và study hai ảnh, nên không do ảnh thứ hai. Theo cutpoint (số chính của dự án)
  hai bên hòa.
- Pilot: tám biến thể (early fusion với ba mức drop ảnh phụ, early fusion chỉ ở
  luồng Swin, ba biến thể late fusion, token ảnh phụ vào MHCAC) đều không vượt OFF
  theo quy tắc khai báo trước.
- Stage 2 sinh báo cáo trên Stage 1 ON và OFF: không khác nhau vượt nhiễu seed.

Đây là kết quả âm có kiểm soát: chọn trên val, quy tắc đọc ghi trước khi chạy,
OFF cùng recipe và cùng budget, đo nhiễu seed. Nó được báo cáo như một kết quả,
không phải bị bỏ qua.

## Giao thức đánh giá

- **Ba lớp** (Negative / Positive / Uncertain) cho mỗi finding, đúng như bài báo
  META-CXR (D-023).
- **Chỉ số:** weighted F1 trung bình 14 finding (wF1), F1 trung bình 5 finding của
  bài báo (F1-5), macro recall, AUROC trung bình. Số chính của dự án dùng hai
  cutpoint mỗi finding fit trên val (*không phải giao thức của bài báo*); argmax
  luôn được in kèm.
- **Nhóm con:** `views_1` (study chỉ có một ảnh, ảnh phụ không tồn tại nên không
  thể giúp) và `views_2` (study có ảnh phụ). Nếu ảnh thứ hai thật sự có ích, phần
  lợi phải nằm ở `views_2`.
- **Chọn trên val, test chỉ đọc một lần** cho ứng viên cuối. Pilot không đọc test.
- **Bootstrap ghép cặp** theo study, 1.000 hoặc 2.000 lần, seed 16. Dấu `*` = CI
  95% không chứa 0.
- **Ngưỡng nhiễu seed (NB).** Bootstrap trên val chỉ đo nhiễu do chọn study, không
  đo nhiễu do seed huấn luyện. Từ 2026-10-26 (user quyết định), một pilot chỉ được
  đi tiếp khi quy tắc CI qua VÀ hiệu số vượt NB đo ở cùng budget. Với trung bình
  hai seed, ngưỡng là NB/√2.
- **Pilot recipe:** chỉ pha 1b, 60.000 study, 2 epoch, khởi từ
  `run_20261005_paper/checkpoint_phase1a.pth`, val n = 1.808 (views_1 988,
  views_2 820), cutpoint fit chéo hai nửa val.

## 1. Run đầy đủ: ON so với OFF (test, n = 3.269)

ON là model đang báo cáo (`run_20261005_paper`, early fusion). OFF là
`run_20261024_nomv` (`multi_view: false`, không StreamAdapter / MPC / view
consistency), cùng checkpoint 1a, cùng recipe 1b + 1c. Một run mỗi arm.

| ON − OFF, argmax | toàn test | views_1 (2.022) | views_2 (1.247) |
|---|---|---|---|
| F1-5 | +0,0129 [+0,0090; +0,0174] * | +0,0126 | +0,0131 |
| weighted F1 | +0,0057 * | | |
| macro recall | +0,0060 * | | |
| AUROC trung bình | +0,0054 [−0,0005; +0,0112] | −0,0003 | +0,0107 [+0,0007; +0,0207] * |

- Phần hơn về F1 bằng nhau ở hai nhóm, nên theo quy tắc khai báo trước nó đến từ
  StreamAdapter / MPC / nhiễu, không từ ảnh thứ hai. Chỉ AUROC tách theo nhóm,
  chủ yếu ở lớp Uncertain.
- Theo cutpoint (số chính, *không phải giao thức của bài báo*): F1-5 ON 0,749,
  OFF 0,748. Hai arm hòa.
- OFF train pha 1b nhanh hơn khoảng 27% (5h12m so với 7h08m).

Nguồn: `docs/handoff/PLAN-2026-10-24-multi-view-ablation.md`.

## 2. Phân tích trên model ON (không train lại)

⚠ Bước 0, 0b, 0c dùng test sau khi test đã được đọc ở mục 1. Chúng là phân tích
mô tả, không phải kết quả kiểm định giữ lại.

| Phân tích | Kết quả |
|---|---|
| Bước 0: chấm ON với ảnh phụ bị bỏ (`max_aux_views=0`), argmax, test views_2 | Ảnh phụ được dùng (6,4% ô argmax đổi) và **làm giảm** F1-5 −0,0116 [−0,0176; −0,0051] *, wF1 −0,0056 *, AUROC phẳng (−0,0005). Giúp Pleural Effusion, hại Atelectasis / Consolidation (wF1 khoảng −0,04, AUROC phẳng): dịch điểm cắt |
| Bước 0b: cùng so sánh nhưng mỗi arm dùng cutpoint val riêng | Ảnh phụ **giúp** views_2: wF1 +0,0053 [+0,0014; +0,0090] *, macro recall +0,0193, F1-5 +0,0067 (CI cắt 0). ON − OFF ở views_2 F1-5 +0,0146 *, nhưng ở views_1 ON kém hơn (F1-5 −0,0084) |
| Bước 0c: cutpoint fit riêng theo số view | **Không qua:** toàn test wF1 +0,0019 [−0,0001; +0,0038], F1-5 +0,0022 (CI cắt 0). Phần thua ở study một ảnh là do model, không do cutpoint |
| Tắt fusion từng luồng (val views_2, suy luận) | Luồng Swin mang phần lợi: F − noSW wF1 +0,0045 *, F1-5 +0,0070 *, AUROC +0,0162 *. Fusion BioViL chỉ giúp AUROC (+0,0097 *), PubMedCLIP gần như không |
| "Ảnh lateral giúp Pleural Effusion" (khai báo trước, val 797 study) | **Không xác nhận:** AUROC Pos +0,0056 [−0,0009; +0,0129] |
| "Ảnh lateral tăng AUROC trung bình" (khai báo trước, test 1.196 study) | **Không xác nhận:** −0,0001 [−0,0080; +0,0105] (val từng là +0,0229 *); hai hiệu ứng đổi chỗ giữa hai split |

Nguồn: `PLAN-2026-10-25-multi-view-step0.md`, `PLAN-2026-10-25-fusion-per-stream.md`,
`PLAN-2026-10-27-multi-view-evidence-val.md`, `PLAN-2026-10-27-multi-view-test-auroc.md`.

## 3. Nhiễu seed ở budget pilot

P15 (early fusion như YAML) và OFF chạy lại ở `run.seed=17`. Cùng recipe, đổi seed,
AUROC val đổi −0,0159 * (ON) và −0,0112 * (OFF). Bootstrap trên val không bao được
nhiễu này.

| NB, toàn val | wF1 | F1-5 | macro recall | AUROC |
|---|---|---|---|---|
| từ 2 seed (2026-10-26) | 0,0027 | 0,0031 | 0,0110 | 0,0276 |
| từ 4 seed, TK và OFF gộp (2026-10-27, mô tả) | 0,0036 | 0,0081 | 0,0101 | 0,0138 |

Ước lượng từ bốn seed cho F1-5 lớn hơn nhiều so với hai seed, nên NB hai seed là
ngưỡng dễ dãi. P15 − OFF trung bình hai seed: F1-5 −0,0015.

Nguồn: `PLAN-2026-10-26-pilot-seed-noise.md`.

## 4. Các biến thể pilot (val, n = 1.808)

| Biến thể | Ý tưởng | Kết quả chính (so với OFF trừ khi ghi khác) | Kết luận |
|---|---|---|---|
| P30, P50 | Tăng xác suất bỏ ảnh phụ khi train (YAML 0,15) | Study một ảnh **xấu hơn**: P50 − P15 views_1 wF1 −0,0056 *, P30 −0,0029 *. P30 giúp views_2 F1-5 +0,0058 * so với P15 nhưng tổng không đổi | Dừng |
| SW | Chỉ fuse luồng Swin | SW − P15 toàn val F1-5 −0,0071 *, AUROC −0,0149 *, cả ở study một ảnh | Dừng |
| LF | Late fusion: MHCAC phân loại từng view, gộp logit bằng attention | 2 seed: wF1 +0,0000 [−0,0017; +0,0017], F1-5 −0,0005 [−0,0034; +0,0024]. Bộ gộp học thành trung bình cộng (trọng số ảnh phụ ~0,5) | Dừng |
| LD | LF, nhánh ảnh phụ không truyền gradient | 2 seed: wF1 +0,0001, F1-5 +0,0007, CI cắt 0. Hết thiệt ở study một ảnh nhưng cũng hết lợi ở study hai ảnh | Dừng |
| AH | LF với MHCAC riêng cho ảnh phụ | 2 seed: wF1 −0,0017 [−0,0036; +0,0002], F1-5 +0,0004. views_2 macro recall −0,0248 *, wF1 −0,0043 *, vượt NB/√2 | Dừng |
| TK | Token ảnh phụ nối vào trục key của MHCAC | Xem mục 5 | Dừng sau khi chạy lặp seed |

Nguồn: `PLAN-2026-10-25-aux-view-drop.md`, `PLAN-2026-10-25-swin-only-fusion.md`,
`PLAN-2026-10-26-late-view-fusion.md`, `PLAN-2026-10-26-late-fusion-detach-aux.md`,
`PLAN-2026-10-27-late-aux-head.md`.

Biến thể cũ hơn cũng thua OFF hoặc model gốc: `max_aux_views: 3` (2026-10-07, run
đầy đủ, F1-5 −0,0114 *, AUROC −0,0085 *) và hai cờ theo EVOKE (2026-10-10, run
đầy đủ, F1-5 −0,0160 *).

## 5. TK: lần qua ở pilot không lặp lại được

Pilot hai seed (42, 17) qua quy tắc trên F1-5: +0,0036 [+0,0006; +0,0067], vượt
NB/√2 = 0,0022. Nhưng phần lợi nằm ở study một ảnh, nơi TK không đọc ảnh phụ
(xác suất trùng với khi bỏ ảnh phụ tới 2,4e-7), và ở study hai ảnh gần như phẳng.
Vì vậy user duyệt chạy thêm seed 7 và 23 cho cả TK và OFF, với quy tắc ghi trước
**bắt buộc có phần lợi ở study hai ảnh**.

| TK − OFF, toàn val | seed 42 | seed 17 | seed 7 | seed 23 | trung bình 4 seed [CI] | NB/√2 |
|---|---|---|---|---|---|---|
| weighted F1 | +0,0022 * | +0,0005 | −0,0010 | +0,0021 | +0,0010 [−0,0002; +0,0024] | 0,0019 |
| F1-5 | +0,0066 * | +0,0006 | −0,0051 * | +0,0045 * | +0,0016 [−0,0007; +0,0042] | 0,0022 |
| macro recall | +0,0053 * | +0,0003 | −0,0046 * | +0,0078 * | +0,0022 [−0,0003; +0,0051] | 0,0077 |
| AUROC | −0,0074 * | +0,0077 | +0,0017 | 0,0000 | +0,0005 | 0,0195 |

- Ở study hai ảnh: wF1 +0,0010 [−0,0008; +0,0030], F1-5 +0,0002 [−0,0031; +0,0038].
- **Kết quả: KHÔNG QUA.** Quy tắc 1 (toàn val) và quy tắc 2 (study hai ảnh) đều
  trượt; quy tắc 3 và 4 (không thiệt) qua.
- Cùng model TK, bật / tắt ảnh phụ, ở study hai ảnh: weighted recall +0,0030 *,
  macro recall −0,0051 *, AUROC +0,0103 *. Ảnh phụ có dịch điểm cắt và xếp hạng
  một chút, trong nhiễu seed.
- Attention ở cả bốn seed: mỗi finding lấy 31–38% từ ảnh phụ (chia đều là 50%),
  không finding nào dựa chủ yếu vào ảnh lateral.

Nguồn: `PLAN-2026-10-27-aux-tokens-mhcac.md` (mục "Seed replicate — execution report").

## 6. Stage 2 trên Stage 1 ON và OFF

Stage 2 train lại trên từng Stage 1 (10.000 study, cùng recipe pilot), beam 4, val
1.360 study, hai seed Stage 2:

- Trung bình hai seed ON' − OFF: CheXbert micro F1-14 −0,0115, BERTScore −0,0020.
- NB/√2 tương ứng: 0,0136 và 0,0038. Trong nhiễu.
- Hướng nghiêng về OFF (seed 17 âm trên cả 9 chỉ số), nhưng không kết luận được
  chiều nào.

Nguồn: `PLAN-2026-10-26-stage2-fusion-on-off.md`, `PLAN-2026-10-26-stage2-seed-noise.md`.

## Vì sao có thể không giúp (giả thuyết, chưa chứng minh)

- **Ít study có ảnh phụ thật sự bổ sung.** Ảnh phụ phần lớn là một ảnh lateral;
  ảnh thứ ba trở đi chỉ có ở khoảng 5% study.
- **Một MHCAC dùng chung phải phục vụ cả hai view.** LF cho thấy ảnh phụ chỉ giúp
  study hai ảnh khi MHCAC được train trên ảnh lateral, và chính việc đó làm ảnh
  chính kém đi gần bằng (LD bỏ được cả lợi lẫn hại).
- **Luồng BioViL đưa vào fusion không phải token theo vị trí** (reshape không
  permute, projector ngẫu nhiên đóng băng ở 1b), nên chỉ PubMedCLIP và Swin cho
  fusion token không gian thật. Phân tích từng luồng ở mục 2 phù hợp với điều này.
- **Nhãn CheXpert lấy từ báo cáo**, mà báo cáo viết sau khi xem mọi ảnh. Ảnh
  frontal thường đã đủ để đoán phần lớn nhãn.

## Giới hạn

- Pilot chỉ là pha 1b, 60.000 study, 2 epoch. Kết quả loại bỏ được các biến thể ở
  budget này, không loại bỏ được ở budget đầy đủ.
- Run đầy đủ chỉ có một run mỗi arm; chưa có NB ở budget đầy đủ.
- NB hai seed ước lượng thấp nhiễu thật (mục 3), nên các quyết định "dừng" dựa
  trên NB hai seed là thận trọng theo hướng đúng: chúng không loại nhầm một biến
  thể đáng lẽ qua.
- TK là biến thể thứ tư (tính cả nhóm late) so với cùng các arm OFF. Thử nhiều lần
  trên cùng ngưỡng làm tăng xác suất qua do may; lần qua ở pilot TK là ví dụ.
- Các phân tích bước 0 / 0b / 0c dùng test sau khi đã đọc test; chỉ mang tính mô tả.

## Ghi vào luận văn thế nào

- Model báo cáo vẫn là `run_20261005_paper` (early fusion bật). Số test theo
  cutpoint (*không phải giao thức của bài báo*): weighted P / R / F1
  0,827 / 0,834 / 0,820, F1-5 0,749 [0,741; 0,757]. Theo argmax: 0,839 / 0,781 /
  0,780, F1-5 0,669.
- View Fusion được trình bày là **khác biệt so với bài báo, kèm ablation âm**: nó
  không cải thiện phân loại một cách đo được so với chỉ dùng ảnh chính. Không viết
  nó là đóng góp làm mô hình tốt hơn.
- Phần có giá trị đóng góp là giao thức: tách theo nhóm `views_1` / `views_2`, bỏ
  ảnh phụ khi suy luận để kiểm tra mô hình có dùng ảnh phụ không, và ngưỡng nhiễu
  seed. Chính giao thức này cho thấy các phần lợi bề ngoài (run đầy đủ theo argmax,
  pilot TK) không đến từ ảnh thứ hai.
- Không thêm biến thể multi-view nào nếu user chưa quyết định mới.
