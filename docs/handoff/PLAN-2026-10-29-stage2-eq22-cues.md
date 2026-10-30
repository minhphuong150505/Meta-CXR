# PLAN 2026-10-29: Stage 2 với cue Eq. 22 của bài báo (ngưỡng thiên về recall)

**Trạng thái:** ĐÃ CHẠY 2026-10-29/30 — **E TỆ HƠN** theo quy tắc đã chốt (xem Execution report). User duyệt chạy 2026-10-29. Không cần sửa code, không sửa YAML. Đây là hướng claude chọn khi
user nhờ "chọn hướng tiếp theo đi", sau khi hai hướng tăng recall ở Stage 1 (asymmetric loss, mở khóa
encoder) đều thất bại.

## 1. Vì sao chọn hướng này

- **Không bỏ sót = cue âm tính sai ít đi.** Theo audit 2026-10-29 trên val, cross-fit:
  - cutpoint đang dùng: recall P/U 0,556, precision 0,641, 1,96 cue dương tính mỗi study;
  - ngưỡng **Eq. 22** (ngưỡng ROC-distance theo từng lớp mà bài báo dùng để tạo prompt; ô không lớp nào
    vượt ngưỡng thì bỏ trống): recall **0,801**, precision 0,357, 5,06 cue mỗi study.
- **Stage 2 sai không đối xứng.** PLAN-2026-10-28-stage2-cue-share đo được:
  - adapter chép theo **khoảng 85%** cue âm tính sai, tức bệnh có thật nhưng cue nói không có;
  - nhưng chỉ chép theo **khoảng 50%** cue dương tính sai.

  Đổi bớt cue âm tính sai lấy cue dương tính sai vì vậy có thể có lợi ròng. Đó là giả thuyết, chưa đo.
- **Đúng khuôn khổ bài báo.** Eq. 22 chính là quy tắc bài báo dùng cho prompt Stage 2. Hiện recipe được
  báo cáo (P2) lại đang dùng cutpoint.
- **Phép thử trước đây không trả lời được câu hỏi này.** Đổi cue chỉ lúc suy luận (2026-10-11) trên
  adapter v3 cho kết quả ngang. Nhưng v3 bỏ qua cue, và adapter đó được train trên cue Eq. 22. Còn adapter
  P2 thì đọc cue (2026-10-28: cue thật cho CheXbert micro F1-14 +0,091). Phép thử công bằng là train lại
  Stage 2 với cue Eq. 22, đúng như CLAUDE.md đã ghi ("a cutpoint-cue Stage-2 retrain is the untested fair
  version", theo chiều ngược lại).

## 2. Arm

Công thức pilot giống hệt ON16 / ON17 (control đã có) và M16 / M17 (pilot share 0,5):

- Stage 1 `run_20261005_paper/phase1c` `checkpoint_best`; P2 (`paper_context` + `oracle_mix` share 0,2
  mặc định); LR 3e-4; 10.000 study train; 1 epoch; batch 2 × 8.
- Sinh beam 4, length penalty 2, 256 token, trên cả 1.360 study val.
- Checkout `~/Meta-CXR-s2seed` @ `948964e`, YAML 67.376 byte, mtime 1792078065, như các control.

| Arm | Khác control ở đâu | Seed | Control |
|---|---|---|---|
| E16 | `--cue-rule paper_thresholds --threshold-path ~/eval_20261005_paper/thresholds_eq22_val.json` | 16 | ON16 (`~/s2fusion_20261026/gen_ON_beam`) |
| E17 | như trên | 17 | ON17 (`~/s2seed_20261026/gen_ON17_beam`) |

**File ngưỡng:**

- Định dạng `meta_cxr_per_class_roc_distance_v1`, fit trên val (1.808 study) từ
  `run_20261005_paper/phase1c/.../val_predictions_epoch_best.npz`, đề ngày 2026-10-06.
- Cutpoint của control cũng fit trên cùng val đó, nên hai arm ngang nhau về điểm này.

**Điều thay đổi:**

- Cue của val khi sinh.
- 20% cue train lấy từ MHCAC (phần còn lại vẫn là nhãn thật).
- Như bẫy đã ghi trong CLAUDE.md: file ngưỡng khác thì draw `oracle_mix` cũng khác, tương tự một seed
  khác. Phần biến động này nằm trong NB của Stage 2.

**Các cổng kiểm tra (dừng ngay nếu một cổng không qua):**

- GPU rảnh, không có job khác, ổ `ntfs3`, thư mục đích chưa có.
- Checkout đúng `948964e`, sạch; YAML đúng kích thước và mtime trên.
- Log ghi `cue_rule paper_thresholds`, nạp Stage 1 với `missing=758 unexpected=0`, `oracle_mix` có
  from_model share trong khoảng 0,15–0,25.
- Sinh đủ 1.360 báo cáo, 0 lỗi.
- Bốn file JSONL khớp nhau về `sample_key` và reference.

**Chi phí:** pilot share 0,5 cùng công thức mất 5h02m cho hai seed, tính cả chấm điểm. **Ước tính
khoảng 5 giờ GPU.**

## 3. Quy tắc đọc, chốt TRƯỚC khi chạy

Giống hệt PLAN-2026-10-28-stage2-cue-share (script `~/s2share_verdict.py`). NB của Stage 2 ở budget pilot
là CheXbert micro F1-14 0,0192 và BERTScore 0,0054, nên NB/√2 là 0,0136 và 0,0038.

**E TỐT HƠN** khi đạt đủ bốn điều kiện:

1. CheXbert micro F1-14 tăng ở **cả hai** seed.
2. Trung bình hai seed > 0,0136.
3. Cận dưới khoảng tin cậy của trung bình (bootstrap theo study, 2.000 lần, seed 16) > 0.
4. BERTScore trung bình ≥ −0,0038.

**E TỆ HƠN** khi:

- CheXbert giảm ở cả hai seed và trung bình < −0,0136, **hoặc**
- BERTScore giảm ở cả hai seed và trung bình < −0,0038.

**Mọi trường hợp còn lại:** KHÔNG XÁC LẬP.

**Chỉ để mô tả:**

- ROUGE-L, BLEU-4, CIDEr, CE lexicon.
- CheXbert macro F1-14, micro và macro F1-5.
- **Recall và precision CheXbert** (để thấy E có bớt bỏ sót thật không).
- Độ dài báo cáo, số báo cáo trùng hệt giữa hai arm.

**Nếu E TỐT HƠN:** không chạy full ngay. Báo user, và đề xuất chạy thêm seed hoặc một run full (khoảng
48 giờ) như một quyết định riêng.

**Nếu không đạt:** ghi lại kết quả. Recipe được báo cáo giữ cue cutpoint.

## 4. Giới hạn đã biết trước

- Chỉ là pilot (10.000 study, 1 epoch), 2 seed, chỉ val.
- Một điểm ngưỡng duy nhất (Eq. 22), không dò.
- Eq. 22 bỏ trống những ô không lớp nào vượt ngưỡng, nên số finding được nhắc trong prompt cũng thay đổi.
  Thay đổi đó là một phần của quy tắc bài báo, không tách riêng ra được.

## Execution report

**Kết luận theo quy tắc đã chốt: E TỆ HƠN** (vì BERTScore giảm ở cả hai seed, trung bình −0,0063 <
−0,0038). Không chạy full. Recipe được báo cáo giữ cue cutpoint.

**Chạy:**

- User duyệt ("ok") ngày 2026-10-29. Chuỗi `~/s2eq22.sh` được khởi chạy đúng một lần lúc 22:45:24 và
  kết thúc `S2EQ22_DONE` lúc 03:58:00 ngày 2026-10-30, tổng 5h13m.
- Checkout `~/Meta-CXR-s2seed` @ `948964e`. Mọi cổng kiểm tra trong script đều qua, nên script không dừng
  giữa chừng:
  - `cue_rule paper_thresholds` có trong manifest lúc train và lúc sinh;
  - nạp Stage 1 đúng;
  - 1.360/1.360 báo cáo mỗi arm;
  - bốn file JSONL khớp nhau (`aligned: True`).
- Phần cue train lấy từ MHCAC: E16 20.521/98.852 = 0,208; E17 20.713/98.852 = 0,210.
- Val loss: E16 1,19159; E17 1,19182.

**E − ON, val n = 1.360, beam 4 / lp 2:**

| Chỉ số | Seed 16 [CI] | Seed 17 [CI] | Trung bình |
|---|---|---|---|
| **BERTScore** (quyết định) | −0,0077 [−0,0141, −0,0020] | −0,0049 [−0,0107, +0,0010] | **−0,0063** (NB/√2 0,0038) |
| **CheXbert micro F1-14** (quyết định) | −0,0056 [−0,0216, +0,0103] | +0,0012 [−0,0145, +0,0162] | **−0,0022** [−0,0148, +0,0101] |
| ROUGE-L | −0,0049 [−0,0102, −0,0001] | −0,0017 | −0,0033 |
| BLEU-4 | −0,0047 | −0,0000 | −0,0024 |
| CIDEr | −0,0082 | +0,0028 | −0,0027 |
| CE lexicon macro F1 | −0,0195 | +0,0098 | −0,0048 |
| CheXbert macro F1-14 | −0,0184 | +0,0188 [+0,0008, +0,0381] | +0,0002 |
| CheXbert micro F1-5 | −0,0233 | +0,0013 | −0,0110 |
| CheXbert macro F1-5 | −0,0213 | +0,0208 | −0,0003 |

**Recall và precision CheXbert (chỉ mô tả, micro trên 14 finding):**

| Arm | Precision | Recall | Số finding dương tính / báo cáo |
|---|---|---|---|
| ON16 | 0,5245 | 0,4431 | 1,89 |
| E16 | 0,5029 | 0,4497 | 2,00 |
| ON17 | 0,5409 | 0,4543 | 1,88 |
| E17 | 0,5253 | 0,4681 | 1,99 |

Báo cáo tham chiếu có 2,24 finding dương tính mỗi báo cáo.

Số báo cáo trùng hệt giữa E và ON: 337 (seed 16) và 353 (seed 17). Độ dài trung vị ON / E: 35 / 37 từ
(seed 16) và 35 / 38 từ (seed 17).

**Đọc kết quả:**

- Cue Eq. 22 làm Stage 2 **nhắc nhiều finding hơn**, khoảng +0,11 finding mỗi báo cáo.
- **Recall CheXbert tăng rất ít:** +0,007 (seed 16) và +0,014 (seed 17).
- **Precision giảm nhiều hơn:** −0,022 và −0,016. Vì vậy micro F1-14 đứng yên (−0,0022, trong khoảng
  nhiễu), còn BERTScore giảm vượt NB/√2.
- So với Stage 1: Eq. 22 nâng recall P/U của cue từ 0,556 lên 0,801 (audit 2026-10-29). Nhưng chỉ một
  phần rất nhỏ của mức tăng đó đến được báo cáo. Phần lớn cue dương tính thêm vào là cue sai: precision
  của cue chỉ 0,357.
- Như vậy, ở budget pilot, đổi bớt cue âm tính sai lấy cue dương tính sai **không** cho lợi ròng.
  Giả thuyết bất đối xứng ở mục 1 (chép cue âm tính sai 85%, cue dương tính sai 50%) không chuyển thành lợi
  ích đo được.

**Giới hạn:** pilot (10.000 study, 1 epoch), 2 seed, chỉ val, một điểm ngưỡng duy nhất. Không kết luận
gì về run full.
