# PLAN 2026-10-28 — Train Stage 2 với nhiều cue MHCAC hơn để mô hình bớt chép cue sai

Status: **USER DUYỆT 2026-10-28** (arm share 0,5, 2 seed). **ĐÃ CHẠY 2026-10-29: KHÔNG XÁC LẬP**, nghiêng về phía xấu hơn; không chạy full. Pilot chỉ đọc trên val, không đổi code
hay YAML.

## Vì sao

Phép thử kênh trên adapter Stage 2 được báo cáo (`PLAN-2026-10-28-stage2-final-channel-probes.md`)
cho thấy mô hình đọc cue rất mạnh. Phân tích tiếp trên CPU (output C/O của phép thử đó, 1.360 study
val, 13 finding, nhãn CheXbert của báo cáo; script nằm ở `~/s2probe_20261028/analysis/` trên máy
train) cho kết quả:

- **10,6% ô cue sai** so với nhãn thật (tính nhị phân, có = Positive hoặc Uncertain).
- **Khi cue đúng,** báo cáo khớp cue ở 95,4% số ô.
- **Khi cue sai,** báo cáo viết theo cue ở **77,6%** số ô, và chỉ 22,4% viết theo nhãn thật:
  - cue nói không có nhưng thật ra có: 87,7% viết theo cue (1.022 ô);
  - cue nói có nhưng thật ra không: 65% viết theo cue (823 ô).
- **47% lỗi CheXbert của arm C nằm ở ô có cue sai.** Trong 931/1.029 ô đó báo cáo viết đúng như cue.
- **Mức chênh O − C trải trên nhiều finding:** Atelectasis 19%, Lung Opacity 13%, Pleural Effusion
  12%, Cardiomegaly 12%, Edema 10%. Không có vài finding nào chiếm phần lớn.
- Nhãn CheXpert và nhãn CheXbert của báo cáo tham chiếu khớp nhau ở 91,2% số ô.

**Giả thuyết:** adapter được train với 80% cue lấy từ nhãn thật (`--train-cue-model-share 0.2`), nên
mô hình học được rằng cue gần như luôn đúng. Lúc suy luận cue lại là của MHCAC. Nếu train với nhiều
cue MHCAC hơn, mô hình có thể học cách kiểm chứng cue bằng ảnh.

## Các arm

Arm đối chứng đã có sẵn. Không chạy lại arm nào cũ.

| arm | share | seed | checkout | output |
|---|---|---|---|---|
| ON16 (có sẵn) | 0,2 | 16 (mặc định) | `~/Meta-CXR-v3` @ `d9e2155` | `~/s2fusion_20261026/gen_ON_beam` |
| ON17 (có sẵn) | 0,2 | 17 | `~/Meta-CXR-s2seed` @ `5e8fefc` | `~/s2seed_20261026/gen_ON17_beam` |
| **M16 (mới)** | **0,5** | 16 | `~/Meta-CXR-s2seed` @ `5e8fefc` | `~/s2share_20261028/gen_M16_beam` |
| **M17 (mới)** | **0,5** | 17 | `~/Meta-CXR-s2seed` @ `5e8fefc` | `~/s2share_20261028/gen_M17_beam` |

Bước G của seed replicate 2026-10-26 đã chứng minh checkout `5e8fefc` với `--seed 16` cho ra adapter
bit-identical với `d9e2155`. Vì vậy M16 so với ON16 chỉ khác đúng share.

**Công thức** (giống hệt arm ON, chỉ đổi share):
- `--pipeline-mode meta_cxr_qformer_with_mhcac_prompt --section-mode findings_only`;
- Stage 1 `run_20261005_paper/phase1c` `checkpoint_best`;
- `--prompt-style paper_context --cue-rule cutpoints`, cutpoints `configs/stage1_cutpoints/run_20261005_paper.json`;
- `--train-cues oracle_mix --train-cue-model-share 0.5 --lora-lr 3e-4`;
- `--train-epochs 1 --batch-size 2 --grad-accum 8 --train-limit 10000`.

**Sinh báo cáo:** beam 4, length penalty 2, 256 token mới, toàn bộ 1.360 study val.

Draw của `oracle_mix` dùng chung một số ngẫu nhiên đều cho mỗi (study, finding). Nhờ vậy các ô dùng
cue MHCAC ở share 0,2 là tập con của các ô ở share 0,5.

## Guard (gặp một cái là dừng)

- GPU trống, không có tiến trình train hay sinh nào khác, ổ dữ liệu là `ntfs3`.
- `~/s2share_20261028` chưa tồn tại.
- `~/Meta-CXR-s2seed` ở `5e8fefc` và sạch. Kích thước và mtime của YAML Stage-1 vẫn như lúc chạy ON17
  (67.376 byte, mtime 1792078065). Mtime đổi thì `sample_key` đổi, kéo theo draw của `oracle_mix`.
- Log train in `[cues] train oracle_mix share=0.5` với `records 7604` và `cells 98852` (giống ON17).
  `from_model / cells` phải nằm trong [0,45; 0,55].
- Stage 1 load ra `missing=758 unexpected=0`. Adapter và `img_proj.pt` được ghi ra.
  `run_manifest.json` ghi đúng seed.
- Sinh được 1.360/1.360 báo cáo, 0 lỗi. Bốn file JSONL (ON16, ON17, M16, M17) có cùng thứ tự
  `sample_key` và cùng báo cáo tham chiếu.

## Quy tắc đọc (chốt trước khi chạy)

So sánh ghép cặp M_s − ON_s ở từng seed s ∈ {16, 17}, bootstrap 2.000 lần, seed 16. Ngưỡng nhiễu là
NB của Stage 2 ở budget pilot (`PLAN-2026-10-26-stage2-seed-noise.md`): CheXbert micro F1-14 0,0192,
BERTScore 0,0054. Mức chênh trung bình hai seed được so với NB/√2, tức 0,0136 và 0,0038.

- **"M tốt hơn"** (đủ điều kiện xin chạy full): cần cả bốn điều kiện sau.
  1. CheXbert micro F1-14 tăng ở cả hai seed.
  2. Trung bình hai seed > 0,0136.
  3. Cận dưới của CI bootstrap cho trung bình hai seed (lấy mẫu lại study, cùng tập chỉ số cho cả hai
     seed) > 0.
  4. BERTScore trung bình hai seed không thấp hơn −0,0038.
- **"M tệ hơn":** CheXbert micro F1-14 giảm ở cả hai seed và trung bình < −0,0136; hoặc BERTScore
  giảm ở cả hai seed và trung bình < −0,0038.
- Các trường hợp còn lại là **"không xác lập"**.
- **Báo cáo kèm, chỉ mô tả:**
  - ROUGE-L, BLEU-4, CIDEr, CE lexicon, CheXbert macro F1-14, micro/macro F1-5, độ dài trung vị;
  - **tỷ lệ chép cue sai** và tỷ lệ khớp cue đúng của cả bốn arm, đo như phần "Vì sao". Cách đo dùng
    `~/s2probe_20261028/analysis/cues.npz`, có cùng checkpoint, cùng cutpoints và cùng study.
- Nếu M tốt hơn, bước tiếp theo là chạy full Stage 2 với share 0,5 (khoảng 48 giờ), rồi chấm test
  đúng một lần. Bước đó cần user xác nhận riêng.

## Không làm

Không thử share khác, không đổi cue rule, không chấm test.

## Chi phí

Mỗi arm mất khoảng 71 phút train và 80 phút beam, cộng chấm điểm. Tổng khoảng 5,2 giờ GPU.

## Xử lý dữ liệu

JSONL, adapter, cache và nhãn chỉ nằm trên máy train. Chỉ số tổng hợp mới được đưa về.

## Execution report

Chạy 2026-10-28 21:49:50 → 2026-10-29 02:51:54 (5h02m), `~/s2share.sh`, log `~/s2share.log`, output
`~/s2share_20261028` trên máy train. Mọi guard đều qua:

- Stage 1 load `missing=758 unexpected=0`; `run_manifest.json` ghi đúng seed.
- `oracle_mix` share 0,5: `records 7604`, `cells 98852`. Tỷ lệ ô lấy từ MHCAC là 0,503 (M16) và
  0,505 (M17).
- `best_val_loss` (300 val, teacher-forced): M16 1,18753; M17 1,19139.
- Sinh 1.360/1.360 báo cáo, 0 lỗi, ở cả hai arm. Bốn file JSONL cùng thứ tự `sample_key` và cùng
  báo cáo tham chiếu.

### Kết quả: **KHÔNG XÁC LẬP**, nghiêng về phía xấu hơn

M_s − ON_s, val n = 1.360, beam 4 / lp 2, bootstrap ghép cặp 2.000 lần, seed 16:

| chỉ số | seed 16 | seed 17 | TB hai seed |
|---|---|---|---|
| **CheXbert micro F1-14** (quyết định) | −0,0082 [−0,0207, +0,0041] | −0,0028 [−0,0161, +0,0108] | **−0,0055** [−0,0150, +0,0043] |
| **BERTScore** (guard) | +0,0019 [−0,0036, +0,0071] | +0,0001 [−0,0058, +0,0057] | +0,0010 |
| ROUGE-L | −0,0017 | −0,0011 | −0,0014 |
| BLEU-4 | −0,0052 [−0,0107, −0,0001] | −0,0023 | −0,0037 |
| CIDEr | −0,0276 | +0,0041 | −0,0118 |
| CE lexicon macro F1 | −0,0252 [−0,0445, −0,0065] | −0,0122 | −0,0187 |
| CheXbert macro F1-14 | −0,0145 | −0,0055 | −0,0100 |
| CheXbert micro F1-5 | −0,0111 | −0,0143 | −0,0127 |
| CheXbert macro F1-5 | −0,0150 | −0,0067 | −0,0108 |

Báo cáo giống hệt nhau giữa M và ON: 435 (s16) và 416 (s17) trên 1.360. Độ dài trung vị gần như
không đổi: ON/M là 35/35 từ ở seed 16 và 35/36 từ ở seed 17.

Đối chiếu với quy tắc chốt trước:

- **"M tốt hơn" trượt.** CheXbert micro F1-14 giảm ở cả hai seed, nên điều kiện 1 trượt. Điều kiện 2
  (> 0,0136) và điều kiện 3 (cận dưới CI > 0) cũng trượt. Chỉ điều kiện 4 (BERTScore ≥ −0,0038) qua.
- **"M tệ hơn" cũng không đạt.** CheXbert giảm ở cả hai seed nhưng trung bình −0,0055 chưa vượt
  −0,0136. BERTScore không giảm ở cả hai seed.
- Kết luận: **không xác lập**. Hai CI có dấu * (BLEU-4 và CE lexicon ở seed 16) đều nằm trong nhiễu
  seed. Cả 7 chỉ số mô tả có trung bình hai seed âm, chỉ BERTScore dương.

### Tỷ lệ chép cue (mô tả)

Đo trên 17.407 ô biết nhãn thật, gồm 1.845 ô cue sai (823 cue nói có nhưng thật ra không, 1.022 cue
nói không nhưng thật ra có). Bốn arm dùng cùng cue MHCAC và cùng study.

| arm | khớp cue khi cue đúng | theo cue khi cue sai | khi cue nói có (sai) | khi cue nói không (sai) | phần lỗi nằm ở ô cue sai |
|---|---|---|---|---|---|
| ON16 | 0,942 | 0,704 | 0,532 | 0,841 | 0,409 |
| ON17 | 0,945 | 0,721 | 0,546 | 0,863 | 0,425 |
| M16 | 0,941 | 0,686 | 0,478 | 0,853 | 0,405 |
| M17 | 0,941 | 0,696 | 0,516 | 0,841 | 0,411 |

Tăng share lên 0,5 giảm tỷ lệ theo cue sai rất ít: trung bình 0,713 → 0,691. Phần giảm nằm ở ô cue
nói có nhưng thật ra không (0,539 → 0,497). Ở ô cue nói không nhưng thật ra có thì không đổi
(0,852 → 0,847), và đây là loại lỗi chiếm đa số. Mô hình không học được cách dùng ảnh để bù cho cue
bỏ sót. Phần độ chính xác lấy được ở ô FP bị mất ở chỗ khác, nên CheXbert không tăng. (Adapter full
được báo cáo theo cue sai 0,776; hai ON pilot ở mức 0,70–0,72, vì budget khác.)

### Kết luận

Giả thuyết "train với nhiều cue MHCAC hơn sẽ giúp mô hình kiểm chứng cue bằng ảnh" **không được ủng
hộ** ở budget pilot. Không chạy full share 0,5. Recipe Stage 2 được báo cáo giữ nguyên share 0,2.
Giới hạn:
- hai seed, 10.000 study train, 1 epoch;
- chỉ đọc trên val, không chấm test;
- một mức share duy nhất.
