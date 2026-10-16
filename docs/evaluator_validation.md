# Evaluator validation

Ngày chạy: 2026-07-22 · Branch `feature/complete-evaluator` · Python 3.12, numpy 2.5.1

Mọi con số dưới đây là **output thật** của code trong repo, chạy trên dữ liệu tổng hợp nhỏ do
chính tài liệu này định nghĩa. **Không có số nào của mô hình thật** — chưa có lần inference nào
trên MIMIC-CXR được thực hiện. Mục đích là chứng minh evaluator hành xử đúng, không phải báo cáo
chất lượng mô hình.
   no Stage-1 model, checkpoint, Q-Former or MHCAC
2. "MedGemma vision tower" — SigLIP, patch 14, 896×896 (aspect NOT preserved)
   → projector pools 4× → 256 image tokens (1chắc hơn là nhờ mô hình sinh ảnh vẽ lại hơn 60 nhãn chữ nhỏ.
- XOÁ dòng vàng cuối "Target guided architecture: ...".

=========== NHÁNH 1 — STAGE 1 (4 hộp) =======
1. "Multi-view vision encoders"
   BioViL-T · 448² · 14×14 = 196 tokens
   PubMedCLIP · 224² · 1 CLS + 7×7 = 50 token
   Per-encoder view fusion: anchor = Query, auxiliary = Key/Value
2. "Projection & concatenation"
   FC → d_model = 1408 ·  246 visual tokens =
   (KHÔNG ghi "shared" — Stage 2 không dùng các token này)
3. "MHCAC"
   14 learnable class queries · multi-head cr
   Report text is a TRAINING-ONLY teacher; inference is image-only
4. "Two prediction heads"  ← đây là điểm đóng góp, phải vẽ rõ hai đầu:
   · Classification head → q : Positive / Negion

        Stage 2: 4 MedGemma QLoRA fine-tuning (arm A / arm B) -> 5 Report generation
        XAI:     6 Post-hoc sentence-level attribution + 2 sanity gates

================ NHỮNG GÌ ĐANG ĐÚNG — GIỮ NGUYÊN ================
- Input: một STUDY, không phải một ảnh. Ancho lateral),
  tối đa 1 auxiliary view (thường là lateral). Cả hai view đi qua CẢ HAI encoder.
- Hai encoder, Swin đã bị loại bỏ.
- Phép cộng token: 246 = 196 (BioViL) + 1 CLS + 49 (PubMedCLIP), chiếu về d_model = 1408.
- Mỗi encoder giữ nguyên độ phân giải gốc của nó; không pooling, không bỏ CLS.
- Positional encoding riêng cho từng encoder.
- Text embeddings chỉ đi vào MHCAC trong lúc TRAIN (teacher), suy luận chỉ dùng ảnh.
- Đầu ra: FINDINGS + IMPRESSION.

---

Một điều cần nói thẳng: mô hình sinh ảnh sẽ l hơn 100 nhãn, khả năng cao chữ sẽ bị nhoèhoặc sai chính tả, và một hình như thế không dùng được trong luận văn.

Đường chắc ăn hơn: dán đúng prompt trên nhưngã draw.io XML (hoặc SVG) để tôi mở và chỉnh".Chữ sẽ là text thật, sắc nét ở mọi kích thước, và bạn sửa được. Nếu bạn muốn, tôi tự sinh file draw.io/SVG đó luôn — tôi đã có đủ số liệu chính xác.

✻ Brewed for 1m 29s · done 7:22 PM

❯ [Image #2] bạn đánh giá pipeline này đã đúng rce code chưa tôi muốn một pipeline chínhtôi để xác minh
[Image #2]

Để trả lời chính xác chứ không đoán, tôi đọc


Tôi đã đọc source. Kết luận: sơ đồ này mô tả mục tiêu, không mô tả code. Dòng cảnh báo đỏ của chính nó là đúng — và còn
nhẹ hơn thực tế.

Vấn đề chặn: arm B hiện là no-op im lặng

Không phải "chưa implement nên sẽ báo lỗi". Ncho ra một model giống hệt arm A, chỉ khácseed. Chuỗi bằng chứng:

┌────────────────────┬─────────────────────────────────────────────────────────────────┐
│        bước        │                 file:line                  │                     thực tế                     │
├────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ Native records     │                        sample_key, ref, image_path,             │
│ sinh ra ở đâu      │ training/dataio/manifest.py:229-243        │ anchor_view, auxiliary_views — không có         │
│                    │                                            │ pred_groups                                     │
├────────────────────┼─────────────────────────────────────────────────────────────────┤
│ Thiếu field thì    │ stage2/prompts/records.py:41-43            │ _as_tuple(None) → (), im lặng, không lỗi        │
│ sao                │                                            │                                                 │
├────────────────────┼─────────────────────────────────────────────────────────────────┤
│ Có chốt chặn không │ builder.py / validation.py                 │ không có guard nào raise khi guided mode nhận 0 │
│                    │                                            │  finding                                        │
├────────────────────┼─────────────────────────────────────────────────────────────────┤
│ Ai sinh            │ train_eval_figure9_llm_variants_200.py:545 │ chỉ build_stage1_records — và docstring của nó  │
│ pred_groups        │                                            │ (dòng 475-481) cấm native mode gọi vào          │
└────────────────────┴─────────────────────────────────────────────────────────────────┘

(Ghi chú: arm A đang chạy full dataset nhưng --train-epochs 1, batch 2 × accum 8.)

Ba điều tôi không tự quyết được:

● User answered Claude's questions:
· Arm B hiện sẽ chạy 70h và cho ra prompt  nào). Xử lý thế nào trước khi arm A xong? →Bỏ arm B khỏi pipeline
· Nếu nối cue vào arm B, dùng điểm nào của Stage 1 để sinh cue P/N/U? → marginal_presence = m × q_pos, ngưỡng
calibrate trên val (Khuyến nghị)
· Hình pipeline chính thức cho luận văn nên bao gồm những gì? → Lớp XAI Stage-2, "Chỉ luồng chính, gọn nhất có thể"

Rõ. Bỏ arm B là việc gấp — nó đang nằm trong ain trước.


Máy sống, mạng thông — vậy là Tailscale SSH đòi phê duyệt lại. Lấy link:

  Ran 1 shell command

Cần bạn bấm duyệt: https://login.tailscale.co

Trong lúc chờ, tôi soạn sẵn pipeline mới — và lần này đưa vào repo thay vì để file rời trên host, đúng như cách supervise_stage1.sh được quản lý:


Hai cổng kiểm chứng vẫn có giá trị — chúng kiểm phương pháp, không kiểm bản fine-tune. Nhưng không được trích chúng như
kết quả về model đã train.
vg.

Đúng phạm vi bạn chọn: luồng chính + lớp XAI. Không arm B, không Q-Former dạng hộp.

Điểm khác biệt lớn nhất so với hình cũ: hai nhánh xuất phát từ input và không bao giờ gặp nhau. Stage 2 đọc thẳng manifest, không import code Stage 1. Mũi tên "MHCAC → prompt" trong hình cũ không tồn tại trong code. MHCAC được vẽ với
cả hai đầu — classification q và mention gate_pos, thứ mà bootstrap đo là tốt hơn ở 14/14nhãn.

Q-Former nằm ở phần "Design notes" dưới cùng,át, bốn phép đo độc lập đều ra mức ngẫu nhiên, nguyên nhân là batch contrastive nhỏ hơn BLIP-2 khoảng 210 lần.

⚠ Tôi chưa xem được hình render — máy này khôairosvg. Tôi mới chỉ kiểm XML hợp lệ và ước

Xong, mọi thứ đã đúng chỗ.

Trạng thái

┌────────────────────────┬─────────────────────────────────────────────┐
│                        │

│ XAI trên model đã fine-tune     │ +30 phút    │
└─────────────────────────────────┴─────────────┘

Toàn bộ xong ~04/09 sáng — thay vì 6,2 ngày như lịch cũ. Bỏ arm B tiết kiệm đúng 70 giờ vốn sẽ cho ra một bản sao của arm A.

Một bài học đáng ghi

pgrep -f "bash /home/phuong/pipeline_rest.sh"ủa tôi, vì chuỗi đó nằm trong command line của

Cùng họ với lỗi grep ghim epoch: [0] hôm trước: cả hai đều là công cụ giám sát tự lừa mình. Tôi ghi vào CLAUDE.md để lần
sau không mất thời gian lại.

- Đầu ra của Stage 1 đi sang khối OUTPUT (đánh giá phân loại), không đi sang Stage 2.
- Thêm một dòng ghi chú dưới hình:
  "Stage 1 and Stage 2 are decoupled. Stage 2ectly

1. "Anchor image"  — read straight from the split manifest;
   no Stage-1 model, checkpoint, Q-Former or MHCAC
2. "MedGemma vision tower" — SigLIP, patch 14ed)

   AUROC · AUPRC · F1 · Precision · Recall · Specificity
2. "Generated report" — FINDINGS
   BLEU-1..4 · ROUGE-L · METEOR · CIDEr · BER
   dòng nhỏ màu đỏ nhạt: "Clinical metrics (CheXbert / RadGraph / RadCliQ) are
   not implemented — reported as unavailable, never as 0."
3. "Explanation maps" — one map per sentence,

=========== CHÚ GIẢI MÀU ===========
Cam = Stage 1 · Xanh dương = Stage 2 · Tím = XAI · Xanh lá = Output
(BỎ mục "Green = MHCAC predictions & guided p

=========== GIỮ NGUYÊN, ĐANG ĐÚNG ===========
Hộp INPUT: một STUDY chứ không phải một ảnh;  AP >
lateral); tối đa một auxiliary view, thường là lateral. Hai ảnh X-quang minh hoạ.

---


Cụm "MHCAC-Guided" là thứ phải bỏ, không phải thứ để sửa chữ. Nó nằm ở tiêu đề, ở tên Stage 2, ở chuỗi xanh lá và ở dòng
vàng cuối — bốn chỗ. Nếu chỉ sửa tiêu đề mà g

Bạn đã có bản chính xác rồi: docs/figures/pip55d4a) — vector, chữ thật, đúng phạm vi "gọn + có XAI" bạn chọn lúc nãy. Mở bằng xdg-open docs/figures/pipeline_official.svg. Nếu nó dùng được thì prompt trên chỉ còn là phương án dự phòng; nếu bố cục có chỗ lệch, bạn báo tôi sửa trực tiếp trong SVG — nhanh và chắc hơn là nhờ mô hìnhsinh ảnh vẽ lại hơn 60 nhãn chữ nhỏ.
```
positive_macro_f1  = 0.5000     # (1.0 + 0.0) / 2
positive_micro_f1  = 0.6667     # gộp tp=2, fp=0, fn=2 -> P=1.0, R=0.5
```

✅ **Đạt.** Hai giá trị khác nhau và khớp với tính tay. Macro cho pathology hiếm trọng số ngang
bằng; micro để study nhiều hơn chi phối.

### A3. AUPRC phản ánh mất cân bằng, AUROC thì không

200 mẫu, đúng 2 positive (prevalence 1%), một positive nằm giữa bảng xếp hạng.

```
macro_auroc = 0.7525
macro_auprc = 0.5100     (prevalence = 0.0100)
```

✅ **Đạt.** AUROC 0,75 nghe khả quan; AUPRC 0,51 cho thấy thực tế kém hơn nhiều. Với positive
hiếm, AUPRC là chỉ số cần đọc. AUPRC được implement theo **step-wise average precision**
(giống `sklearn.average_precision_score`), không dùng nội suy hình thang — hình thang lệch lạc
quan trên dữ liệu mất cân bằng.

### A4. Threshold calibrated khác 0.5 khi cần

6 study, toàn bộ score nằm **dưới** 0.5.

```
threshold 0.5   -> positive_macro_f1 = 0.0000
calibrated 0.350 -> positive_macro_f1 = 1.0000
```

✅ **Đạt.** Ở ngưỡng mặc định không có gì được dự đoán positive. Calibration tìm được cutoff
0,350 và tách hoàn hảo. Đây là trường hợp `argmax` 3 lớp của evaluator cũ **không thể** xử lý,
vì nó không có tham số nào để điều chỉnh.

### A5. Uncertain policy làm thay đổi metric đúng như kỳ vọng

4 study: 1 positive, 1 uncertain, 2 negative; cả positive lẫn uncertain đều có score 0,9.

| Policy | positive_macro_f1 | tp | fp | n_valid |
|---|---:|---:|---:|---:|
| `three_class` | 0.6667 | 1 | 1 | 4 |
| `uncertain_as_positive` | 1.0000 | 2 | 0 | 4 |
| `uncertain_as_negative` | 0.6667 | 1 | 1 | 4 |
| `ignore_uncertain` | 1.0000 | 1 | 0 | **3** |

✅ **Đạt.** Cả bốn policy cho kết quả đúng như định nghĩa:
- U-Ones biến mẫu uncertain thành true positive.
- U-Zeros biến nó thành false positive.
- U-Ignore **loại hẳn** mẫu đó (`n_valid` giảm từ 4 xuống 3) — đây là điểm phân biệt duy nhất
  giữa `ignore_uncertain` và `uncertain_as_negative` ở đây.
- `three_class` trùng `uncertain_as_negative` trên metric nhị phân, đúng như đã ghi trong
  docstring của `uncertain_policy.py`.

### A6. Metric legacy không đổi

`tests/test_stage1_eval_hook.py` chạy `ImageTextPretrainTask.evaluation` với model giả:

- P0 dự đoán hoàn hảo và có positive; P1 không có positive nào.
- `f1_positive_macro` (legacy) = **0.5000** — giữ nguyên hành vi cũ, kể cả khuyết điểm.
- `f1_positive_macro_defined_only` (mới) = **1.0000** — loại pathology không có positive.

✅ **Đạt.** Selection metric giữ nguyên bit-for-bit, nên checkpoint cũ và log cũ vẫn so sánh
được. Metric đã sửa được báo cáo song song dưới tên khác.

---

## Phần B — Report generation

### B1. Report trùng khớp hoàn toàn

```
BLEU-1 = 1.0000   BLEU-4 = 1.0000   ROUGE-L = 1.0000
```

✅ **Đạt.**

### B2. Sai negation nhưng lexical overlap vẫn cao ⚠️ **quan trọng nhất**

- Reference: `There is no pneumothorax.`
- Generated: `There is a pneumothorax.`

```
BLEU-1  = 0.8000
ROUGE-L = 0.8000
flags   = ['possible_false_positive_finding', 'possible_negation_error']
negation_mismatches = ['Pneumothorax']
```

✅ **Đạt, và đây là luận điểm trung tâm của toàn bộ tầng đánh giá.** Hai báo cáo **đối lập hoàn
toàn về lâm sàng** vẫn đạt BLEU-1 và ROUGE-L bằng 0,80. Không có ngưỡng nào trên metric lexical
phát hiện được lỗi này. Chỉ phân tích ở mức mệnh đề mới bắt được.

Hệ quả cần nhớ khi viết khóa luận: **một bảng chỉ có BLEU/ROUGE/METEOR/CIDEr/BERTScore không
chứng minh được tính đúng đắn lâm sàng.**

### B3. Report rỗng

```
bleu_4 = 0.0000   rouge_l = 0.0000   (không crash)
flags  = ['empty_output', 'possible_omitted_finding']
```

✅ **Đạt.** Điểm 0 ở đây là điểm 0 **thật** cho một output rỗng, khác hoàn toàn với điểm 0 do
exception mà `compute_nlg` cũ tạo ra.

### B4. Temporal hallucination

Generated: `The effusion is unchanged from the prior study.`

```
không có prior context -> flags = ['possible_temporal_hallucination']
                           temporal_phrases = ['prior study', 'unchanged']
có prior context       -> flags = []
```

✅ **Đạt.** Cờ bật khi input không có prior, và tự tắt khi context có prior. Tên cờ là
`possible_*` vì heuristic không thể chắc chắn.

### B5. Dependency thiếu KHÔNG trở thành điểm 0

Yêu cầu `bleu, meteor, cider, bertscore` trong môi trường không có nltk/pycocoevalcap/bert_score:

```
corpus keys      = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']
unavailable keys = ['bertscore', 'cider', 'meteor']
```

Kèm log:
```
metric 'meteor' needs the 'nltk' package, which is not installed.
Install it with: pip install nltk && python -m nltk.downloader wordnet.
The evaluator does not substitute a different implementation and does not report 0.0.
```

✅ **Đạt.** Ba metric thiếu **không xuất hiện** trong `corpus` (nên không thể lọt vào bảng kết
quả như số 0), mà nằm trong `unavailable` kèm câu lệnh cài đặt. Đây chính là lỗi S2-A trong
`evaluator_audit.md` đã được đóng.

---

## Phần C — Tính toàn vẹn của pipeline

| Kiểm tra | Kết quả |
|---|---|
| Chạy lại evaluator từ file prediction, không cần model/GPU/dataset | ✅ `test_full_stage1_pipeline_runs_without_a_model` |
| Calibration trên test bị từ chối (CLI trả exit code 2) | ✅ `test_calibration_cli_refuses_a_test_split` |
| Threshold fit trên test bị từ chối khi load (exit code 2) | ✅ `test_evaluator_refuses_thresholds_calibrated_on_test` |
| `metrics.json` không chứa token `NaN`/`Infinity`, parse được bằng `json.loads` | ✅ `test_metrics_json_is_parseable_and_has_no_nan_token` |
| Giá trị undefined ghi là `null` kèm lý do trong `skipped` | ✅ `auroc = null`, `skipped.Fracture = ["no_positive_samples"]` |
| Hai lần chạy cùng seed cho kết quả trùng khớp | ✅ `test_reevaluating_the_same_file_is_deterministic` |
| Text báo cáo không bị ghi ra trừ khi có `--include-text` | ✅ `test_full_stage2_pipeline_runs_without_a_model` |
| Sinh đủ PNG (ROC, PR, confusion ×N, bar ×2, histogram, reliability) | ✅ smoke run, 6 confusion + 6 plot |

---

## Những gì KHÔNG được kiểm chứng ở đây

- **Không có metric nào của mô hình thật.** Chưa chạy inference trên MIMIC-CXR; toàn bộ số trên
  là dữ liệu tổng hợp.
- **METEOR, CIDEr, BERTScore chưa từng chạy** trong môi trường này (thiếu package). Đường dẫn
  code của chúng chỉ được kiểm tra ở nhánh "dependency thiếu".
- **CheXbert / RadGraph / RadCliQ / RadFact chưa được implement.** `training/evaluation/clinical.py`
  raise `NotImplementedError` kể cả khi package đã cài — đây là chủ ý, xem mục F của báo cáo.
- **Các cờ `possible_*` chưa được bác sĩ xác nhận.** Chúng là heuristic từ vựng để sàng lọc.
- **Chưa chạy trên GPU.** Hook lưu prediction trong `image_text_pretrain.py` chỉ được test bằng
  model giả trên CPU.
