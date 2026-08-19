# Repo này khác gì so với bài báo gốc

So sánh với `DasithEdirisinghe/META-CXR` (bản code đi kèm bài báo IEEE Access,
09/2025), đối chiếu trực tiếp trên source ngày **2026-08-19**. Mọi khẳng định
dưới đây đều lấy từ file thật của cả hai bên, không lấy từ mô tả trong paper.

⚠ **Đọc kèm phần "Điều KHÔNG phải cải tiến" ở cuối.** Phần lớn thay đổi dưới đây
**chưa được xác thực end-to-end trên GPU**. Chúng đúng về mặt kỹ thuật và có test
CPU, nhưng "khác" không tự động nghĩa là "tốt hơn" cho tới khi có số đo.

## 0. Phạm vi repo gốc

Repo gốc có: `biovil_t/`, `mhcac/`, `model/lavis/`, `pretraining/`,
`vision_encoders/`, `utils/`, và `inference.py`. **Không có** `training/`,
`stage2/`, `scripts/`, `tests/`, `preporcessing/`, `safety/`, `runtime/`,
`docs/`, `configs/`. Tức là: có Stage 1 và một demo Gradio Vicuna; không có
pipeline tiền xử lý, không có code huấn luyện Stage 2, không có evaluator, không
có test nào.

| | gốc | ở đây |
|---|---|---|
| `mhcac/loss.py` | 401 dòng | **944** |
| `model/lavis/data/ReportDataset.py` | 780 | **1,640** |
| `model/lavis/runners/runner_base.py` | 791 | **1,310** |
| file test | **0** | 42 |
| CLI script | **0** | 14 |

## 1. Ngữ nghĩa nhãn — khác biệt lớn nhất, và là khác biệt khoa học chứ không phải kỹ thuật

**Ô CheXpert trống: gốc coi là ÂM TÍNH, ở đây coi là KHÔNG BIẾT.**

Gốc, `ReportDataset.py:253`:

```python
self.chexpert[column].fillna(0.0, inplace=True)  # fill with negative class for NaN values
```

Ở đây, ô trống mang `IGNORE_LABEL = -100` và bị loại từng ô khỏi loss.

Vì sao quan trọng: labeler để trống khi **không tìm thấy đề cập** đến bệnh lý
đó, chứ không phải khi bác sĩ loại trừ nó. **79.4% ma trận nhãn là ô trống**, nên
chính sách của bản gốc biến khoảng chín trên mười "âm tính" thành *thiếu bằng
chứng* được gán nhãn là *có bằng chứng phủ định*. Hệ quả đo được sau khi sửa:
chỉ còn 2.86/14 nhãn sống sót mỗi study, và **chiều mất cân bằng bị đảo** —
dương tính trở thành đa số ở 12/14 bệnh lý (Atelectasis: 44,718 dương / 1,502
âm). `default_class_weights` phải tính lại toàn bộ.

Kèm theo đó, ba thứ chỉ tồn tại ở repo này:

- **Mention gate** (`lambda_gate`): một head nhị phân trả lời *bệnh lý này có
  được nhắc đến không*. Đây là consumer duy nhất của 79.5% ô trống. Trước khi có
  nó, model bị ép chọn một trong ba lớp cho cả 14 bệnh lý trên mọi ảnh — đo được
  **10.8/14 bệnh lý bị gọi là Positive mỗi study**, và **cả 3,269 study gắn
  `No Finding = Positive` đồng thời có 8.8 phát hiện khác**, tức 100% mâu thuẫn nội tại.
- **Loss phân cấp mention-conditioned** (`lambda_mention_conditioned_cls`, mặc
  định tắt): gộp gate và classifier thành một likelihood thay vì hai head không
  bao giờ nói chuyện với nhau.
- **Chính sách `No Finding`**: gốc `fillna(0.0)` riêng cho cột này. Ở đây ghi
  nhận thẳng rằng nhãn này có 74,305 dương và **0 âm** trên toàn bộ dataset, nên
  head phân loại chỉ học được hằng số; nó bị loại khỏi macro metric qua
  `include_meta_labels: false` thay vì được báo cáo với F1 miễn phí 1.0000.

## 2. Teacher/student cho text đặc quyền

Gốc: MHCAC được gọi **một lần**, luôn kèm `text_embeddings` khi train
(`blip2_qformer.py:299`), và kèm `None` khi inference (`:581`). Hai layer đầu của
`mhcac_12` cross-attend vào text; cầu nối duy nhất giữa hai chế độ là
`text_dropout_rate`.

Ở đây: student (chỉ ảnh — đúng thứ inference chạy) và teacher (ảnh + text, chỉ
lúc train) là **hai đường tách bạch**, teacher được chưng cất vào student bằng
`soft_target_kl_loss` với teacher đã detach, và text attention bị chặn theo
layer qua `num_text_teacher_layers`. Nói cách khác: bản gốc train một model
text-conditioned rồi rút text ra lúc chạy thật; ở đây model chạy thật được huấn
luyện tường minh ở chế độ không-text và học từ model có text.

## 3. Multi-view

Gốc, `ReportDataset.py:244`: `metadata[metadata['ViewPosition'].isin(['PA','AP'])]`
— **chỉ giữ ảnh thẳng**, không có lateral, không có khái niệm study nhiều view.

Ở đây: sampling **một hàng mỗi study** thay vì mỗi ảnh (365,293 hàng ảnh →
220,216 study; trước đó report bị lặp 145,077 lần và study nhiều view bị đánh
trọng số quá cao). Anchor chọn theo `anchor_priority: [PA, AP, lateral]`, tối đa
một view phụ được fuse bằng `ViewFusionModule` **trước** projection, với `W_O` và
FFN cuối zero-init nên block là identity chính xác ở step 0. 121,738/220,216
study train có nhiều hơn một view. Kèm `view_consistency_loss` (dạng hinge có
gate độ tự tin) và `MultiPositiveContrastiveLoss`.

## 4. Token layout — mỗi encoder giữ nguyên thang đo của nó

Cả hai đều dùng BioViL-T + PubMedCLIP. Nhưng ở repo này, mỗi encoder giữ chuỗi
token gốc của nó (246 token) và có positional encoding riêng, thay vì bị ép về
cùng một lưới thô. Lý do đo được: CLS của PubMedCLIP **không** khôi phục được từ
các patch (`cos(CLS, mean patch) = 0.21`), trong khi token global của BioViL
*chính là* trung bình patch (`cos = 1.0000`) — nên việc bỏ CLS là vứt đi đúng cái
không thay thế được. Thêm nữa, patch của PubMedCLIP được đọc qua `post_layernorm`
rồi trừ trung bình theo ảnh: chưa làm vậy thì cosine trung bình giữa 49 patch là
**0.674** (BioViL: 0.0017), tức cả stream hoạt động như một bias hằng số.

## 5. Vòng huấn luyện

Repo gốc có `resume_ckpt_path` và lưu checkpoint mỗi epoch. Ở đây thêm:

- `save_every_iters` — ghi `checkpoint_last` giữa epoch (một epoch ~4 giờ; mất
  điện giữa chừng ở bản gốc là mất cả epoch);
- resume khôi phục optimizer, scheduler, epoch kế tiếp, best-loss counter,
  trạng thái generator của DataLoader, và RNG state của CPU/CUDA;
- `eval_start_epoch` — bỏ chấm các epoch đầu, và vì `checkpoint_best` chỉ được
  ghi trong nhánh eval nên nó cũng đảm bảo không epoch chưa chấm nào được chọn;
- early stopping đếm **epoch đã chấm**, cửa sổ mở tại `eval_start_epoch`;
- scheduler đếm `warmup_steps` theo **optimizer update**, không theo microbatch,
  và giữ nguyên tỉ lệ `lr_scale` của từng param group;
- `scripts/supervise_stage1.sh` — watchdog theo GPU utilisation và theo *lần ghi*
  checkpoint, fallback khi OOM, resume, và tính "không tiến triển" theo tiến độ
  chứ không theo số lần restart.

Một sửa lỗi đáng kể về hiệu năng, không liên quan model: `remap_to_uint8` chạy
min-max float64 trên ảnh 7 MP trước khi resize — chiếm **45.7%** thời gian xử lý
một study, nhiều hơn cả giải mã JPEG, và trên ảnh 8-bit thì nó là một bảng tra
256 phần tử mà ở dataset này là **ánh xạ đồng nhất**. Sửa xong: loader
0.6520 → 0.1722 s/batch, toàn vòng **0.6346 → 0.2347 s/it (2.70×)**, một run 10
epoch từ ~25 giờ còn **~9.1 giờ**.

## 6. Tiền xử lý và dữ liệu

Repo gốc không có script tiền xử lý; nó đọc thẳng CSV và thư mục report.

Ở đây có `preporcessing/preprocess_mimic_cxr.py` dựng split đầy đủ, và:

- **Parser FINDINGS**: 32.6% trong 227,835 report **không có tag FINDINGS**. Bản
  gốc `dropna(subset=['findings'])` — vứt bỏ. Ở đây parser lấy phần thân tự sự,
  và hàng không có FINDINGS dùng được vẫn train classification (chỉ bị mask khỏi
  generative loss) thay vì biến mất khỏi dataset.
- Split đã kiểm chứng tách biệt theo patient và theo study, đối chiếu khớp nguồn
  (377,110 hàng, không trùng `dicom_id`, khớp split chính thức của MIMIC).
- Có `impression_clean` / `impression_valid` riêng, với bound độ dài lấy từ train,
  và **không bao giờ thay thế section này bằng section kia**.
- `python -m training.dataio.validate_manifest` kiểm tra rò rỉ split và cột bắt buộc.

## 7. Đánh giá — phần lớn là mới, và có tính trung thực

Repo gốc không có script đánh giá nào; `threshold.json` nằm ở root như một
artifact rời.

Ở đây: `scripts/calibrate_thresholds.py` (**chỉ trên validation**),
`scripts/evaluate_stage1.py`, `scripts/evaluate_stage2.py`,
`scripts/evaluate_explanation.py`, bootstrap CI, error analysis. Test split bị
giữ ngoài quá trình chọn checkpoint và chỉ được chấm một lần từ `checkpoint_best`.

Đáng chú ý hơn là những gì được ghi lại như **giới hạn** thay vì được báo cáo như
kết quả: `positive_macro_f1` không dùng được làm headline trên phân bố nhãn này
(sáu nhãn đạt recall đúng 1.0000 với precision bằng prevalence, tức đoán dương
tính cho tất cả; `Fracture` đạt F1 0.983 trên AUROC **0.442**); `Pleural Other`
có 63 dương / **0 âm** trên test; metric lâm sàng (CheXbert, RadGraph, RadCliQ,
RadFact) **báo là không khả dụng chứ không bao giờ báo là 0**, và metric từ vựng
không được trình bày như độ chính xác lâm sàng.

## 8. Stage 2

Không so sánh được trực tiếp: repo gốc **không có code huấn luyện Stage 2** (chỉ
có `inference.py` demo Vicuna + `utils/prompter.py`). Ở đây Stage 2 là một hệ
thống riêng: MedGemma 1.5 4B-it QLoRA, `training/pipeline_modes.py` với các kiến
trúc được đặt tên tường minh (`medgemma_direct`, `meta_cxr_qformer`,
`meta_cxr_qformer_with_mhcac_prompt`, `text_only_language_prior_ablation`), Prompt
v2 (`stage2/prompts/`) dùng chung một `PromptBuilder` cho train và inference nên
parity kiểm tra được từng byte, và một **bất biến độc lập** được test cưỡng chế:
mọi import LAVIS/Stage-1 chỉ nằm trong `training/stage1/lavis_loader.py`, để
baseline native không bao giờ vô tình nhiễm Stage 1.

## 9. Thứ hoàn toàn không có ở bản gốc

- **`safety/`** — pipeline draft → parse claim → verify → báo cáo hoặc từ chối,
  với `parse_coverage` lộ ra ngoài có chủ đích (một pipeline chỉ parse được 2/12
  câu thì chưa kiểm tra gì, dù số liệu trông sạch).
- **`runtime/budget.py`** — tính chi phí theo wall-clock, mang
  `prior_elapsed_seconds` để resume không reset trần.
- **Explanation-aware loss + XAI evaluator** (CheXmask lung + MS-CXR bbox
  Grad-CAM). Đã chạy A/B có kiểm soát 5 epoch và **bị tắt** vì không giúp
  classification, kèm ghi chú rằng saliency precision 0.25 chính là mức ngẫu
  nhiên và không bao giờ được trích dẫn thiếu baseline diện tích mask.
- **42 file test CPU** cưỡng chế các bất biến kiến trúc: tách teacher/student,
  mask loss, đuôi accumulation, scheduler giữ `lr_scale`, phát hiện rò rỉ split,
  stream layout, view fusion, mention gate.
- **`struct/`** — knowledge base source-code được commit cùng thay đổi hành vi.

## Điều KHÔNG phải cải tiến — đọc phần này trước khi trích dẫn phần trên

- **Chưa có gì trong pipeline Stage-1/Stage-2 hiện tại được xác thực end-to-end
  trên GPU.** Bằng chứng GPU duy nhất được track là Table 5 — ablation encoder
  **chỉ inference** — và nó không xác thực training.
- **Khối vision-language (ITC/ITM/LM) tắt ở cả hai repo.** Ở bản gốc nó bị comment
  out; ở đây nó từng được bật lại rồi tắt ngày 2026-08-19 sau khi
  `scripts/check_itc_gate.py` cho thấy ITC nằm đúng mức ngẫu nhiên ở mọi nhánh.
  Hệ quả giống nhau ở cả hai bên: **cross-attention của Q-Former không được huấn
  luyện trên ảnh y khoa**, nên checkpoint Stage-1 không dùng được cho các mode
  soft-token của Stage 2.
- Bảng kappa theo bệnh lý trong `ClassificationLoss` là **đề xuất**, cần bác sĩ
  ký duyệt trước khi công bố; kappa không tạo ra kỹ năng, nó chỉ chọn điểm vận
  hành trên đường AUROC đã cố định.
- Loss phân cấp mention-conditioned mới chỉ **smoke trên GPU** (600 study, 3
  epoch) — đủ để nói nó train được, quá nhỏ để nói nó tốt hơn.
- Repo này **bỏ hẳn** đường cloud/Kaggle mà lịch sử git từng có. Đó là thu hẹp
  phạm vi có chủ đích, không phải cải tiến.
