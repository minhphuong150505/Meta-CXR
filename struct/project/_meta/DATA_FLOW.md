> Source: `preporcessing/`, `model/lavis/data/ReportDataset.py`, `blip2_qformer.py`, `training/dataio/manifest.py`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-14

# Data Flow

Dữ liệu đi từ đĩa tới output cuối, kèm shape ở từng chặng.

**Quy ước shape:** `B` = batch, `N` = số auxiliary view (ragged, pad về `N_max`),
`P` = số patch của một encoder, `D` = chiều embedding.
Shape lấy từ code/config. Chỗ nào phụ thuộc model đã load, trang này ghi
`⚠ cần runtime verification` thay vì đoán.

---

## 0. Toàn cảnh

```text
MIMIC-CXR raw (CSV.gz + report .txt + JPG)
        │  preporcessing/preprocess_mimic_cxr.py     [CPU, một lần]
        ▼
train.csv / val.csv / test.csv          ← patient- và study-disjoint
        │
        ├──────────────────────────┬───────────────────────────┐
        ▼                          ▼                           ▼
  MIMIC_CXR_Dataset          dataio/manifest.py          dataio/manifest.py
  (Stage 1, LAVIS)           (Stage 2 native)            (P8 external)
        │                          │                           │
        ▼                          ▼                           ▼
   Blip2Qformer              MedGemma QLoRA            MedGemma inference
        │                          │                           │
   logits + ckpt             adapter + JSONL              JSONL + budget
        │                          │                           │
        ▼                          ▼                           ▼
  evaluate_stage1.py         evaluate_stage2.py         evaluate_stage2.py

Stage-1 checkpoint + project split + private mask cache
        │  evaluate_explanation.py (model.eval, grad enabled)
        ▼
per-stream CAM 112² → metric JSON tách lung/bbox (+ private NPZ/PNG tùy chọn)
```

Ba đường đọc CSV **độc lập nhau**. `manifest.py` cố ý chỉ dùng pandas, không
import LAVIS — đó là cách ranh giới Stage 1 / Stage 2 được giữ.

---

## 1. Preprocessing → split CSV

`preporcessing/preprocess_mimic_cxr.py` + `preporcessing/mimic_report_parser.py`

```text
mimic-cxr-*.csv.gz        (metadata, CheXpert labels)
mimic-cxr-reports/files/  (227.835 file .txt)
        │
        ├── parse FINDINGS  ─► mimic_report_parser.py
        ├── parse IMPRESSION
        ├── join CheXpert labels
        ├── chia split (disjoint theo patient VÀ study)
        └── ghi CSV
```

Một tỉ lệ đáng kể report **không có tag FINDINGS**. Parser khôi phục phần thân
tường thuật thay vì âm thầm rơi về nguyên văn cả báo cáo — hành vi cũ đó tạo ra
target rác.

### Cột quan trọng trong CSV

| Cột | Kiểu | Ai đọc | Ý nghĩa |
|---|---|---|---|
| `image_path` | str **tương đối** | `_row_visual` | `files/p1X/pXXXXXXXX/sYYYYYYY/<dicom>.jpg` |
| `dicom_id` | str | dataset, feature cache | Khóa ảnh |
| `study_id`, `subject_id` | str | study index, leakage check | ⚠ không bao giờ ghi vào `struct/` |
| `ViewPosition` | str | `_view_id`, anchor selection | PA / AP / LATERAL / LL / UNKNOWN |
| `findings` | str | `text_output` | Target sinh báo cáo |
| `impression_clean` | str | Stage 2 native | ⚠ manifest cũ thiếu cột này |
| `target_valid` | bool | → `generation_mask` | FINDINGS dùng được không |
| `classification_valid` | bool | → `classification_mask` | Có nhãn CheXpert không |
| 14 cột CheXpert | float | `classification_labels` | `0` neg, `1` pos, `-1` uncertain |

**`image_path` phải giữ nguyên dạng tương đối.** `_row_visual` nối nó với
`vis_root`, và chủ động **raise** nếu gặp đường dẫn tuyệt đối hoặc path traversal
(`ReportDataset.py:637-644`) — đây là chốt bảo mật, không phải kiểm tra hình thức.

---

## 2. Stage 1 — CSV → tensor

### 2.1 Study index

`MIMIC_CXR_Dataset._init_study_index` + `model/lavis/data/mimic_cxr_utils.build_study_index`

Với `study_sampling: true`, nhiều dòng ảnh của cùng một study được gộp thành **một**
mẫu:

```text
CSV rows                      studies[]
────────────                  ─────────────────────────────
s5041, dicom_a, PA      ─┐    {anchor: idx_a,
s5041, dicom_b, LATERAL ─┴──► anchor_view_id: id(PA),
                               aux: [idx_b],
                               aux_view_ids: [id(LATERAL)]}
```

Anchor chọn theo `anchor_priority: [PA, AP, lateral]`. Tối đa `max_aux_views: 1`.

→ `__len__` trả về **số study**, không phải số ảnh. Một epoch là một lượt qua các
study duy nhất.

### 2.2 `__getitem__` → một sample

```python
{
  "image":                [3, 448, 448]          # hoặc "<enc>_feat" nếu dùng cache
  "text_output":          str                    # findings đã strip
  "image_id":             int
  "classification_labels":[14]        long       # 0 / 1 / -1
  "classification_mask":  []          bool
  "generation_mask":      []          bool
  "dicom_id":             str
  "image_path":           str
  # chỉ khi multi_view:
  "anchor_view_id":       int
  "aux_view_ids":         list[int]              # RAGGED
  "aux_image":            list[[3,448,448]]      # RAGGED
}
```

Ảnh: đọc → `remap_to_uint8` (percentile) → `general_trans` (resize 512 → crop 448).
Augmentation train (affine ±5°, translate 0.02, scale ±0.05, jitter brightness/
contrast 0.1) áp **một lần trong dataset**, để mọi encoder thấy **cùng một** ảnh
đã biến đổi.

### 2.3 `collater` → batch

Việc duy nhất của collater là **pad số auxiliary view ragged về `N_max` của batch**.
Khóa không liên quan được giao lại cho default collate nguyên vẹn — nên batch
`multi_view=False` giống hệt bản gốc từng byte.

```text
sample.aux_image  list dài n_i (khác nhau giữa các mẫu)
        │
        ▼  pad bằng torch.zeros_like(anchor)
batch["aux_image"]     [B, N_max, 3, 448, 448]
batch["aux_mask"]      [B, N_max]  bool   ← True ở vị trí view thật
batch["aux_view_ids"]  [B, N_max]  long   ← UNKNOWN_VIEW_ID ở chỗ pad
```

`N_max == 0` (không mẫu nào có aux) → tensor rỗng shape `[B, 0, …]`, không phải
`None`. Nhờ vậy code xuôi dòng không cần nhánh `if`.

### 2.4 Trong model

```text
image [B,3,448,448]                    aux_image [B,N,3,448,448]
      │                                        │ flatten(0,1) → [B*N,3,448,448]
      │                                        │ torch.no_grad()   ← chỉ FEATURE bị detach,
      │                                        │                     W_K/W_V của fusion VẪN có grad
      ▼                                        ▼
┌─ BioViL-T ────► [B,P₁,1408] ◄──── fuse ──── [B,N,P₁,1408] ─┐
├─ PubMedCLIP ──► [B,P₂,768]  ◄──── fuse ──── [B,N,P₂,768]   ┤  ViewFusionModule
└─ SwinV2 ──────► [B,P₃,Dₛ]   ◄──── fuse ──── [B,N,P₃,Dₛ]    ┘  (một cái / encoder)
                        │
                        │  ln_vision chỉ áp cho biovil (chuẩn hóa, KHÔNG phải chiếu chiều)
                        ▼
              SharedVisualTokenProjector
                        ▼
        SharedVisualTokens(
            tokens = [B, P₁+P₂+P₃, 1408],
            spans  = {biovil: slice(0,P₁), pubmedclip: slice(P₁,P₁+P₂), swin: …}
        )
                        │
        ┌───────────────┴────────────────┐
        ▼                                ▼
  MHCAC (mhcac_12)                  Q-Former
        │                                │
  per-encoder pos-enc                query_tokens [B,32,768]
  + resize về target_patch_count     cross-attn mỗi 2 block
        │                                │
  expert_tokens [B,14,768]          last_hidden_state [B,32,768]
  × 6 lớp cross-attention                │
        │                          ┌─────┴──────┬──────────┐
  14 × Linear                      ▼            ▼          ▼
        ▼                    vision_proj    itm_head    Qformer LM
  logits [B,14,3]            [B,32,256]     [·,2]      loss_lm
```

`P₁`, `P₂`, `P₃` phụ thuộc kiến trúc encoder và `image_size: 448`.
⚠ **Cần runtime verification** — không đoán ở đây.

`⚠ Cảnh báo shape đã biết:` MHCAC resize mỗi stream về `target_patch_count` chung
(`_resize_patch_sequence`, `mhcac_12.py:317`) bằng adaptive avg-pool 2-D khi số
patch là số chính phương, ngược lại nội suy tuyến tính 1-D. Đây là chỗ Swin từng
gây mismatch shape và bị vá thủ công trong notebook legacy.

### 2.5 Text đi đâu

```text
text_output (findings)
      │ tokenizer, max_length = max_txt_len (production: 256)
      ▼
text_tokens.input_ids [B, 256]
      │
      ├─► Qformer.bert ─► text_output.last_hidden_state [B,256,768]
      │        │                    │
      │        │                    ├─► text_proj → [B,256] → normalize → ITC
      │        │                    └─► MHCAC TEACHER  ⚠ CHỈ LÚC TRAIN
      │        │
      │        └─► ITM (hard negative), LM (labels = input_ids, pad → -100)
      │
      └─► KHÔNG BAO GIỜ tới nhánh student
```

Đây là ranh giới quan trọng nhất của Stage 1. Nhánh student
(`blip2_qformer.py:907`) gọi MHCAC với `text_embeddings=None`. Nhánh teacher
(`:925`) mới truyền text. Student là thứ duy nhất chạy ở inference.

### 2.6 Mask lan tới loss

```text
classification_valid (CSV) ─► classification_mask ─┬─► L_cls
                                                    └─┐
target_valid        (CSV) ─► generation_mask ──────┬──┴─► teacher_mask ─► L_teacher, L_distill
                                                    └─► L_itc, L_itm, L_lm
```

`ClassificationLoss` nhận `sample_mask`, nên dòng không có nhãn **đóng góp đúng 0**
— không phải đóng góp một giá trị nhỏ.

---

## 3. Stage 1 → đĩa

```text
<output_dir>/<run_name>/
├── checkpoint_best.pth              ← chọn theo macro_auprc trên VALIDATION
├── checkpoint_last.pth
├── checkpoint_<epoch>.pth           ← chỉ mỗi save_freq epoch (production: 5)
└── result/
    ├── val_predictions_epoch_best.npz    ← save_predictions: true
    └── test_predictions*.npz             ← chạy MỘT LẦN sau train, từ checkpoint_best
```

`.npz` chứa logits + label + mask → đủ để tính lại mọi chỉ số **không cần GPU**.

⚠ `.npz` là dẫn xuất từ dữ liệu bệnh nhân → `.gitignore` chặn. Không commit.

### 3.1 Stage 1 → XAI artifact

```text
checkpoint_best.pth + MIMIC_CXR_Dataset(val|test)
        │  project manifest chọn anchor/split
        ├─ classification_labels + explanation mask/source
        └─ image → encoder/view fusion → shared visual → MHCAC
                                      │ capture _last_cam_streams
                                      ▼
                      Logit Difference Squared → Grad-CAM
                                      ▼
                      native 14²/7² → bilinear 112² → [0,1]
                                      ▼
                explanation_metrics (lung/bbox KHÔNG gộp)
```

MS-CXR từng box đi qua chính helper Resize(512) → CenterCrop(448) → 112² của
mask builder. `split` của MS-CXR không vào data flow. JSON aggregate không chứa
identifier; NPZ/PNG tùy chọn vẫn là dẫn xuất bệnh nhân và phải ở private/ignored
storage.

---

## 4. Stage 2 — CSV → tensor

### 4.1 Đường native (`medgemma_direct`)

```text
split CSV ─► training/dataio/manifest.py::build_records()   [CHỈ pandas]
                     │  assert_no_leakage()
                     │  assert_columns()  ← fail nêu tên cột thiếu
                     ▼
            records: list[dict]
            {
              "image_path": str,
              "ref": str,              ← target theo --section-mode
              "views": [...],
              "pred_groups": None,     ← native không có Stage-1 labels
              "prior_available": bool,
            }
                     ▼
            stage2/prompts/PromptBuilder.build(context)
                     ▼
            PromptPart[] + prompt_version + config_hash + template_hash
                     ▼
            MedGemma processor (ảnh + text)
                     ▼
            image tower + projector của MedGemma
```

### 4.2 Đường Q-Former (`meta_cxr_qformer*`)

```text
MIMIC_CXR_Dataset (LAVIS)  ← qua training/stage1/lavis_loader.py
        ▼
Blip2Qformer từ checkpoint_best.pth
        ▼
Q-Former output [B, 32, 768]
        ▼
img_proj  ─► [B, 32, hidden_size_gemma]
        ▼
SoftTokenEmbeddingWrapper
        │  THAY THẾ embedding tại vị trí <qformer_soft_token>
        │  (KHÔNG cộng vào)
        ▼
Gemma decoder + LoRA
```

**Chỗ dễ sai nhất repo.** Nếu index theo hàng sai, loss vẫn giảm nhưng mỗi study
đang được mô tả bằng ảnh của study khác — hoàn toàn im lặng. Vì thế
`validate_soft_token_batch` kiểm tra shape theo từng hàng và fail-closed.

Soft token cũng vào `bad_words_ids` khi generate.

### 4.3 Section target

| `--section-mode` | native | Q-Former |
|---|---|---|
| `findings_only` | ✅ | ✅ |
| `impression_only` | ✅ | ❌ **lỗi và dừng** |
| `findings_and_impression` (mặc định) | ✅ | ❌ **lỗi và dừng** |

Lý do: `ReportDataset.text_output` không phát ra IMPRESSION. Code **báo lỗi**
thay vì âm thầm thay section này bằng section kia.

FINDINGS và IMPRESSION có **giới hạn độ dài riêng**, dẫn xuất từ train split, và
không bao giờ thay thế cho nhau.

---

## 5. Stage 2 → đĩa

```text
<output-dir>/
├── <mode>/
│   ├── adapter_model.safetensors    ← LoRA
│   ├── adapter_config.json
│   ├── trainer_state.pt             ← điều kiện để resume
│   └── img_proj*                    ← chỉ mode Q-Former
├── meta.json                        ← prompt version, config hash, template hash
└── generated_reports.jsonl          ← input cho evaluate_stage2.py
```

`resumable_adapter()` (`run_medgemma_qlora.py:180`) coi một thư mục là resume được
chỉ khi có **đủ** weight + projector + `adapter_config.json` + `trainer_state.pt`.

⚠ `.jsonl` chứa text báo cáo sinh ra → `.gitignore` chặn `*.jsonl`. Không commit.

---

## 6. Feature cache (tùy chọn)

```text
pretraining/precompute_features.py
        ▼
<feature_cache_dir>/
├── biovil/       ← raw patch, TRƯỚC ln_vision và trước projection
├── pubmedclip/   ← P × 768
└── swin/
```

Trong `_encode_image_streams`, `cached[name]` thay thế forward của encoder đóng
băng. **Các projection có thể train vẫn chạy** → training giống hệt.

⚠ Dựng cache với `study_sampling=false`, nếu không auxiliary view sẽ vắng mặt và
`_row_visual` raise `KeyError` nêu đúng tên DICOM.

---

## 7. Bảng tra shape nhanh

| Tensor | Shape | Nguồn |
|---|---|---|
| `image` | `[B, 3, 448, 448]` | `image_size: 448` trong YAML |
| `aux_image` | `[B, N_max, 3, 448, 448]` | collater |
| `aux_mask` | `[B, N_max]` bool | collater |
| BioViL-T out | `[B, P₁, 1408]` | `VISUAL_DIM = 1408` |
| PubMedCLIP out | `[B, P₂, 768]` | hardcode 768 tại `blip2_qformer.py:310` |
| SwinV2 out | `[B, P₃, swin.embed_dim]` | ⚠ runtime |
| `SharedVisualTokens.tokens` | `[B, ΣP, 1408]` | `SharedVisualTokenProjector` |
| `query_tokens` | `[B, 32, 768]` | `num_query_token: 32` |
| `image_features` (ITC) | `[B, 32, 256]` | `embed_dim=256` |
| `text_features` (ITC) | `[B, 256]` | |
| ITC image queue | `[1024, 32, 256]` fp16 | `itc_queue_size: 1024` |
| MHCAC expert tokens | `[B, 14, 768]` | `num_abnormalities=14, embed_dim=768` |
| MHCAC logits | `[B, 14, 3]` | `num_classes=3` |
| `classification_labels` | `[B, 14]` long | 0 / 1 / -1 |
| Soft token Stage 2 | `[B, 32, hidden_gemma]` | ⚠ runtime |

`P₁`, `P₂`, `P₃`, `hidden_gemma` **chưa được verify** — cần một GPU run để chốt.

---

← [Về HOME](../../HOME.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [CALL_GRAPH.md](CALL_GRAPH.md)
