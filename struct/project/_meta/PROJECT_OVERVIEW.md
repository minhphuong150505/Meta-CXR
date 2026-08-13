> Source: toàn bộ repository `Meta-CXR-source/`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# Meta-CXR — Tổng quan project

Trang này dành cho người **chưa biết gì** về Meta-CXR. Đọc hết trang này bạn sẽ
hiểu project giải quyết vấn đề gì và các mảnh ghép lớn nằm ở đâu. Chi tiết kỹ
thuật để dành cho [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Bài toán

Bác sĩ X-quang phải đọc ảnh chụp ngực và viết báo cáo bằng lời. Việc này tốn thời
gian và thiếu nhất quán giữa người đọc. Meta-CXR tự động hóa bước đó: **nhận ảnh
X-quang ngực, sinh ra phần FINDINGS (và tùy chế độ, cả IMPRESSION) của báo cáo.**

Điểm khác biệt của Meta-CXR so với "cho ảnh vào LLM rồi bảo nó viết báo cáo":
project **phân loại bất thường trước**, rồi dùng kết quả phân loại đó dẫn đường
cho mô hình ngôn ngữ. Đây là ý tưởng "abnormality-guided" trong bài báo gốc.

Repository kế thừa công trình META-CXR (IEEE Access, 2025) nhưng đã viết lại
đáng kể — xem mục 7.

---

## 2. Input và Output

### Input

Một **study**, không phải một ảnh.

Một study MIMIC-CXR là một lần chụp của một bệnh nhân, có thể gồm nhiều tấm ảnh ở
các góc khác nhau (PA nhìn từ sau ra trước, AP nhìn từ trước ra sau, LATERAL nhìn
nghiêng). Meta-CXR chọn một **anchor view** theo thứ tự ưu tiên `PA → AP → lateral`
và tối đa một **auxiliary view** làm bổ trợ.

```text
study minh họa
  ├── ảnh 1: view PA        → anchor      (đưa vào mô hình như tín hiệu chính)
  └── ảnh 2: view LATERAL   → auxiliary   (fuse vào anchor để bổ sung thông tin)
```

Ảnh được resize 512 rồi crop 448×448.

### Output

Hai thứ, tùy stage:

| Output | Hình dạng | Sinh ra ở đâu |
|---|---|---|
| Phân loại bất thường | 14 bệnh lý × 3 lớp {Positive, Negative, Uncertain} | Stage 1, khối MHCAC |
| Text báo cáo | Chuỗi FINDINGS (± IMPRESSION) | Stage 2, MedGemma |

14 bệnh lý là bộ nhãn CheXpert: `No Finding`, `Enlarged Cardiomediastinum`,
`Cardiomegaly`, `Lung Opacity`, `Lung Lesion`, `Edema`, `Consolidation`,
`Pneumonia`, `Atelectasis`, `Pneumothorax`, `Pleural Effusion`, `Pleural Other`,
`Fracture`, `Support Devices`.

Lớp **Uncertain** không phải lỗi thiết kế — báo cáo X-quang thật sự có rất nhiều
câu kiểu "không loại trừ khả năng viêm phổi". Ép về nhị phân sẽ làm mất thông tin
này. Xem [GLOSSARY.md](GLOSSARY.md#pnu).

---

## 3. Dữ liệu

**MIMIC-CXR-JPG**, subset p10–p19 (toàn bộ).

Môi trường train là **một host duy nhất**, `phuong@minhphuong`
([D-011](DECISIONS.md#d-011--máy-train-hiện-tại)) — máy cá nhân, **1× RTX 5060 Ti
16 GB**, verify qua SSH 2026-08-13. Dữ liệu nằm ở `/mnt/drive1tb` (NTFS, không
auto-mount). Không còn đường chạy cloud nào
([D-013](DECISIONS.md#d-013--gỡ-toàn-bộ-đường-chạy-cloud)).

| | |
|---|---|
| Nguồn ảnh | Bản mirror cục bộ trên `/mnt/drive1tb` của máy train |
| Nguồn text | Report `.txt` thô, parse lấy phần FINDINGS/IMPRESSION |
| Định dạng split | Ba file CSV: train / val / test |
| Ràng buộc pháp lý | PhysioNet credentialed access, DUA **cấm redistribute** |

### Ràng buộc dữ liệu — không thương lượng

MIMIC-CXR là dữ liệu bệnh nhân. Remote GitHub của repo này là **public**. Vì vậy:

- Không bao giờ commit ảnh, report text, split CSV, feature cache, prediction
  JSONL, credential hay model weight. `.gitignore` chặn rộng.
- **Notebook đã chạy là đường rò rỉ dễ nhất** — outputs của nó nhúng `subject_id`,
  `study_id` và nguyên văn report. `scripts/check_notebook_privacy.py` chạy như
  pre-commit hook; đừng bypass.
- `image_path` trong CSV là đường dẫn **tương đối** (`files/p1X/pXXXXXXXX/sYYYYYYY/<dicom>.jpg`),
  được nối với `mimic_cxr_jpg_root`. Đừng đổi thành đường dẫn tuyệt đối.
- Bản thân `struct/` cũng phải sạch: không viết ID bệnh nhân hay report text vào
  bất kỳ trang documentation nào (xem [D-012](DECISIONS.md#d-012--đưa-struct-vào-repository)).

### Đặc điểm dữ liệu ảnh hưởng tới thiết kế

Hai con số giải thích phần lớn các quyết định kỹ thuật trong repo:

1. **Một study có nhiều ảnh.** Nếu coi mỗi ảnh là một mẫu độc lập, cùng một báo
   cáo sẽ bị lặp lại nhiều lần và study nhiều view bị đánh trọng số cao hơn.
   → Repo dùng **study-level sampling**: một dòng cho một study.
2. **Một tỉ lệ đáng kể report không có tag FINDINGS**, và một tỉ lệ nhỏ dòng
   không có nhãn CheXpert. → Repo dùng **mask riêng cho từng loss**:
   `classification_mask` và `generation_mask`. Một dòng thiếu nhãn CheXpert vẫn
   train được phần sinh báo cáo, chỉ bị loại khỏi loss phân loại.

---

## 4. Hai Stage, và tại sao chúng tách rời

```text
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1 — học biểu diễn + phân loại                         │
│ pretraining/train.py                                        │
│ Đầu ra: checkpoint_best.pth                                 │
└─────────────────────────────────────────────────────────────┘
                            │
              (tùy chế độ — có thể bỏ qua hoàn toàn)
                            │
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2 — sinh báo cáo                                      │
│ training/run_medgemma_qlora.py                              │
│ Đầu ra: LoRA adapter + reports JSONL                        │
└─────────────────────────────────────────────────────────────┘
```

**Tách rời là ràng buộc thiết kế nặng nhất repository.** Chế độ mặc định của
Stage 2 (`medgemma_direct`) **không** dùng Stage 1 gì cả — nó dùng image tower
riêng của MedGemma. Điều đó cho phép so sánh công bằng: nếu đường Q-Former của
Stage 1 tốt hơn, ta biết chắc lợi ích đến từ Stage 1 chứ không phải từ việc
Stage 2 vô tình được nhận thêm thông tin.

Ràng buộc này được **enforce bằng test**, không phải bằng quy ước:
`tests/test_native_independence.py` kiểm tra rằng mọi import LAVIS/Stage-1 chỉ
nằm trong `training/stage1/lavis_loader.py` và không nơi nào khác.

### Stage 1 làm gì

1. Ảnh đi qua **3 encoder đóng băng** (BioViL-T, PubMedCLIP, SwinV2) — chúng
   không được train, chỉ trích xuất đặc trưng.
2. **View fusion** hợp nhất anchor với auxiliary view, ngay trên đầu ra thô của
   từng encoder.
3. Các luồng được chiếu về cùng chiều 1408 rồi nối lại thành một chuỗi token.
4. **MHCAC** đọc token đó và dự đoán 14×3 nhãn.
5. **Q-Former** nén chuỗi token thành 32 query token, học căn chỉnh ảnh↔text
   qua ba mục tiêu ITC / ITM / LM.

Điểm tinh tế: MHCAC có **hai nhánh**. Nhánh *student* chỉ nhìn ảnh — đây là thứ
chạy khi inference. Nhánh *teacher* được nhìn cả report text, nhưng **chỉ lúc
train**, và kiến thức của nó được chưng cất (distill) sang student. Text không
bao giờ rò vào đường inference.

### Stage 2 làm gì

Fine-tune `google/medgemma-1.5-4b-it` bằng QLoRA 4-bit để sinh báo cáo. Có nhiều
kiến trúc, chọn bằng `--pipeline-mode`:

| Mode | Cần Stage 1? | Đường thị giác |
|---|---|---|
| `medgemma_direct` *(mặc định)* | Không | Image tower + projector của chính MedGemma |
| `meta_cxr_qformer` | Có | 32 query token của Q-Former, đưa vào dạng soft token |
| `meta_cxr_qformer_with_mhcac_prompt` | Có | Soft token + text P/N/U có cấu trúc |
| `text_only_language_prior_ablation` | Không | Không có ảnh — đây là sàn so sánh |
| `both_for_ablation` | — | Chạy `medgemma_direct` rồi `meta_cxr_qformer` |

Chi tiết ở [PIPELINES.md](PIPELINES.md).

---

## 5. Đánh giá

Evaluator nằm ở `training/evaluation/`, chạy qua CLI trong `scripts/`. Nguyên tắc
quan trọng: **evaluator đọc file kết quả đã lưu, không cần GPU và không cần model.**
Đổi threshold hay đổi policy không được tốn thêm một GPU-hour nào.

| Stage | Input | Chỉ số |
|---|---|---|
| 1 | `.npz` logits | Precision/Recall/F1 positive macro, per-pathology, AUROC, AUPRC, bootstrap CI, confusion matrix 3 lớp |
| 2 | `.jsonl` reports | BLEU, ROUGE-L (tự implement); METEOR, CIDEr, BERTScore (package tùy chọn) |

Threshold **chỉ được calibrate trên validation**, rồi mới áp lên test. Test split
bị giữ hoàn toàn ngoài quá trình chọn checkpoint.

### Cảnh báo về chỉ số lâm sàng

CheXbert, RadGraph, RadCliQ, RadFact **cố ý không được cài đặt** như extras — đó
là research code sau license riêng, không phải pin tái lập được.
`training/evaluation/clinical.py` sẽ báo `unavailable` hoặc `NotImplementedError`.

**Một chỉ số lâm sàng thiếu được báo là "không có", không bao giờ báo là điểm 0.**
Và BLEU/ROUGE không được trình bày như độ chính xác lâm sàng — chúng đo trùng lặp
từ ngữ, không đo đúng/sai y khoa.

---

## 6. Công nghệ chính

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Framework | PyTorch | Stage 2 lock include Stage 1 rồi thêm QLoRA packages |
| Vision encoders | BioViL-T (1408), PubMedCLIP (768), SwinV2 | Đều đóng băng; RadDINO có code nhưng tắt |
| Alignment | BLIP-2 / Q-Former, fork từ Salesforce LAVIS | `model/lavis/` là fork đã sửa, không phải pip package |
| LLM Stage 2 | MedGemma 1.5 4B-it, QLoRA NF4 | Gated model, cần `HF_TOKEN` |
| LLM legacy | Vicuna-7B + LoRA | Đường demo Gradio, xem [D-002](DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active) |
| Logging | Weights & Biases | Chỉ rank 0 log; rank khác dùng mode disabled |

### Hai virtual environment, không phải một

`requirements-stage2.txt` include `requirements-stage1.txt` rồi thêm QLoRA
packages. Cloud setup khuyến nghị hai venv để cách ly workflow, dù pin hiện tại
không xung đột nhau. `pyproject.toml` **cố ý khai báo zero runtime dependency** vì
repo chưa phải package cài đặt và lock nằm theo workflow; file này chủ yếu cấu
hình tooling. Module import theo path từ repo root.

---

## 7. Trạng thái thực tế — đọc kỹ

> **Chưa có gì trong repo này được validate trên GPU ở commit hiện tại.**

Không có Stage-1/Stage-2 smoke test nào đã chạy, không có metric full-data nào
được tái lập. Đừng mô tả bất kỳ kết quả nào là "đã kiểm chứng" trừ khi chính
phiên làm việc của bạn có một GPU run tạo ra nó.

Các số trong bài báo META-CXR gốc là **tham chiếu lịch sử**, không phải kết quả
của commit này.

Những gì đã thay đổi so với công trình gốc (đã có trong code, chưa có bằng chứng
tốt hơn):

- Study-level sampling và multi-view fusion anchor/auxiliary.
- Đường Stage 2 MedGemma native, độc lập hoàn toàn với Stage 1.
- Prompt v2 có version, config hash và template hash.
- Bộ evaluator riêng cho classification, generation, error analysis, counterfactual.
- Workflow VM/preflight, config Stage 1 cho 1 GPU hoặc 2-GPU DDP.

---

## 8. Đi tiếp

| Bạn muốn hiểu | Đọc |
|---|---|
| Các khối ghép với nhau ra sao | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Có những pipeline nào, chạy thế nào | [PIPELINES.md](PIPELINES.md) |
| Dữ liệu biến đổi từ CSV tới tensor tới output | [DATA_FLOW.md](DATA_FLOW.md) |
| Gõ lệnh gì để chạy | [ENTRYPOINTS.md](ENTRYPOINTS.md) |
| Hàm nào gọi hàm nào | [CALL_GRAPH.md](CALL_GRAPH.md) |
| Cái gì còn dùng, cái gì đã chết | [ACTIVE_COMPONENTS.md](ACTIVE_COMPONENTS.md) · [LEGACY_AND_OPTIONAL.md](LEGACY_AND_OPTIONAL.md) |
| Thuật ngữ lạ | [GLOSSARY.md](GLOSSARY.md) |
| Tại sao lại quyết định như vậy | [DECISIONS.md](DECISIONS.md) |

← [Về HOME](../../HOME.md)
