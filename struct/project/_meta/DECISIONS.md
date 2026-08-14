# Project Decisions

Bộ nhớ bền vững của project. Mọi quyết định của user về component nào còn dùng,
component nào bỏ, pipeline nào là chính, và phạm vi documentation đều được ghi ở
đây — **không** giữ trong conversation memory.

Khi source code thay đổi khiến một decision không còn đúng, hãy thêm decision mới
ghi đè (kèm `Supersedes: D-00X`) thay vì sửa lịch sử.

| ID | Chủ đề | Status | Ngày |
|---|---|---|---|
| [D-001](#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline) | Hạ tầng chưa wire | ❓ Unknown | 2026-08-12 |
| [D-002](#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active) | Vicuna + Docker demo | ✅ Confirmed | 2026-08-12 |
| [D-003](#d-003--mhcac-variants-và-encoder-trùng-lặp-là-legacy) | MHCAC variants, encoder dup | ✅ Confirmed | 2026-08-12 |
| [D-004](#d-004--di-sản-kagglep10-đã-xóa) | Di sản Kaggle/p10 | 🗑 Đã xóa khỏi tree | 2026-08-13 |
| [D-005](#d-005--track-inference-checkpoint-ngoài-là-baseline-chính-thức) | External MedGemma baseline | ✅ Confirmed | 2026-08-12 |
| [D-006](#d-006--độ-sâu-documentation-cho-động-cơ-stage-2) | Độ sâu doc Stage 2 | ✅ Confirmed | 2026-08-12 |
| [D-007](#d-007--độ-sâu-documentation-cho-fork-lavis) | Độ sâu doc LAVIS | ✅ Confirmed | 2026-08-12 |
| [D-008](#d-008--struct-là-bộ-nhớ-local-không-commit) | struct/ từng git-ignored | ↪ Superseded by D-012 | 2026-08-12 |
| [D-009](#d-009--rule-đồng-bộ-đặt-ở-cả-hai-claudemd) | Vị trí agent rule | ✅ Confirmed | 2026-08-12 |
| [D-010](#d-010--tests-document-theo-nhóm-component) | Độ sâu doc tests | ✅ Confirmed | 2026-08-12 |
| [D-011](#d-011--máy-train-hiện-tại) | Host train `phuong@minhphuong` — RTX 5060 Ti 16 GB | ✅ Confirmed | 2026-08-13 |
| [D-012](#d-012--đưa-struct-vào-repository) | Track và push `struct/` | ✅ Confirmed | 2026-08-12 |
| [D-013](#d-013--gỡ-toàn-bộ-đường-chạy-cloud) | Gỡ toàn bộ đường chạy cloud | ✅ Confirmed | 2026-08-13 |
| [D-014](#d-014--mask-giải-thích-hai-tầng-và-split-project-là-nguồn-chân-lý) | Mask explanation hai tầng | ✅ Confirmed | 2026-08-13 |
| [D-015](#d-015--đánh-giá-xai-dùng-entrypoint-có-grad-riêng) | XAI evaluator có grad, metric NumPy tách source | ✅ Confirmed | 2026-08-14 |

---

## D-001 — Hạ tầng đã viết nhưng chưa nối vào pipeline

Status: **Unknown — chờ xác nhận**
Date: 2026-08-12

### Context

Bốn cụm code hoàn chỉnh, có test đầy đủ, nhưng không có caller nào trong đường
chạy production. Cần biết chúng là hạ tầng chuẩn bị, thiết kế bị bỏ dở, hay được
gọi thủ công từ ngoài repository.

### Evidence

| Path | LOC | Bằng chứng |
|---|---|---|
| `training/trainer/checkpointing.py` + `state.py` | 178 | `CheckpointManager` / `TrainingState` / `RngSnapshot` chỉ được import bởi `tests/test_trainer_resume.py`. Stage 2 dùng logic checkpoint riêng trong `training/stage2_utils.py`; Stage 1 dùng `model/lavis/runners/runner_base.py`. |
| `safety/pipeline.py`, `verifiers.py`, `reconciler.py` | 877 | Chỉ `tests/test_safety_pipeline.py` import. **Ngoại lệ:** `safety/claims.py` có caller thật tại `training/evaluation/error_analysis.py:30` → module đó là ACTIVE. |
| `training/evaluation/config.py` | 238 | Chỉ `tests/test_evaluation_config.py`. `scripts/evaluate_stage1.py` và `evaluate_stage2.py` dùng `argparse`, không đọc config block này. |
| `training/evaluation/counterfactual.py` + `perturbations.py` | 268+ | Chỉ `tests/test_counterfactual.py`. `perturbations` được `counterfactual` import, nhưng chuỗi này không có điểm bắt đầu nào trong `scripts/`. |

Đã kiểm tra và loại trừ: không có dynamic import, không có registry lookup, không
có tên module trong bất kỳ YAML nào, không có subprocess/shell script nào gọi tới.

### User decision

`[5] Chưa chắc` — chưa quyết định được tại thời điểm audit.

### Documentation impact

- Gắn nhãn `❓ UNKNOWN` (không phải `LEGACY`, không phải `ACTIVE`).
- Vẫn document đầy đủ ở mức file + method, vì code chất lượng cao và có thể được
  wire vào bất cứ lúc nào — documentation sai lệch sẽ tốn hơn là documentation dư.
- Mỗi `.doc.md` liên quan phải mở đầu bằng khối cảnh báo nêu rõ: *"Chưa xác định
  được caller production. Xem D-001."*
- Liệt kê trong `LEGACY_AND_OPTIONAL.md` ở mục riêng **"Chưa phân loại"**, tách
  khỏi mục Legacy đã xác nhận.

---

## D-002 — Đường Vicuna-7B legacy vẫn là demo ACTIVE

Status: **Confirmed**
Date: 2026-08-12

### Context

`README.md` và `CLAUDE.md` mô tả `inference.py` là "legacy, chưa migrate sang
MedGemma", gợi ý nó đã chết. Nhưng `Dockerfile` vẫn đặt `ENTRYPOINT` là
`inference.sh`, nghĩa là container build ra vẫn chạy đường Vicuna.

### Evidence

- `Dockerfile:5` → `ENTRYPOINT ["/bin/bash", "inference.sh"]`
- `inference.sh:7` → `python3 inference.py --cfg-path pretraining/configs/blip2_pretrain_stage1_emb.yaml`
- `inference.py` (670 LOC) load Vicuna-7B + LoRA adapter, dựng Gradio UI, import
  `model.lavis.models.blip2_models.modeling_llama_imgemb.LlamaForCausalLM`.
- `utils/prompter.py`, `utils/callbacks.py`, `utils/datacollator.py` (231 LOC):
  **zero import trong toàn repository**, kể cả từ `inference.py`.
  ⚠ `docs/stage2_prompt_audit.md:20` ghi "Legacy Vicuna JSON prompter (inference.py only)"
  — grep không xác nhận điều này. **Code thắng documentation.**
- `utils/split_emb.py`: chạy ngay khi import (không có `if __name__` guard),
  hardcode `pretraining/embs/…pkl` — đường dẫn này bị git-ignore và không tồn tại
  trong working tree.

### User decision

`[1] Active, vẫn demo Docker` — đường Vicuna vẫn được dùng để demo qua container.

### Documentation impact

- `inference.py`, `inference.sh`, `Dockerfile`, `build_container.sh`,
  `run_container.sh` → status `✅ ACTIVE`, document đầy đủ kèm `.methods/`.
- `pretraining/configs/blip2_pretrain_stage1_emb.yaml` → `✅ ACTIVE` (config duy
  nhất mà đường demo này dùng), tách khỏi nhóm legacy ở D-004.
- `utils/` → document riêng ở mức file, nhưng ghi trung thực trạng thái
  `⚠ POTENTIALLY_UNUSED`: không có caller nào trong code, mâu thuẫn với docs cũ.
  Không suy diễn rằng `inference.py` dùng chúng.
- `ENTRYPOINTS.md` phải liệt kê Docker/Gradio là entrypoint `PRIMARY` cho demo,
  song song với hai entrypoint training.

---

## D-003 — MHCAC variants và encoder trùng lặp là legacy

Status: **Confirmed**
Date: 2026-08-12

### Context

`mhcac/` có 13 file variant nhưng chỉ một file được wire. `vision_encoders/`
chứa một bản sao đầy đủ của `biovil_t/` và một encoder có import bị comment.

### Evidence

| Path | Bằng chứng |
|---|---|
| `mhcac/mhcac.py`, `mhcac_2.py` … `mhcac_11.py` (11 file) | Zero reference toàn repo. Chỉ `mhcac_12` được import: `model/lavis/models/blip2_models/blip2_qformer.py:23`. |
| `mhcac/utils.py` (624 LOC) | Caller duy nhất từng là notebook 03; notebook đó đã bị xóa 2026-08-13 (xem D-004), nên hiện **không còn caller nào**. |
| `mhcac/aggregator.py` (63 LOC) | Không có `import` nào. Chỉ xuất hiện dưới dạng **chuỗi ký tự** trong freeze-list `runner_base.py:189` (`for token in ("mhcac", "aggregator", "cls_loss_fn")`) và một dòng đã comment `runner_base.py:1073`. Tức là chỉ còn dấu vết trong logic freeze parameter, không phải module được dùng. |
| `vision_encoders/biovil_t/` (8 file) | Bản sao chức năng của `biovil_t/` top-level. Zero external reference — mọi import đều là `biovil_t.*`. |
| `vision_encoders/medclip/` (1 file) | Import bị comment: `blip2_qformer.py:30` (`# from vision_encoders.medclip.medclip import Medclip`) và `:286` (`# self.medclip = Medclip().eval()`). |

**Không thuộc nhóm này:** `vision_encoders/rad_dino/` có wire thật vào
`blip2_qformer.py` nhưng mọi config đặt `raddino: false` → đây là
`🟡 CONDITIONAL`, không phải legacy. Xem `LEGACY_AND_OPTIONAL.md`.

### User decision

`[3] Legacy — chỉ liệt kê`.

### Documentation impact

- Không tạo `.doc.md` riêng cho `mhcac_2..mhcac_11`, `mhcac.py`,
  `vision_encoders/biovil_t/`, `vision_encoders/medclip/`.
- Ghi toàn bộ vào `LEGACY_AND_OPTIONAL.md`: path, status, evidence, thứ thay thế.
- `mhcac/_index.md` phải nêu rõ chỉ `mhcac_12.py`, `loss.py`, `view_fusion.py` là
  active, và giải thích tại sao 11 file kia còn nằm trong tree.
- `vision_encoders/_index.md` phải cảnh báo về bản sao `biovil_t/` để developer
  mới không sửa nhầm file không được import.
- `mhcac/utils.py` và `mhcac/aggregator.py` ghi `🕰 LEGACY` kèm evidence chính xác
  (đặc biệt là chi tiết "aggregator chỉ còn là string trong freeze-list").

---

## D-004 — Di sản Kaggle/p10 đã xóa

Status: **Superseded by deletion**
Date: 2026-08-12 · cập nhật 2026-08-13

### Context

Repository từng chạy trên Kaggle với subset p10. Đường đó đã bị gỡ (MIMIC-CXR
không được publish lên Kaggle theo PhysioNet DUA), nhưng file vẫn còn trong tree.

**Cập nhật 2026-08-13:** user quyết định xóa hẳn, không giữ backup — training
chuyển về máy cá nhân để tối ưu chi phí (xem [D-013](#d-013--gỡ-toàn-bộ-đường-chạy-cloud)).
Toàn bộ file liệt kê dưới đây **không còn trong tree**; phần Evidence giữ lại làm
lịch sử.

### Evidence

- `notebooks/01_generate_mimic_cxr_cleaned_csv.ipynb`,
  `02_preprocess_mimic_cxr_p10_splits.ipynb`,
  `03_train_meta_cxr_2xT4_kaggle.ipynb` — cả ba **tự gắn nhãn** "LEGACY / DO NOT RUN"
  ngay cell markdown đầu tiên. Outputs đã sạch (kiểm tra: `outputs present: False`).
- `configs/kaggle_datasets.yaml` — chỉ được tham chiếu bởi notebook 01/03 và docs.
  Không code Python production nào đọc nó.
- `pretraining/configs/blip2_pretrain_stage1.yaml` — zero reference.
- `pretraining/configs/mimic_cxr_2gpu.yaml` — recipe 2×T4 cũ; `multi_view: false`,
  `warmup_steps: 32000` (không bao giờ hoàn tất ramp trong một run thực tế).
  Chỉ được tham chiếu từ comment `pretraining/train.py:39` và notebook 03.

**Ngoại lệ:** `blip2_pretrain_stage1_emb.yaml` KHÔNG legacy — nó là config mà
`inference.sh` dùng, xem D-002.

### User decision

`[1] Chỉ liệt kê`.

### Documentation impact

- Không tạo `struct/project/notebooks/`. Thư mục `notebooks/` không còn tồn tại.
- Thay thế: `preporcessing/preprocess_mimic_cxr.py` cho notebook 01/02;
  `pretraining/train.py` gọi trực tiếp cho notebook 03.
- `HOME.md` source tree đã bỏ node `notebooks/` và `cloud/`.
- `pretraining/configs/_index.md` giờ chỉ còn ba nhóm: production
  (`mimic_cxr_full` — recipe duy nhất), demo (`blip2_pretrain_stage1_emb`),
  legacy (`blip2_pretrain_stage1`).
- `scripts/check_notebook_privacy.py` và `tests/test_notebook_privacy.py` **được
  giữ nguyên**: hook vẫn phải chặn bất kỳ notebook nào được thêm về sau.

---

## D-005 — Track inference checkpoint ngoài là baseline chính thức

Status: **Confirmed**
Date: 2026-08-12

### Context

Repository có một pipeline hoàn chỉnh chạy inference trên checkpoint MedGemma do
bên thứ ba fine-tune, tách biệt hoàn toàn khỏi Stage 1 → Stage 2.

### Evidence

- `medgemma_inference/` (6 file, 939 LOC) — entrypoint
  `python -m medgemma_inference.run_pretrained_findings`.
- `model/pretrained_medgemma/` (6 file, 526 LOC) — loader, reporter, output schema.
- `runtime/budget.py` + `runtime/device.py` — chỉ track này dùng
  (`medgemma_inference/runner.py:23`, `model/pretrained_medgemma/findings_loader.py:24`).
- `configs/experiments/pretrained_medgemma_findings_first.yaml`.
- `tests/test_pretrained_findings.py` (553 dòng) — test rất kỹ.
- Checkpoint: `erjui/medgemma-4b-srrg-findings`, fine-tune từ `google/medgemma-4b-it`
  trên csrrg_ift (MIMIC-CXR + CheXpert+) bởi bên thứ ba, **không phải project này**.
- Code tự phòng vệ: `training/pipeline_modes.py:resolve_pipeline_modes` **từ chối**
  chạy mode này qua CLI fine-tuning; `model/pretrained_medgemma/impression_reporter.py`
  bị runtime guard vô hiệu hóa hoàn toàn (Phase 2 chưa được duyệt ngân sách).

### User decision

`[1] Baseline so sánh chính thức` — đây là số đối chứng sẽ được báo cáo.

### Documentation impact

- Document sâu ngang Stage 2: `.doc.md` đầy đủ + `.methods/` cho entrypoint,
  `runner.py`, `findings_loader.py`, `findings_reporter.py`, `prediction_writer.py`,
  `progress.py`, `budget.py`.
- `PIPELINES.md` liệt kê đây là pipeline độc lập (P8), **không** phải một
  `--pipeline-mode` của Stage 2 — nhấn mạnh nó không chạy qua fine-tuning CLI.
- Mọi documentation phải nêu rõ **provenance**: đây là checkpoint bên thứ ba, không
  train trên split của repo này. Đây là nghĩa vụ học thuật, không phải chi tiết kỹ thuật.
- Ghi rõ `pretrained_medgemma_impression_phase2` là DECLARED BUT DISABLED, không
  có implementation phía sau.

---

## D-006 — Độ sâu documentation cho động cơ Stage 2

Status: **Confirmed**
Date: 2026-08-12

### Context

`training/train_eval_figure9_llm_variants_200.py` (1.746 dòng) có tên gợi ý là
script vẽ figure một lần, nhưng thực tế là **động cơ Stage 2**: `run_medgemma_qlora.py:49`
import nó (`as fig9`) và class `VariantLLM` (~660 dòng, ~25 method) lo toàn bộ
model loading, QLoRA, prompt, collate, train loop, generate, NLG metrics, checkpoint.

Docs nội bộ (`docs/refactor_hotspots.md`, `docs/pending_medgemma_finetuning_teardown.md`)
ghi file này đang chờ được tách nhỏ.

### Evidence

- `wc -l` = 1746, lớn hơn file thứ hai (`inference.py`, 670) gần 3×.
- `training/run_medgemma_qlora.py:49` — import ở module scope.
- `docs/refactor_hotspots.md:11` — "Split — in progress".
- `docs/pending_medgemma_finetuning_teardown.md:31` — "must not be kept as a
  compatibility layer".

### User decision

`[2] Mức class + flow`.

### Documentation impact

- Tạo `train_eval_figure9_llm_variants_200.py.doc.md` đầy đủ + `.methods/VariantLLM/_index.md`
  mô tả vòng đời, trạng thái, I/O của class.
- `.methods/` chỉ cho các method có ML logic thực sự (~8 method), không phải cả 25.
- Trang nào cũng phải ghi: **tên file gây hiểu nhầm** — đây không phải script figure.
- Ghi nhận file đang chờ refactor, để lần cập nhật `struct/` sau biết là thay đổi
  lớn đã được dự báo.

---

## D-007 — Độ sâu documentation cho fork LAVIS

Status: **Confirmed**
Date: 2026-08-12

### Context

`model/lavis/` là fork Salesforce LAVIS đã sửa: 24 file Python, 10.862 LOC. Đây là
nơi model Stage 1 thực sự sống, nhưng phần lớn là code upstream.

### Evidence

Sáu file chiếm ~6.400 LOC và là nơi mọi sửa đổi của project nằm:

| File | LOC | Vai trò |
|---|---|---|
| `models/blip2_models/blip2_qformer.py` | 1.459 | Trung tâm Stage 1: encoder → fusion → MHCAC → Q-Former → tổng hợp 11 loss |
| `models/blip2_models/Qformer.py` | 1.221 | Q-Former (BERT + cross-attention) |
| `runners/runner_base.py` | 1.174 | Train loop, checkpoint selection, early stopping, freeze logic |
| `data/ReportDataset.py` | 1.130 | `MIMIC_CXR_Dataset`, study sampling, mask, collate |
| `models/blip2_models/modeling_llama_imgemb.py` | 975 | Llama có inject image embedding (đường Vicuna) |
| `tasks/image_text_pretrain.py` | 320 | Task hook, gọi evaluator Stage 1 |

### User decision

`[1] 6 file cốt lõi sâu`.

### Documentation impact

- Sáu file trên: `.doc.md` đầy đủ + `.methods/` cho method quan trọng.
- 18 file còn lại (`registry.py`, `optims.py`, `dist_utils.py`, `builders/`,
  `processors/`, …): chỉ được mô tả trong `model/lavis/_index.md` dạng bảng —
  vai trò một dòng, có sửa hay giữ nguyên upstream.
- `model/lavis/_index.md` phải nói rõ đây là fork đã sửa, **không phải** thư viện
  LAVIS cài qua pip, và bị loại khỏi ruff (reformat sẽ làm mọi diff upstream sau
  này không đọc được).

---

## D-008 — `struct/` là bộ nhớ local, không commit

Status: **Superseded by D-012**
Date: 2026-08-12

### Context

`.gitignore` có thay đổi chưa commit, dòng cuối là `struct/`.

### Evidence

```diff
 cloud/env.local.sh
+
+struct/
```
(`git diff .gitignore`, chưa commit tại thời điểm audit)

### User decision

`[1] Đúng — giữ local`. Knowledge base chỉ tồn tại trên máy này.

### Documentation impact

- **Không** sửa `.gitignore`.
- `struct/` không đi theo `git clone`; máy khác sẽ không có. `HOME.md` phải ghi
  rõ điều này để không ai tưởng documentation đã mất.
- Hệ quả tích cực: rủi ro rò rỉ dữ liệu PhysioNet qua documentation giảm mạnh.
  Dù vậy vẫn giữ nguyên nguyên tắc: **không viết `subject_id`, `study_id`,
  `dicom_id`, đường dẫn ảnh thật hay report text vào bất kỳ file nào trong `struct/`.**

---

## D-009 — Rule đồng bộ đặt ở cả hai CLAUDE.md

Status: **Confirmed**
Date: 2026-08-12

### Context

Có hai file agent instruction và chúng mâu thuẫn nhau:

| File | Git | Mô tả |
|---|---|---|
| `Meta-CXR-source/CLAUDE.md` | untracked | Checkout hiện tại; thắng cho work trong thư mục này |
| `../CLAUDE.md` | tracked (repo cha, không remote) | Mô tả checkout `META-CXR/` cũ — layout khác, remote khác, số test khác |

Không có `AGENTS.md` / `AGENT.md` trong checkout này.

### User decision

`[2] Cả hai CLAUDE.md`.

### Documentation impact

- Thêm section `## Source Documentation Synchronization` vào **cả hai** file.
- **Không overwrite** nội dung sẵn có của file nào — chỉ append.
- Trong `../CLAUDE.md` (repo cha), rule phải ghi rõ đường dẫn tuyệt đối tương đối
  là `Meta-CXR-source/struct/`, vì file cha mô tả nhiều checkout.
- Lưu ý: `../CLAUDE.md` được track trong git repo ngoài (không remote), nên thay
  đổi sẽ hiện trong `git status` của repo cha.

---

## D-010 — `tests/` document theo nhóm component

Status: **Confirmed**
Date: 2026-08-12

### Context

`tests/` có 30 file (~5.000 LOC) và là **thứ duy nhất** enforce các ràng buộc kiến
trúc của repo. Ví dụ `test_native_independence.py` chặn Stage 2 import LAVIS —
nếu test này biến mất, invariant quan trọng nhất repo không còn được bảo vệ.

### Evidence

- `tests/test_native_independence.py:41,161-163` — kiểm tra danh sách module cấm.
- `tests/test_inference_only_invariants.py` — chặn `model.train()` / optimizer
  trong đường inference-only.
- `tests/conftest.py` — đăng ký `model` / `model.lavis` là package path-only để
  suite chạy được trên máy CPU không có torchvision.

### User decision

`[1] _index.md nhóm theo component`.

### Documentation impact

- Một `struct/project/tests/_index.md` duy nhất, nhóm test theo vùng: Stage 1 /
  Stage 2 / prompt / evaluation / dataio / privacy / safety.
- Với mỗi nhóm: nêu **invariant nào đang được bảo vệ**, không chỉ liệt kê tên file.
- Không tạo `.doc.md` cho từng test file.
- Bù lại: **mọi** `.doc.md` của source phải có mục `## Tests` link ngược về nhóm
  tương ứng — đây là cách quan hệ test↔source được giữ.

---

## D-011 — Máy train hiện tại

Status: **Confirmed — hardware đã verify**
Date: 2026-08-12 · verify qua SSH 2026-08-13

### Context

Các recipe và tài liệu trong repository nhắc tới GCP L4, một GPU và 2× RTX 3090,
nhưng tên recipe không chứng minh môi trường đang được dùng ở thời điểm hiện tại.

### User decision

Project dùng **một máy train duy nhất**: `phuong@minhphuong`, máy cá nhân.

### Evidence (SSH, 2026-08-13)

| Thuộc tính | Giá trị |
|---|---|
| GPU | 1× NVIDIA RTX 5060 Ti, 16 GB (`nvidia-smi`) |
| Driver / CUDA | 580.173.02 / 13.0 |
| Disk `/` | 58 GB, còn ~5 GB |
| Disk `/home` | 185 GB, còn ~19 GB |
| Dữ liệu + checkpoint | `/mnt/drive1tb` — `nvme1n1p2`, 930 GB **NTFS**, **không có trong `/etc/fstab`** |
| Checkout | `~/Documents/2026/KLTN/Code_github/META-CXR-full-smoke-git` |
| sudo | có mật khẩu — agent qua SSH không mount được |

### Documentation impact

- Một GPU ⇒ **không còn đường DDP nào**. `--nproc_per_node` luôn là `1`.
- `/mnt/drive1tb` không auto-mount: sau reboot mọi path trong `env_config.yaml`
  đều treo. Kiểm tra mount trước khi debug lỗi thiếu file.
- Mọi thao tác chạy project đều qua SSH vào host này, và phải `git pull origin main`
  trước — checkout ở đó nhiều lần đã đi sau remote.
- Vẫn chạy `scripts/vm_preflight.py` trước một run dài.

---

## D-013 — Gỡ toàn bộ đường chạy cloud

Status: **Confirmed**
Date: 2026-08-13

### Context

Repository mang ba recipe phần cứng (GCP L4, Kaggle 2×T4, 2× RTX 3090), một thư
mục `cloud/` gồm 11 script GCP, 7 tài liệu `docs/cloud/`, và một file chính sách
Kaggle. Không cái nào mô tả môi trường thật: training đã hoàn tất trên máy cá
nhân (D-011).

### User decision

Xóa hết, **không cần backup** — lý do là tối ưu chi phí, training đã chuyển về
máy cá nhân.

### Đã xóa

`cloud/` · `docs/cloud/` · `notebooks/` (cả 3) · `configs/kaggle_datasets.yaml` ·
`pretraining/configs/mimic_cxr_2gpu.yaml` · `mimic_cxr_2x3090.yaml` ·
`docs/{SETUP_GUIDE,CHECKPOINT_WORKFLOW,VM_TRAINING_FINAL,gpu_pilot_checklist}.md`

Đổi tên: `mimic_cxr_full_l4.yaml` → `mimic_cxr_full.yaml`.

### Được giữ, có chủ ý

- `scripts/check_notebook_privacy.py` + test của nó — chữ "Kaggle" ở đó là **logic
  thật** (chặn false-positive cho ID 8 chữ số), không phải di sản.
- Lệnh cấm publish MIMIC-CXR lên Kaggle/open-data: chuyển từ
  `configs/kaggle_datasets.yaml` sang `CLAUDE.md` §Data handling. Ràng buộc
  PhysioNet DUA không mất đi khi file mã hóa nó bị xóa.
- Các `docs/*audit*.md`, `*_baseline.md`, `final_*.md` — biên bản lịch sử, có nhắc
  tới L4/Kaggle nhưng không phải hạ tầng. Sửa chúng sẽ làm sai lệch bản ghi.

### Documentation impact

- Không được tái tạo `cloud/`, `docs/cloud/`, đường upload GCS, hay config đặt tên
  theo phần cứng.
- Chốt chặn `require_private_bucket` mất theo `cloud/lib/common.sh`. Hiện không
  còn code nào đẩy dữ liệu ra khỏi máy train. Nếu thêm lại đường upload nào thì
  phải dựng lại chốt chặn tương đương **trước**.

---

## D-012 — Đưa `struct/` vào repository

Status: **Confirmed**
Date: 2026-08-12
Supersedes: **D-008**

### Context

D-008 từng giữ `struct/` local bằng rule cuối `.gitignore`. Sau khi review và
làm sạch knowledge base, user yêu cầu bỏ rule đó, commit và push lên GitHub.

### User decision

`struct/` phải được Git track và chia sẻ cùng source repository.

### Documentation impact

- Xóa `struct/` khỏi `.gitignore`; D-008 được giữ lại như lịch sử quyết định.
- `struct/` trở thành living documentation dùng được trên fresh clone.
- Privacy rule vẫn bắt buộc: không đưa patient ID, report text, đường dẫn dữ liệu
  thật, credential hoặc artifact mô hình vào tài liệu.
- Thay đổi source ảnh hưởng hành vi phải cập nhật `struct/` trong cùng commit.

---

## D-014 — Mask giải thích hai tầng và split project là nguồn chân lý

Status: **Confirmed**
Date: 2026-08-13

### Context

Explanation-aware loss cần vùng giám sát nhưng MIMIC-CXR không có bbox bác sĩ cho
toàn bộ tập. CheXmask phủ rộng bằng segmentation giải phẫu; MS-CXR có bbox bệnh
lý thật nhưng chỉ trên một tập con và mang split riêng không khớp split project.

### Decision

- MS-CXR bbox là tầng ưu tiên (`mask_source=1`).
- Nếu anchor không có bbox, dùng union phổi trái/phải CheXmask như anatomical
  prior (`mask_source=0`); không dùng heart.
- CheXmask dưới `Dice RCA (Mean) < 0.7` được xem là không có.
- Manifest train/val/test của project là nguồn chân lý duy nhất để định tuyến;
  cột split MS-CXR chỉ dùng để audit số dòng lệch.
- Rasterize ở kích thước gốc, rồi dùng đúng geometry ảnh train: resize cạnh ngắn
  512 → center crop 448 → nearest 112².
- Train affine được sample một lần cho ảnh/mask; ảnh bilinear, mask nearest.
- Cache + JSON index là dẫn xuất MIMIC-CXR, chỉ ở private storage và không log
  identifier.

### Evidence

- `preporcessing/build_explanation_masks.py` thực hiện chunked CheXmask join,
  bbox priority, Dice gate và split audit.
- `model/lavis/data/ReportDataset.py` mở mask memmap lazy theo worker PID và phát
  ba field explanation có điều kiện.
- `tests/test_explanation_mask_pipeline.py` ghim RLE, priority, quality gate,
  geometry, synchronized affine và no-cache behavior.

### Documentation impact

`preporcessing/`, `ReportDataset.py`, production config, test index và `HOME.md`
phải cùng mô tả nguồn mask, source code 0/1, geometry và privacy boundary.

---

## D-015 — Đánh giá XAI dùng entrypoint có grad riêng

Status: **Confirmed**
Date: 2026-08-14

### Context

`scripts/evaluate_stage1.py` cố ý model-free: nó chỉ đọc prediction `.npz` để
đổi threshold/policy không tốn GPU. Grad-CAM lại cần đạo hàm từ logit score tới
activation trong một graph autograd còn sống. Luồng evaluation LAVIS hiện có
cũng không thể cung cấp graph đó vì `RunnerBase.eval_epoch` mang
`@torch.no_grad()`.

### Decision

- Giữ nguyên `evaluate_stage1.py`; không thêm mask/checkpoint/model vào nó.
- `training/evaluation/explanation_metrics.py` chỉ dùng NumPy, để công thức
  Eq. (7)–(9) độc lập model và test được trên máy dev.
- `scripts/evaluate_explanation.py` là entrypoint riêng: lazy-load Stage-1,
  `model.eval()` nhưng mở grad, không optimizer/update, và dùng lại
  `mhcac.explanation.logit_difference_squared` + `grad_cam`.
- Báo cáo không có aggregate trộn source. `lung` (anatomical prior) và `bbox`
  (expert pathology annotation) luôn ở hai field riêng; annotation coverage của
  lung là unavailable, không phải 0.
- MS-CXR `split` không được đọc; manifest project quyết định split. Từng box đi
  qua geometry helper đã kiểm chứng của mask builder.
- PNG/NPZ là dữ liệu bệnh nhân: chỉ ghi private/ignored output, tên figure và
  stdout không mang identifier.

### Evidence

- Docstring `scripts/evaluate_stage1.py`: re-evaluation từ saved prediction là
  chủ ý trung tâm.
- `model/lavis/runners/runner_base.py::eval_epoch`: `@torch.no_grad()`.
- `tests/test_explanation_loss.py::test_explanation_loss_cannot_run_without_a_live_graph`
  ghim failure khi Grad-CAM chạy dưới no-grad.
- `tests/test_explanation_metrics.py` ghim công thức NumPy, per-box coverage,
  unavailable semantics và source separation.

### Documentation impact

`training/evaluation/`, `scripts/`, test index, `HOME.md`, `ENTRYPOINTS.md`,
`CALL_GRAPH.md`, `PIPELINES.md`, CLAUDE.md và README.md phải mô tả XAI như một
đường model-driven riêng, không làm suy yếu invariant evaluator offline.

---

## Ghi chú vận hành

Khi thêm decision mới:

1. Cấp ID kế tiếp (`D-015`, …), không tái sử dụng ID cũ.
2. Thêm một dòng vào bảng mục lục đầu file.
3. `Status` chỉ nhận: `Confirmed`, `Unknown — chờ xác nhận`, `Superseded by D-0XX`.
4. Mục `Evidence` phải là bằng chứng kiểm chứng được (đường dẫn + số dòng, kết quả
   grep), không phải suy đoán.
5. Mục `Documentation impact` phải nêu cụ thể file nào trong `struct/` bị ảnh hưởng.

## Ô trống CheXpert bị mask; chọn checkpoint theo val loss; manifest v2 (2026-08-14)

**Ô trống ≠ âm tính.** Export CheXpert để trống khi labeler không thấy nhắc tới
tổn thương, không phải khi bác sĩ loại trừ nó. 79,4% ma trận nhãn là ô trống, nên
việc gộp vào lớp 0 khiến khoảng 9/10 mẫu "âm tính" là *thiếu bằng chứng* chứ không
phải *bằng chứng về sự vắng mặt*. Nay chúng mang `IGNORE_LABEL = -100` và bị bỏ
theo từng ô.

Cái giá, đã đo trên `full_allviews_v2` (222.758 study train): chỉ còn 2,86/14 nhãn
mỗi study (31% study còn đúng 1 nhãn), mất cân bằng **lật chiều** — dương tính
thành đa số ở 12/14 nhãn — nên `default_class_weights` trong `blip2_qformer.py`
phải tính lại xuống 0,18–2,01. `No Finding` còn 0 mẫu âm (CheXpert chỉ đánh 1 hoặc
để trống) nên thành nhãn một lớp; nó đã bị `include_meta_labels: false` loại khỏi
macro metric.

**Chọn checkpoint theo `loss`, không phải `macro_auprc`.** Quyết định của người
dùng sau khi được cảnh báo. Ghi lại đánh đổi để không ai phải suy luận lại: val
loss là tổng có trọng số bị các nhãn phổ biến chi phối, nên một model **ngừng hẳn**
việc dự đoán nhãn hiếm có thể đạt điểm tốt hơn model đôi khi tìm ra nó.
`macro_auprc` không có lỗi này vì tính theo từng nhãn và không cần ngưỡng; nó bị
thay vì val quá mỏng cho hai nhãn (Pleural Other 14 dương, Fracture 16). Cả hai
vẫn được log mỗi epoch được chấm — đối chiếu xem hai tiêu chí chọn ra epoch nào
trước khi trích dẫn.

**Manifest: `processed/full_allviews_v2`.** `meta-cxr-manifests-upgraded-20260806`
là export cũ dù tên nói ngược lại. Ba dấu hiệu: `extraction_method` bằng
`legacy_preprocessed` cho 100% dòng (chuỗi không tồn tại ở đâu trong repo),
`target_valid` đúng 100% True (bộ lọc độ dài chưa từng chạy), và thiếu 4 cột —
`impression_valid`, `impression_token_count`, `findings_token_count`,
`target_filter_reason` — khiến Stage-2 mặc định `findings_and_impression` chết ở
`assert_columns`.

v2 đã đối chiếu ngược về nguồn ngày 2026-08-14: 377.110 dòng = **đúng toàn bộ**
metadata (hao hụt 0%), 0 trùng `dicom_id`, split khớp official MIMIC **0 lệch**,
ảnh tồn tại thật, `findings_clean` rỗng ⟺ `target_valid=False` **0 lệch** cả ba
split, và bound độ dài lấy từ train (val/test 0 dòng vượt trần).


## Tắt Swin, tăng num_workers, xoá toàn bộ checkpoint cũ (2026-08-14)

**`encoders.swin: false`.** Đây là đổi **kiến trúc**, không phải hyperparameter:
`SharedVisualTokenProjector` nhận hai stream thay vì ba và MHCAC thấy 98 visual
token thay vì 147, nên state dict không chuyển qua lại được giữa hai cấu hình.
Grad-CAM còn hai độ phân giải: BioViL 14×14, PubMedCLIP 7×7.
`explanation.streams` vẫn liệt kê `swin` — nó là bộ lọc trên các stream thực sự
được capture (`blip2_qformer.py`), nên tên không được sinh ra thì bị bỏ qua.

**`num_workers: 12`** (từ 4). Sau khi bỏ Swin và đưa resize CLIP lên GPU, bước
train đủ nhanh để bốn worker không theo kịp: chúng bám 96–98% CPU trạng thái `R`
trong khi GPU đứng chờ. Đo: 0.305 → 0.253 s/it. **Không phải nghẽn I/O** — epoch
nằm hết trong page cache chạy 0.2529 s/it, bằng epoch cache lạnh. Chi phí là giải
mã JPEG ở độ phân giải đầy đủ.

**Toàn bộ checkpoint trước 2026-08-14 đã bị xoá** theo yêu cầu người dùng (15
file, 39 GB) vì các run đó đi sai hướng. Chúng cũng đã không nạp được vào recipe
hiện tại do Swin tắt, và có trước manifest v2, việc mask ô trống, trọng số lớp
mới và tiêu chí chọn checkpoint mới. Số liệu Table 5 còn trong `results/` nhưng
**không tái lập được** vì checkpoint sinh ra chúng đã mất.

---

## D-016 — Mỗi encoder giữ thang đo riêng

**Ngày:** 2026-08-14
**Trạng thái:** ✅ Đã áp dụng, đã smoke trên GPU
**Liên quan:** [D-014](#d-014--mask-giải-thích-hai-tầng-và-split-project-là-nguồn-chân-lý)

### Bối cảnh

Lý do chạy hai encoder đóng băng là để model vừa nhìn cục bộ vừa nhìn toàn cục.
Đo trên ảnh thật của dataset cho thấy code **không** làm được điều đó — nó cắt
đúng phần tạo nên vai trò của mỗi bên:

| Đo được (ảnh thật, CPU, 2026-08-14) | Giá trị |
|---|---|
| cosine đôi một giữa 49 patch PubMedCLIP | **0.6755** |
| cosine đôi một giữa 196 patch BioViL | **0.0017** |
| tỉ lệ hướng chung đó là hằng số toàn dataset | **97%** (cos 0.970) |
| `cos(CLS PubMedCLIP, mean patch)` | **0.2109** |
| `cos(global BioViL, mean patch)` | **1.0000** |
| norm positional encoding vs norm token | **27.8 : 1.0** |

Ba hệ quả:

1. `_resize_patch_sequence` **xoá CLS** của PubMedCLIP (`50 == 49+1`). Đó là
   token duy nhất qua `post_layernorm` và là vector CLIP được huấn luyện
   contrastive — tức là tín hiệu toàn cục thật sự — và nó không tái tạo được từ
   phần giữ lại. Ngược lại `projected_global_embedding` của BioViL **chính là**
   trung bình các patch (`biovil_t/model.py:84`) nên bỏ đi không mất gì. Code
   đang vứt của PubMedCLIP thứ không tái tạo được và của BioViL thứ tái tạo được.
2. `cnn_downsampler` nén BioViL từ 14×14 xuống 7×7 **trước** MHCAC, bỏ đi đúng
   phần chi tiết là lý do tồn tại của encoder đó. Chỗ duy nhất còn dùng 14×14 là
   nhánh capture Grad-CAM.
3. Sau hai bước trên, hai encoder thành hai bản đồ 7×7 giống hệt nhau, dùng
   **chung một** positional encoding, nên không có gì trong đầu vào cho biết một
   token đến từ encoder nào.

### Quyết định

Mỗi encoder giữ nguyên chuỗi token gốc, mô tả bằng `StreamLayout`:

| Stream | Token | Lưới | Vai trò |
|---|---|---|---|
| `biovil` | 196 | 14×14, ô 32 px | cục bộ, chi tiết |
| `pubmedclip` | 1 + 49 | CLS + 7×7, ô 64 px | toàn cục + ngữ cảnh vùng |

Kèm ba thay đổi:

- `Pubmedclip.forward` áp `post_layernorm` rồi **trừ mean token của từng ảnh**
  khỏi 49 patch → cosine 0.674 → **−0.014**. CLS giữ nguyên. Sự phân công trở
  nên tường minh: token 0 = toàn cục, 49 patch = sai lệch cục bộ so với nó.
- Positional encoding **riêng cho mỗi encoder**, khởi tạo `std=0.02` thay vì
  `randn` trần (27.8 : 1 → 0.55 : 1).
- `model.image_size: 448` khai báo tường minh trong run YAML.

### Vì sao giữ PubMedCLIP ở 224 chứ không đẩy lên 448

Đã đo. Ở 448 với `interpolate_pos_encoding=True` PubMedCLIP cho 196 token khớp
BioViL, nhưng cosine thô *tệ hơn* (0.714 so với 0.674) và nó đẩy CLIP ra ngoài
phân bố huấn luyện trong khi encoder đóng băng nên không có cơ hội thích nghi.
Quan trọng hơn: hai bản đồ 14×14 là hai bản sao cùng thang đo. Ở 224 ta có **ba**
thang đo thật — 32 px, 64 px, toàn ảnh — đúng thứ hai encoder được dựng để cho.

### Chi phí

Chuỗi thị giác 98 → **246 token**. Đo trên host, batch 6, cùng manifest:
**0.2543 s/it so với 0.2529** — **+0.6%**. Rẻ vì `visual_proj` vốn đã chạy trên
246 token trước khi cắt; chỉ có cross-attention 14 query × 246 key dài ra, và
encoder đóng băng mới là phần chiếm thời gian.

### Hệ quả

- **Mọi checkpoint trước 2026-08-14 không load được.** `pos_enc` đổi từ một
  Parameter sang `ModuleDict`, và `cnn_downsampler` biến mất khỏi state dict.
- `loss_sparsity` là entropy trên số key, nên thang đo của nó đổi theo
  `ln(246)/ln(98) ≈ 1.20`. Đừng so trực tiếp giá trị term này với log cũ.
- Nhánh legacy (`stream_layouts=None`) vẫn giữ nguyên và được dùng khi
  `swin`/`raddino` bật, vì số token của chúng không suy ra được từ config.

### Đã bác bỏ

- **Đẩy PubMedCLIP lên 448** — xem trên.
- **Chỉ áp `post_layernorm`** — chỉ đưa cosine về 0.587, không giải quyết.
- **Để `Linear(768,1408)` của shared projector tự học trừ DC** — nó *có* bias
  nên về nguyên tắc làm được, và đây là lý do vấn đề này là "xuất phát điểm tệ"
  chứ không phải bug. Nhưng không có gì trong loss thúc nó về đó, và cái giá của
  việc đoán sai là cả stream vô dụng suốt 10 epoch.
