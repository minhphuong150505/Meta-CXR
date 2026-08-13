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

## Ghi chú vận hành

Khi thêm decision mới:

1. Cấp ID kế tiếp (`D-014`, …), không tái sử dụng ID cũ.
2. Thêm một dòng vào bảng mục lục đầu file.
3. `Status` chỉ nhận: `Confirmed`, `Unknown — chờ xác nhận`, `Superseded by D-0XX`.
4. Mục `Evidence` phải là bằng chứng kiểm chứng được (đường dẫn + số dòng, kết quả
   grep), không phải suy đoán.
5. Mục `Documentation impact` phải nêu cụ thể file nào trong `struct/` bị ảnh hưởng.
