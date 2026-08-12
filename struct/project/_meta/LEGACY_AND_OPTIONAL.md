> Source: kết quả trace import/caller trên toàn repository
> Status: ✅ ACTIVE (bản thân trang này)
> Last verified against source: 2026-08-12

# Legacy, Optional & Unknown

Mọi thứ **không** nằm trong đường chạy production. Ba nhóm, và ranh giới giữa
chúng quan trọng:

| Nhóm | Nghĩa là gì |
|---|---|
| 🕰 **LEGACY** | Đã xác nhận không dùng nữa. Giữ trong tree, **không xóa**. |
| 🟡 **CONDITIONAL / OPTIONAL** | Có wire thật, chạy khi config bật. **Không phải** dead code. |
| ❓ **UNKNOWN** | Chưa có caller production, chưa xác nhận được ý định. |

> **Không có mục nào trong trang này bị xóa khỏi source.** Task documentation
> chỉ audit và ghi lại.

---

## 🕰 LEGACY — đã xác nhận

### L1 · MHCAC variants
> Quyết định: [D-003](DECISIONS.md#d-003--mhcac-variants-và-encoder-trùng-lặp-là-legacy)

| Path | LOC | Evidence | Thay bằng |
|---|---|---|---|
| `mhcac/mhcac.py` | ~150 | Zero reference toàn repo | `mhcac/mhcac_12.py` |
| `mhcac/mhcac_2.py` … `mhcac_7.py` | | Zero reference | `mhcac_12.py` |
| `mhcac/mhcac_8.py` (208) `mhcac_9.py` `mhcac_10.py` `mhcac_11.py` | | Zero reference | `mhcac_12.py` |

Cách kiểm chứng:
```bash
grep -rn "mhcac" --include='*.py' --include='*.yaml' . | grep -v '^./mhcac/'
# → chỉ mhcac.mhcac_12, mhcac.loss, mhcac.view_fusion
```

`mhcac_8..11` cũng bị loại khỏi ruff (`pyproject.toml`) — dấu hiệu chúng đã được
coi là legacy từ trước.

### L2 · `mhcac/utils.py`

| | |
|---|---|
| LOC | 624 |
| Caller duy nhất | `notebooks/03_train_meta_cxr_2xT4_kaggle.ipynb:1364` — `from mhcac.utils import compute_metrics_for_tasks, get_task_list` |
| Mà notebook đó | tự gắn nhãn LEGACY / DO NOT RUN (xem L6) |
| Thay bằng | `training/evaluation/classification_metrics.py` |

### L3 · `mhcac/aggregator.py`

| | |
|---|---|
| LOC | 63 |
| Evidence | **Không có `import` nào.** Chỉ còn hai dấu vết: chuỗi `"aggregator"` trong freeze-list `runner_base.py:189`, và một dòng đã comment `runner_base.py:1073`. |

Nghĩa là: `RunnerBase` vẫn *tìm* một attribute tên `aggregator` khi quyết định
param nào bị đóng băng, nhưng `Blip2Qformer` **không còn tạo** attribute đó nữa.
Danh sách freeze giờ trỏ vào một thứ không tồn tại. Đây là **tàn dư**, không phải
tính năng đang chạy.

⚠ **Potential issue** — nếu ai đó thêm lại `self.aggregator`, nó sẽ **tự động bị
đóng băng** mà không có thay đổi code nào ở chỗ khác. Đáng biết trước.

### L4 · `vision_encoders/biovil_t/` — bản sao

| | |
|---|---|
| Files | 8 (`encoder.py`, `model.py`, `modules.py`, `pretrained.py`, `resnet.py`, `transformer.py`, `types.py`, `__init__.py`) |
| Evidence | Zero external reference. Mọi import trong repo đều là `biovil_t.*` (top-level). |
| Bản được dùng | `biovil_t/` ở root |

```bash
grep -rn "vision_encoders.biovil_t" --include='*.py' . | grep -v '^./vision_encoders/biovil_t/'
# → rỗng
```

⚠ **Bẫy cho developer mới:** hai thư mục giống hệt nhau về chức năng. Sửa nhầm
bản trong `vision_encoders/` sẽ **không có tác dụng gì** và rất khó debug.
`vision_encoders/_index.md` phải nhắc điều này.

### L5 · `vision_encoders/medclip/`

| | |
|---|---|
| Evidence | `blip2_qformer.py:30` → `# from vision_encoders.medclip.medclip import Medclip` (comment) · `:286` → `# self.medclip = Medclip().eval()` (comment) |
| Ngoài ra | `medclip.py:3` import package `medclip` bên ngoài — không có trong requirements nào |

### L6 · Notebooks Kaggle / p10
> Quyết định: [D-004](DECISIONS.md#d-004--di-sản-kagglep10-chỉ-liệt-kê)

Cả ba **tự gắn nhãn** "LEGACY / DO NOT RUN" ở cell markdown đầu tiên. Outputs đã
sạch (`outputs present: False` — quan trọng, vì output notebook là đường rò rỉ dữ
liệu bệnh nhân dễ nhất).

| Notebook | Mục tiêu lịch sử | Thay bằng |
|---|---|---|
| `01_generate_mimic_cxr_cleaned_csv.ipynb` | Sinh `mimic_cxr_cleaned.csv` từ GCS | `preporcessing/preprocess_mimic_cxr.py` |
| `02_preprocess_mimic_cxr_p10_splits.ipynb` | Preprocess subset p10 | `preporcessing/preprocess_mimic_cxr.py` (p10–p19) |
| `03_train_meta_cxr_2xT4_kaggle.ipynb` | Train 2×T4 trên Kaggle | `cloud/run_stage1.sh` / `pretraining/train.py` |

Lý do đường Kaggle bị gỡ: MIMIC-CXR không được publish thành Kaggle Dataset theo
PhysioNet DUA. `configs/kaggle_datasets.yaml` mã hóa chính sách này thành
`policy.storage: private-gcs-only`.

⚠ Notebook 03 chứa code **vá `mhcac_12.py` bằng string replacement lúc runtime**
(`ensure_swin_mhcac_shape_patch`, dòng 843–874) để khớp shape Swin. Đây là dấu
vết của một lỗi shape đã có; hiện `mhcac_12._resize_patch_sequence` xử lý việc
này trong source.

### L7 · Config Kaggle / cũ

| Path | Evidence | Ghi chú |
|---|---|---|
| `configs/kaggle_datasets.yaml` | Chỉ notebook 01/03 + docs tham chiếu. Không code Python production nào đọc. | Giờ chủ yếu là **file chính sách** (`policy.storage: private-gcs-only`) |
| `pretraining/configs/blip2_pretrain_stage1.yaml` | Zero reference | |
| `pretraining/configs/mimic_cxr_2gpu.yaml` | Chỉ comment `pretraining/train.py:39` + notebook 03 | 2×T4; `multi_view: false`; ⚠ `warmup_steps: 32000` **không bao giờ hoàn tất ramp** — đừng copy giá trị này |

**Ngoại lệ quan trọng:** `blip2_pretrain_stage1_emb.yaml` **KHÔNG** legacy — nó là
config mà `inference.sh` dùng ([D-002](DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active)).

---

## ⚠ POTENTIALLY_UNUSED — `utils/`

Nhóm riêng vì có **mâu thuẫn giữa code và documentation**, và
[D-002](DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active) xác nhận
đường Vicuna là ACTIVE — nhưng `utils/` vẫn không có caller.

| Path | LOC | Evidence |
|---|---|---|
| `utils/prompter.py` | 51 | Zero import. `docs/stage2_prompt_audit.md:20` ghi "dùng bởi inference.py" — **grep không xác nhận**. Class `Prompter` đọc `data/templates/{name}.json`; file template thật nằm ở `model/lavis/data/templates/vicuna.json`, đường dẫn không khớp. |
| `utils/callbacks.py` | 75 | Zero import. Docstring ghi mượn từ text-generation-webui. |
| `utils/datacollator.py` | 107 | Zero import. `MyDataCollatorForSeq2Seq`. |
| `utils/split_emb.py` | 22 | **Chạy ngay khi import** — không có `if __name__` guard. Hardcode `pretraining/embs/2025_05_20_…pkl`; thư mục `pretraining/embs/` bị `.gitignore` và **không tồn tại** trong working tree. |

**Nguyên tắc áp dụng: code thắng documentation.** Documentation sẽ ghi trung thực
"không có caller", không suy diễn rằng `inference.py` dùng chúng.

⚠ **Potential issue** — `utils/split_emb.py` sẽ **thực thi và crash** nếu bị
`import` (chứ không phải chạy). Bất kỳ `from utils import *` nào cũng vỡ.

### `threshold.json`

Status: **⚠ POTENTIALLY_UNUSED**, chưa có user confirmation.

- `rg 'threshold.json' --glob '*.py'` không tìm thấy code nào hardcode path này.
- `inference.py:get_response` gọi `classify_abnormalities(logits)` mà không truyền
  `thresholds`, nên demo hiện tại dùng argmax.
- Stage 2 mặc định cũng dùng argmax; chỉ đọc file được chọn tường minh bằng
  `--threshold-path`.
- Comment `fig9:192-194` cảnh báo threshold lịch sử không có provenance
  checkpoint/validation và không được load ngầm.

File vẫn có thể được dùng thủ công từ bên ngoài repo, nên chưa gắn nhãn LEGACY
hoặc UNUSED cuối cùng.

---

## ❓ UNKNOWN — chưa phân loại
> Quyết định: [D-001](DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline)

Code chất lượng cao, có test đầy đủ, **nhưng chưa tìm được caller production**.
Không gắn nhãn LEGACY vì có thể là hạ tầng chuẩn bị.

### U1 · `training/trainer/`

| | |
|---|---|
| Files | `checkpointing.py` (`CheckpointManager`), `state.py` (`TrainingState`, `RngSnapshot`) |
| LOC | 178 |
| Caller | chỉ `tests/test_trainer_resume.py` |
| Trùng lặp với | Stage 2 dùng logic checkpoint riêng trong `training/stage2_utils.py`; Stage 1 dùng `runner_base._save_checkpoint` |

Nó làm được thứ hai bên kia không làm: **snapshot RNG state** (`RngSnapshot.capture`)
để resume tái lập được bit-for-bit. Đó là dấu hiệu của hạ tầng chuẩn bị chứ không
phải code chết.

### U2 · `safety/` — phần orchestration

| | |
|---|---|
| Files | `pipeline.py` (`SafetyPipeline`), `verifiers.py`, `reconciler.py` |
| LOC | 877 |
| Caller | chỉ `tests/test_safety_pipeline.py` |

**Ngoại lệ:** `safety/claims.py` (286 LOC) **có caller production** —
`training/evaluation/error_analysis.py:30`. File đó là ✅ ACTIVE, không thuộc U2.

`SafetyPipeline` cố ý không chứa logic verify, để cắm một model phrase-grounding
thật vào qua cùng protocol. Đây là interface chờ implementation, không phải
implementation bị bỏ.

### U3 · `training/evaluation/config.py`

| | |
|---|---|
| LOC | 238 |
| Caller | chỉ `tests/test_evaluation_config.py` |
| Vì sao chưa dùng | `scripts/evaluate_stage1.py` và `evaluate_stage2.py` nhận tham số qua `argparse`, không đọc block `evaluation:` trong config |

Nó validate được `evaluation.bootstrap.samples`, `evaluation.clinical_metrics`,
`evaluation.thresholds`… — tức là hình dung một luồng config-driven chưa được nối.

### U4 · `training/evaluation/counterfactual.py` + `perturbations.py`

| | |
|---|---|
| LOC | 268+ |
| Caller | chỉ `tests/test_counterfactual.py` |
| Chuỗi | `counterfactual` → `perturbations`; nhưng không script nào bắt đầu chuỗi này |

---

## 🟡 CONDITIONAL — có wire, tắt theo config

**Không phải legacy.** Đây là code chạy thật khi cờ tương ứng bật.

| Component | Bật khi | Mặc định hiện tại |
|---|---|---|
| `vision_encoders/rad_dino/` | `model.encoders.raddino: true` | `false` ở **mọi** config |
| `mhcac/view_fusion.py` | `model.multi_view: true` | `true` ở `mimic_cxr_full_l4.yaml`, `false` ở `mimic_cxr_2gpu.yaml` |
| `training/medgemma/soft_tokens.py` | `--pipeline-mode meta_cxr_qformer*` | mặc định là `medgemma_direct` (không dùng) |
| `training/stage1/lavis_loader.py` | mode cần Stage 1 | như trên |
| `training/evaluation/visualization.py` | `--plots` | tắt; cần extra `eval-plots` |
| `pretraining/precompute_features.py` | `run.feature_cache_dir` được đặt | không đặt |
| `training/evaluation/clinical.py` | không `--skip-clinical-metrics` | luôn báo unavailable |

> **Đừng gắn nhãn "unused" chỉ vì default đang `false`.** Ví dụ RadDINO có đầy đủ
> đường dữ liệu trong `blip2_qformer.py` (`:274`, `:530`, `:1353`) và
> `CANONICAL_STREAM_ORDER`. Bật một dòng config là nó chạy.

---

## ⛔ DISABLED có chủ đích

| Component | Cơ chế chặn |
|---|---|
| `model/pretrained_medgemma/impression_reporter.py` | Runtime guard. Module cố ý **trơ**: import nó không tải checkpoint, không dựng processor, không cấp VRAM, không import transformers. |
| `pipeline_modes.PRETRAINED_MEDGEMMA_IMPRESSION_PHASE2` | Khai báo nhưng nằm trong `EXTERNAL_INFERENCE_MODES` → `resolve_pipeline_modes` raise |
| `pipeline_modes.PRETRAINED_MEDGEMMA_FINDINGS_FIRST` | Cùng cơ chế — chỉ chạy qua `medgemma_inference/`, không qua CLI fine-tuning |

Đây là **thiết kế**, không phải hỏng. Lý do: Phase 2 Impression chưa được duyệt
ngân sách, và một run inference ngoại lai không được lọt vào đường fine-tuning.

---

## ⚠ Potential issues — ghi nhận, KHÔNG sửa

Task documentation không sửa code. Ghi lại để quyết định sau.

| # | Vấn đề | Vị trí |
|---|---|---|
| I1 | File `.pyc` được **git track** — build artifact Python 3.7/3.8 lọt vào repo dù `.gitignore` có `__pycache__/` | `biovil_t/__pycache__/` (11 file), `model/lavis/**/__pycache__/` (13 file) |
| I2 | Freeze-list trỏ tới attribute không tồn tại (`aggregator`) | `runner_base.py:189` |
| I3 | `utils/split_emb.py` thực thi lúc import, đường dẫn hardcode không tồn tại | `utils/split_emb.py:6` |
| I4 | `device_map={"": 0}` ghim cứng GPU 0 | `inference.py:312` |
| I5 | Documentation nói `selection_metric: f1_positive_macro`, config thực tế là `macro_auprc` | `CLAUDE.md`, `README.md` vs `mimic_cxr_full_l4.yaml:109` |
| I6 | Tên file `mimic_cxr_full_l4.yaml` nói L4, comment trong file nói "Verified on RTX 5060 Ti 16 GB" | `mimic_cxr_full_l4.yaml` |
| I7 | `cloud/run_stage2.sh` dùng alias deprecated `--image-mode` | `cloud/run_stage2.sh:34` |
| I8 | `docs/stage2_prompt_audit.md` nói `utils/prompter.py` được `inference.py` dùng — không có import | `docs/stage2_prompt_audit.md:20` |
| I9 | Typo tham số `num_commmon_tokens` (ba chữ m) trong API công khai của MHCAC | `mhcac/mhcac_12.py:208`, `blip2_qformer.py:347` |
| I10 | `pandas 3.0` Copy-on-Write: `df[col].fillna(x, inplace=True)` âm thầm không làm gì | `ReportDataset.py:897` (trong `CheXpertDataset` — **không** nằm trên đường MIMIC-CXR) |
| I11 | Demo gọi `demo.launch(share=True)`, tạo public Gradio share URL trong khi UI nhận ảnh X-quang credentialed | `inference.py:670` |
| I12 | Launcher paper asset gọi entrypoint không tồn tại trong repository | `cloud/run_paper_assets.sh:24` → thiếu `paper_assets.py` |
| I13 | Container demo chạy `--privileged`, mount writable toàn repo và cấp `--gpus all`; quyền rộng hơn nhu cầu UI | `run_container.sh:17-21` |
| I14 | Comment `pyproject.toml` nói Stage 1/2 pin torch/transformers xung đột, nhưng lock hiện tại là additive: Stage 2 include Stage 1 | `pyproject.toml:13-16` vs `requirements-stage2.txt:3` |

---

## `docs/` — biên bản, không phải spec

Khoảng 15 file trong `docs/` là **bản ghi tại một thời điểm**, không phải tài liệu
sống. Chúng mâu thuẫn nhau và mâu thuẫn với code hiện tại.

**Nên theo:** `docs/VM_TRAINING_FINAL.md`, `docs/STAGE2_PIPELINE_MODES.md`,
`docs/SETUP_GUIDE.md`, `docs/CHECKPOINT_WORKFLOW.md`, `docs/FEATURE_CACHE.md`,
`docs/notebook_privacy.md`, `docs/cloud/*`.

**Chỉ đọc như lịch sử:** `code_audit_latest.md`, `code_audit_second_pass.md`,
`evaluator_audit.md`, `evaluator_validation.md`, `final_branch_integration_audit.md`,
`final_merge_conflicts.md`, `final_merge_plan.md`, `findings_phase_a.md`,
`gpu_pilot_checklist.md`, `medgemma_real_runtime_smoke.md`, `migration_guide.md`,
`pending_medgemma_finetuning_teardown.md`, `refactor_hotspots.md`,
`round4_baseline.md`, `round5_baseline.md`, `test_report_second_pass.md`,
`stage2_temporal_target_audit.md`, `stage2_prompt_audit.md`.

Chúng dẫn nhiều đường đã chết (`evaluation/eval_final_200.py`, `outputs/paper_assets.py`,
bucket GCS đã xóa). Đừng trích dẫn chúng trong công việc mới nếu chưa kiểm lại với code.

---

## Cách thoát khỏi trang này

Một mục rời khỏi trang này khi:

| Từ | Sang | Điều kiện |
|---|---|---|
| ❓ UNKNOWN | ✅ ACTIVE | Có caller production thật → thêm decision, chuyển sang `ACTIVE_COMPONENTS.md` |
| ❓ UNKNOWN | 🕰 LEGACY | User xác nhận không dùng → cập nhật D-001 |
| 🕰 LEGACY | (xóa khỏi tree) | Chỉ khi user yêu cầu rõ ràng. Ghi decision trước khi xóa documentation. |
| 🟡 CONDITIONAL | ✅ ACTIVE | Config production bật nó lên |

---

← [Về HOME](../../HOME.md) · [ACTIVE_COMPONENTS.md](ACTIVE_COMPONENTS.md) · [DECISIONS.md](DECISIONS.md)
