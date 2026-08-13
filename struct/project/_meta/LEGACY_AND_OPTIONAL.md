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
| Caller | **Không còn**. Caller duy nhất từng là notebook 03, đã xóa 2026-08-13 (xem L6) |
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

### L6 · Notebooks Kaggle / p10 — 🗑 ĐÃ XÓA
> Quyết định: [D-004](DECISIONS.md#d-004--di-sản-kagglep10-đã-xóa) ·
> [D-013](DECISIONS.md#d-013--gỡ-toàn-bộ-đường-chạy-cloud)

Cả ba notebook đã bị xóa ngày 2026-08-13. Chúng từng tự gắn nhãn "LEGACY / DO NOT
RUN"; outputs đã sạch nên việc xóa không phát tán dữ liệu.

| Notebook (đã xóa) | Mục tiêu lịch sử | Thay bằng |
|---|---|---|
| `01_generate_mimic_cxr_cleaned_csv.ipynb` | Sinh `mimic_cxr_cleaned.csv` từ GCS | `preporcessing/preprocess_mimic_cxr.py` |
| `02_preprocess_mimic_cxr_p10_splits.ipynb` | Preprocess subset p10 | `preporcessing/preprocess_mimic_cxr.py` (p10–p19) |
| `03_train_meta_cxr_2xT4_kaggle.ipynb` | Train 2×T4 trên Kaggle | `pretraining/train.py` gọi trực tiếp |

Lý do đường Kaggle bị gỡ: MIMIC-CXR không được publish thành Kaggle Dataset theo
PhysioNet DUA. Lệnh cấm đó **vẫn còn hiệu lực** và giờ nằm ở `CLAUDE.md`
§Data handling, sau khi `configs/kaggle_datasets.yaml` bị xóa.

⚠ `scripts/check_notebook_privacy.py` và test của nó **được giữ nguyên** — hook
vẫn phải chặn bất kỳ notebook nào được thêm về sau.

📝 Notebook 03 từng chứa code vá `mhcac_12.py` bằng string replacement lúc runtime
để khớp shape Swin. Lỗi shape đó hiện đã được `mhcac_12._resize_patch_sequence`
xử lý trong source, nên không mất gì khi xóa notebook.

### L7 · Config legacy

| Path | Evidence | Ghi chú |
|---|---|---|
| `pretraining/configs/blip2_pretrain_stage1.yaml` | Zero reference | |

🗑 Đã xóa 2026-08-13: `configs/kaggle_datasets.yaml`,
`pretraining/configs/mimic_cxr_2gpu.yaml`, `pretraining/configs/mimic_cxr_2x3090.yaml`.

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
| `mhcac/view_fusion.py` | `model.multi_view: true` | `true` ở `mimic_cxr_full.yaml` |
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
| I5 | Documentation nói `selection_metric: f1_positive_macro`, config thực tế là `macro_auprc` | `CLAUDE.md`, `README.md` vs `mimic_cxr_full.yaml:109` |
| I6 | ✅ **Đã sửa 2026-08-13** — đổi tên `mimic_cxr_full_l4.yaml` → `mimic_cxr_full.yaml` | — |
| I7 | ✅ **Không còn** — `cloud/run_stage2.sh` đã bị xóa cùng `cloud/` | — |
| I8 | `docs/stage2_prompt_audit.md` nói `utils/prompter.py` được `inference.py` dùng — không có import | `docs/stage2_prompt_audit.md:20` |
| I9 | Typo tham số `num_commmon_tokens` (ba chữ m) trong API công khai của MHCAC | `mhcac/mhcac_12.py:208`, `blip2_qformer.py:347` |
| I10 | `pandas 3.0` Copy-on-Write: `df[col].fillna(x, inplace=True)` âm thầm không làm gì | `ReportDataset.py:897` (trong `CheXpertDataset` — **không** nằm trên đường MIMIC-CXR) |
| I11 | Demo gọi `demo.launch(share=True)`, tạo public Gradio share URL trong khi UI nhận ảnh X-quang credentialed | `inference.py:670` |
| I12 | ✅ **Không còn** — `cloud/run_paper_assets.sh` đã bị xóa cùng `cloud/` | — |
| I13 | Container demo chạy `--privileged`, mount writable toàn repo và cấp `--gpus all`; quyền rộng hơn nhu cầu UI | `run_container.sh:17-21` |
| I14 | Comment `pyproject.toml` nói Stage 1/2 pin torch/transformers xung đột, nhưng lock hiện tại là additive: Stage 2 include Stage 1 | `pyproject.toml:13-16` vs `requirements-stage2.txt:3` |

---

## `docs/` — biên bản, không phải spec

Khoảng 15 file trong `docs/` là **bản ghi tại một thời điểm**, không phải tài liệu
sống. Chúng mâu thuẫn nhau và mâu thuẫn với code hiện tại.

**Nên theo:** `docs/STAGE2_PIPELINE_MODES.md`, `docs/FEATURE_CACHE.md`,
`docs/notebook_privacy.md`.

🗑 Đã xóa 2026-08-13: `docs/cloud/*` (7 file), `docs/VM_TRAINING_FINAL.md`,
`docs/SETUP_GUIDE.md`, `docs/CHECKPOINT_WORKFLOW.md`, `docs/gpu_pilot_checklist.md`.

**Chỉ đọc như lịch sử:** `code_audit_latest.md`, `code_audit_second_pass.md`,
`evaluator_audit.md`, `evaluator_validation.md`, `final_branch_integration_audit.md`,
`final_merge_conflicts.md`, `final_merge_plan.md`, `findings_phase_a.md`,
`medgemma_real_runtime_smoke.md`, `migration_guide.md`,
`pending_medgemma_finetuning_teardown.md`, `refactor_hotspots.md`,
`round4_baseline.md`, `round5_baseline.md`, `test_report_second_pass.md`,
`stage2_temporal_target_audit.md`, `stage2_prompt_audit.md`.

Chúng dẫn nhiều đường đã chết (`evaluation/eval_final_200.py`, `outputs/paper_assets.py`,
bucket GCS đã xóa, và từ 2026-08-13 là cả `cloud/` lẫn `docs/cloud/`). Đừng trích
dẫn chúng trong công việc mới nếu chưa kiểm lại với code. Nội dung lịch sử của
chúng **cố ý không được sửa** khi gỡ đường cloud — sửa sẽ làm sai lệch bản ghi.

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
