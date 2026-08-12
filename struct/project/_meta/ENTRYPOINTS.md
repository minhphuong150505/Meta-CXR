> Source: mọi file có `if __name__ == "__main__"` + `cloud/*.sh` + `Dockerfile`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# Entrypoints

Bảng tra lệnh. Mọi mục ở đây được xác nhận bằng `if __name__ == "__main__"` hoặc
bằng nội dung shell script — không có mục nào là suy đoán.

Phân loại: `PRIMARY` (đường chính) · `UTILITY` (hỗ trợ) · `EXPERIMENTAL` ·
`LEGACY` · `LAUNCHER` (wrapper gọi entrypoint khác).

---

## Python — PRIMARY

### `pretraining/train.py`

| | |
|---|---|
| Lệnh | `python -m torch.distributed.run --standalone --nproc_per_node=1 -m pretraining.train --cfg-path <yaml>` |
| Pipeline | [P1](PIPELINES.md#p1--stage-1-pretraining) |
| Args | `--cfg-path` (bắt buộc), `--local_rank`, `--options k=v …` |
| Status | ✅ PRIMARY |
| Doc | [`pretraining/train.py.doc.md`](../pretraining/train.py.doc.md) |

Phải chạy qua `torch.distributed.run` kể cả với 1 GPU — `init_distributed_mode`
đọc `RANK`/`WORLD_SIZE`/`LOCAL_RANK` do torchrun đặt.

### `training/run_medgemma_qlora.py`

| | |
|---|---|
| Lệnh | `python training/run_medgemma_qlora.py [flags]` |
| Pipeline | [P2](PIPELINES.md#p2--stage-2-medgemma-qlora) |
| Args chính | `--pipeline-mode`, `--section-mode`, `--prompt-config`, `--checkpoint-root`, `--stage1-run`, `--stage1-config`, `--output-dir`, `--train-limit`/`--val-limit`/`--test-limit`, `--no-upload`, `--gcs-output` |
| Args cũ | `--image-mode {native,qformer,both}` (alias, vẫn chạy) |
| Status | ✅ PRIMARY |
| Doc | [`training/run_medgemma_qlora.py.doc.md`](../training/run_medgemma_qlora.py.doc.md) |

Chạy **trực tiếp bằng python**, không qua torchrun — Stage 2 là single-process.

### `medgemma_inference/run_pretrained_findings.py`

| | |
|---|---|
| Lệnh | `python -m medgemma_inference.run_pretrained_findings --config <yaml> --split validation [--max-samples N]` |
| Pipeline | [P8](PIPELINES.md#p8--external-medgemma-inference-baseline) |
| Status | ✅ PRIMARY (baseline) |
| Doc | [`medgemma_inference/run_pretrained_findings.py.doc.md`](../medgemma_inference/run_pretrained_findings.py.doc.md) |

Inference-only. **Không** gọi được qua `run_medgemma_qlora.py` — `resolve_pipeline_modes`
chủ động từ chối.

### `inference.py`

| | |
|---|---|
| Lệnh | `bash inference.sh` hoặc `python inference.py --cfg-path <yaml>` |
| Pipeline | [P9](PIPELINES.md#p9--gradio-demo-vicuna-7b) |
| Status | ✅ PRIMARY (demo) — kiến trúc legacy, xem [D-002](DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active) |
| Doc | [`inference.py.doc.md`](../inference.py.doc.md) |

Gradio ở `:7860`. Cần Java cho CheXpert labeler.

### `preporcessing/preprocess_mimic_cxr.py`

| | |
|---|---|
| Lệnh | `python preporcessing/preprocess_mimic_cxr.py --raw-dir … --reports-root … --output-dir …` |
| Pipeline | [P3](PIPELINES.md#p3--preprocessing--dựng-split) |
| Args thêm | `--views frontal`, `--limit-studies N` |
| Status | ✅ PRIMARY |
| Doc | [`preporcessing/preprocess_mimic_cxr.py.doc.md`](../preporcessing/preprocess_mimic_cxr.py.doc.md) |

CPU-only, không đọc ảnh.

---

## Python — UTILITY

| Entrypoint | Lệnh | Vai trò |
|---|---|---|
| `scripts/vm_preflight.py` | `python scripts/vm_preflight.py --stage 1` | Kiểm tra CUDA, RAM, disk, shm, path, HF auth **trước** mọi run dài. Không tải weight, không download. |
| `scripts/calibrate_thresholds.py` | `… --predictions <val.npz> --objective f1 --min-positive 20 --output <json>` | Calibrate threshold, **chỉ trên validation** |
| `scripts/evaluate_stage1.py` | `… --predictions <test.npz> --thresholds <json> --output-dir <dir>` | Chấm điểm classification |
| `scripts/evaluate_stage2.py` | `… --predictions <jsonl> --metrics bleu,rouge,… --output-dir <dir>` | Chấm điểm generation |
| `training/dataio/validate_manifest.py` | `python -m training.dataio.validate_manifest --section-mode findings_and_impression` | Kiểm tra leakage split, cột bắt buộc, section target |
| `scripts/check_notebook_privacy.py` | chạy như pre-commit hook | Chặn notebook mang dữ liệu MIMIC vào Git |
| `pretraining/precompute_features.py` | `python -m pretraining.precompute_features --cfg-path … --options …` | [P7](PIPELINES.md#p7--feature-precompute) |

---

## Python — EXPERIMENTAL / phân tích prompt

| Entrypoint | Vai trò | Cảnh báo |
|---|---|---|
| `scripts/run_prompt_ablation.py` | [P6](PIPELINES.md#p6--prompt-ablation-dry-run) — dry run, không load model | Không sinh metric model |
| `scripts/export_stage2_prompt_samples.py` | Render prompt ra JSONL để debug | ⚠ **Có chứa findings text** — `--output` phải ở nơi riêng tư |
| `scripts/prompt_length_statistics.py` | Thống kê độ dài prompt/target | Không có tokenizer MedGemma → fallback whitespace, đánh dấu `"approximate": true` |
| `scripts/audit_temporal_targets.py` | Đo ngôn ngữ so sánh thời gian trong target khi input không có prior | Bằng chứng để chọn `temporal_target_policy` |

---

## Python — không nên gọi trực tiếp

### `training/train_eval_figure9_llm_variants_200.py`

Có `if __name__ == "__main__"` và `--output-dir` riêng, nên **kỹ thuật** là chạy
được. Nhưng đây là **động cơ Stage 2**, được `run_medgemma_qlora.py:49` import.
Gọi thẳng nó bỏ qua toàn bộ phần resolve pipeline mode, kiểm tra leakage và ràng
buộc section ở entrypoint chính.

→ Dùng `training/run_medgemma_qlora.py`. Xem [D-006](DECISIONS.md#d-006--độ-sâu-documentation-cho-động-cơ-stage-2).

---

## Shell — LAUNCHER

Tất cả nằm trong `cloud/`. Không script nào hardcode project/bucket — identity
đến từ biến môi trường. Trước khi chạy:

```bash
source cloud/env.local.sh     # untracked, chứa GCP_PROJECT / GCS_BUCKET / GCS_DATA_BUCKET
```

| Script | Gọi cái gì | Status |
|---|---|---|
| `cloud/run_stage1.sh` | `python -m pretraining.train` + upload GCS | ✅ ACTIVE |
| `cloud/run_stage2.sh` | `python training/run_medgemma_qlora.py --image-mode $STAGE2_IMAGE_MODE` | ✅ ACTIVE ⚠ dùng alias cũ |
| `cloud/setup_vm.sh` | Cài dependency hệ thống trên VM | 🧰 UTILITY |
| `cloud/push_from_local.sh` | Đẩy code lên VM | 🧰 UTILITY |
| `cloud/run_encoder_comparison.sh` | Alias cũ → `run_stage1.sh`; không tự sweep | 🕰 COMPATIBILITY |
| `cloud/run_medgemma_pipeline.sh` | Alias cũ → `run_stage2.sh` | 🕰 COMPATIBILITY |
| `cloud/run_medgemma_l4_bucket_pipeline.sh` | Alias cũ → `run_stage2.sh` | 🕰 COMPATIBILITY |
| `cloud/run_medgemma_qformer_eval.sh` | Alias Q-Former → full `run_stage2.sh`; có thể train, không chỉ eval | 🕰 COMPATIBILITY |
| `cloud/run_paper_assets.sh` | Gọi `paper_assets.py` đang không tồn tại | ⚠ BROKEN / POTENTIALLY_UNUSED |
| `cloud/lib/common.sh` | Thư viện dùng chung: `log`, `require_gcp_config`, `require_private_bucket`, `upload_gcs` | 🧰 UTILITY |

`require_private_bucket` **từ chối** bucket không bật uniform bucket-level access
**và** public-access prevention. Đây là chốt chặn dữ liệu PhysioNet, không phải
kiểm tra hình thức.

### Root shell

| Script | Vai trò |
|---|---|
| `inference.sh` | Chạy `inference.py` với config demo |
| `build_container.sh` | Build Docker image |
| `run_container.sh` | Chạy container với GPU + Gradio `:7860` |

---

## Docker

```dockerfile
ENTRYPOINT ["/bin/bash", "inference.sh"]     # Dockerfile:5
```

→ Container build ra chạy **đường demo Vicuna** ([P9](PIPELINES.md#p9--gradio-demo-vicuna-7b)),
không phải Stage 1 hay Stage 2.

---

## Test — không phải entrypoint production

`tests/` có 12 file mang `if __name__ == "__main__"` (để chạy lẻ khi debug). Đường
chạy chuẩn là:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q
```

⚠ Trên máy CPU không có torchvision/transformers, 5 test fail và 1 file không
collect được — đây là trạng thái đã biết, không phải hỏng:

| Không chạy được | Lý do |
|---|---|
| `test_native_independence` (4 test) | import `model.lavis` |
| `test_stage1_eval_hook` (1 test) | import `model.lavis` |
| `test_blip2_negative_sampling` (cả file) | cần torchvision để collect |

---

## Điều kiện tiên quyết chung

Trước **mọi** entrypoint Python đọc dữ liệu:

```bash
cp configs/env_config.yaml.example configs/env_config.yaml   # rồi điền path
```

Thiếu file này → `local_config.py:6` raise `FileNotFoundError`.

Trước mọi run GPU dài:
```bash
python scripts/vm_preflight.py --stage 1
```

---

← [Về HOME](../../HOME.md) · [PIPELINES.md](PIPELINES.md) · [CALL_GRAPH.md](CALL_GRAPH.md)
