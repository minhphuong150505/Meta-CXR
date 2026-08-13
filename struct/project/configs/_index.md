> Source: `configs/`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `configs/`

## Purpose

Cấu hình **không thuộc về một Stage-1 run cụ thể**: đường dẫn của máy, prompt v2,
experiment inference ngoài, và các biến thể prompt ablation.

Siêu tham số Stage 1 nằm ở [`pretraining/configs/`](../pretraining/configs/_index.md).

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| Đường dẫn | Doc | Status | Vai trò |
|---|---|---|---|
| `env_config.yaml.example` | [📄](env_config.yaml.example.doc.md) | ✅ ★ | Mẫu; copy thành `env_config.yaml` |
| `env_config.yaml` | — | ✅ | **git-ignored** ở checkout này |
| `stage2_prompt_v2.yaml` | [📄](stage2_prompt_v2.yaml.doc.md) | ✅ | Prompt v2 — **opt-in** |
| `experiments/pretrained_medgemma_findings_first.yaml` | [📄](experiments/pretrained_medgemma_findings_first.yaml.doc.md) | ✅ | Baseline P8 |
| `prompt_ablation/P1..P9.yaml` | [📄](prompt_ablation/_index.md) | 🧪 | 9 biến thể prompt |
| `stage1_thresholds_f1_val.json` | [📄](stage1_thresholds_f1_val.json.doc.md) | ✅ | Threshold validation dùng cho Stage-1 Table 5 |

## `env_config.yaml` — bắt buộc trước mọi thứ

```bash
cp configs/env_config.yaml.example configs/env_config.yaml
```

Thiếu file này → `local_config.py:6` raise `FileNotFoundError` **lúc import**,
trước cả khi `main()` chạy.

Nội dung: `paths.data_root`, `paths.mimic_cxr_jpg_root` (⚠ phải trỏ vào thư mục
chứa **trực tiếp** `files/`), `paths.processed_{train,val,test}_csv`,
`paths.output_dir`, `java.home`, `java.path`, `wandb.entity`, `wandb.project`.

⚠ **Ở checkout này `env_config.yaml` được git-ignore** (khác với checkout
`META-CXR/` cũ, nơi nó bị track). Sửa `.example` cho thứ dùng chung.

## Chín biến thể prompt ablation

| | | | |
|---|---|---|---|
| P1 legacy style | P2 pos+unc, bỏ neg | P3 pos+unc+neg quan trọng | P4 thêm views |
| P5 visual primary | P6 qformer visual only | P7 confidence bins | P8 compact normal |
| P9 full negative control | | | |

Chạy bằng `scripts/run_prompt_ablation.py` — **dry run**, không load model.

## Main responsibilities

1. Tách đường dẫn của máy khỏi siêu tham số của run.
2. Cấu hình prompt có version.
3. Track threshold calibration có provenance cho evaluation tái lập.

## Entry points

Không phải entrypoint.

## Dependencies

`omegaconf` (env_config qua `local_config.py`) · `yaml` (prompt config)

## Used by

`local_config.py` · `training/run_medgemma_qlora.py:410` (`--prompt-config`) ·
`scripts/*` (prompt) · `medgemma_inference/config.py` (validate `experiments/*.yaml`)

## Status

```text
✅ ACTIVE
🧪 prompt_ablation/
```

## Notes

- ⚠ **`medgemma_inference/config.py` chỉ validate `configs/experiments/*.yaml`.**
  Config Stage 1 dưới `pretraining/configs/` là namespace riêng, không bị parse ở
  đó — `learning_rate`, `optimizer`, `warmup_steps` ở đó vẫn hợp lệ.

- **Prompt v2 là opt-in.** Không có `--prompt-config` thì code dùng prompt legacy.
  Điều này dễ bị bỏ sót khi so sánh kết quả.

- **Chính sách dữ liệu giờ nằm ở `CLAUDE.md`.** `kaggle_datasets.yaml` đã bị xóa
  ngày 2026-08-13 cùng toàn bộ đường chạy cloud. Lệnh cấm vẫn nguyên hiệu lực:
  MIMIC-CXR và mọi dẫn xuất không được publish thành Kaggle Dataset hay open data
  theo PhysioNet DUA.

- ⚠ **Không bao giờ commit token/credential** vào bất kỳ file nào ở đây.
  `.gitignore` chặn rộng nhưng không thay được sự cẩn thận.

## Related documentation

[ARCHITECTURE.md §6](../_meta/ARCHITECTURE.md#6-cấu-hình-phân-tầng) ·
[`pretraining/configs/_index.md`](../pretraining/configs/_index.md) ·
[`local_config.py.doc.md`](../local_config.py.doc.md)

← [Về HOME](../../HOME.md)
