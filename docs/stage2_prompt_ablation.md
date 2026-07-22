# Stage-2 Prompt Ablation

Prompts are chosen on the **validation** split only. The test split is never used to
pick a variant. `scripts/run_prompt_ablation.py` runs a **dry run** by default (render
only, no generation, no metrics); a real comparison requires the GPU runtime.

## Variants (`configs/prompt_ablation/`)
| ID | File | What it isolates |
|---|---|---|
| P1 | `P1_legacy_style.yaml` | nearest policy-equivalent of the shipped prompt (full negatives, no visual-primary, no context) |
| P2 | `P2_pos_unc_no_neg.yaml` | positive + uncertain, no negatives |
| P3 | `P3_pos_unc_critical_neg.yaml` | + critical negatives (capped) |
| P4 | `P4_add_views.yaml` | + view / indication / technique metadata |
| P5 | `P5_visual_primary.yaml` | + explicit visual-primary instruction (recommended) |
| P6 | `P6_qformer_visual_only.yaml` | Q-Former soft tokens, **no** structured labels |
| P7 | `P7_confidence_bins.yaml` | P5 + calibrated confidence bins (needs calibration) |
| P8 | `P8_compact_normal.yaml` | compact normal policy end-to-end (= v2) |
| P9 | `P9_full_negative_control.yaml` | full negative list (control) |

The byte-exact legacy prompt is produced by the old `build_instruction` (i.e.
`prompt_config=None`); P1 is the closest reconstruction within the v2 framework.

## Run
```bash
# Dry run on synthetic fixtures (no data, no GPU) — sanity only:
python scripts/run_prompt_ablation.py \
  --prompt-configs configs/prompt_ablation/P*.yaml \
  --max-samples 200 --output-dir outputs/prompt_ablation

# Dry run on real validation records (restricted; private box):
python scripts/run_prompt_ablation.py \
  --records outputs/validation_records.jsonl \
  --prompt-configs configs/prompt_ablation/P*.yaml \
  --max-samples 1000 --output-dir outputs/prompt_ablation
```
The dry run writes `<variant>_per_sample_results.jsonl` (prompt hash, cues, view/prior
flags, `no_prior_guard_present`, approx prompt length, `possible_temporal_in_reference_without_prior`)
and `ablation_summary.json` (per-variant prompt-length, average negatives shown, guard
rate). No generated report and no NLG/clinical metric is produced in dry-run mode.

## Metrics for real selection (when a checkpoint exists)
Rank in this priority order: clinical factuality → hallucination rate → omission rate
→ clinical metrics (CheXbert F1 if available) → lexical metrics (BLEU-4, ROUGE-L,
METEOR, CIDEr, BERTScore F1) → token length / latency / VRAM. Do not choose on BLEU
alone. RadGraph/RadCliQ are **not** implemented in this repo; do not report them.

## Results — NOT RUN
No ablation model metrics exist. This document defines the protocol; running it against
a real Stage-2 checkpoint on the validation split is future work. Any numbers produced
from the synthetic fixtures are illustrative and are **not** MIMIC-CXR results.
