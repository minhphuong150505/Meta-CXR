# Evaluation report

Generated 2026-08-20T03:30:54.067744+00:00 from commit `d623cb514a85` (**working tree dirty**)


---

## Experiment metadata

| Field | Value |
| --- | --- |
| Split | `test` |
| Studies | 3269 |
| Pathologies | 14 |
| Checkpoint | `run_20260819_xmpoff/checkpoint_best.pth (epoch 6)` |
| Config | `pretraining/configs/mimic_cxr_full.yaml` |
| Seed | 42 |
| Uncertain policy | `ignore_uncertain` |
| Threshold source | `Test/stage1_test/thresholds_presence_f1.json` |
| Device | cpu |
| Git commit | `d623cb514a853bb41124c2b83f00af7b2d6334e8` |


---

### Metric package versions

| Package | Version |
| --- | --- |
| `python` | 3.12.3 |
| `numpy` | 2.5.1 |
| `pandas` | 3.0.3 |
| `scikit-learn` | not installed |
| `torch` | 2.12.0+cpu |
| `nltk` | not installed |
| `bert-score` | not installed |
| `pycocoevalcap` | not installed |
| `matplotlib` | 3.11.1 |

---

## Stage 1 — classification

### Headline metrics

| Metric | Value | 95% CI |
| --- | ---: | :---: |
| `positive_macro_f1` | 0.3074 | [0.2963, 0.3167] |
| `positive_macro_recall` | 0.5786 | [0.5639, 0.5927] |
| `positive_macro_precision` | 0.2564 | [0.2454, 0.2667] |
| `macro_auroc` | 0.6767 | [0.6655, 0.6883] |
| `macro_auprc` | 0.2657 | [0.2585, 0.2776] |
| `positive_micro_f1` | 0.3316 | [0.3230, 0.3398] |

### All aggregates

| Metric | Value | 95% CI |
| --- | ---: | :---: |
| `accuracy` | 0.3681 | - |
| `binary_accuracy` | 0.6340 | - |
| `balanced_accuracy` | 0.6072 | - |
| `macro_precision` | 0.4927 | - |
| `macro_recall` | 0.5942 | - |
| `macro_f1` | 0.3316 | - |
| `micro_precision` | 0.3681 | - |
| `micro_recall` | 0.3681 | - |
| `micro_f1` | 0.3681 | - |
| `weighted_precision` | 0.7157 | - |
| `weighted_recall` | 0.3681 | - |
| `weighted_f1` | 0.3843 | - |
| `positive_micro_precision` | 0.2227 | - |
| `positive_micro_recall` | 0.6492 | - |
| `positive_weighted_precision` | 0.3949 | - |
| `positive_weighted_recall` | 0.6492 | - |
| `positive_weighted_f1` | 0.4724 | - |
| `macro_specificity` | 0.6359 | - |
| `macro_npv` | 0.9039 | - |
| `micro_auroc` | 0.6384 | - |
| `micro_auprc` | 0.2276 | - |

### Baseline comparison

A model whose accuracy is close to `all_negative` while its positive macro F1 is near 0 has not learned to detect findings.

| Model | Accuracy | Positive Macro F1 | Macro AUROC | Macro AUPRC |
| --- | ---: | ---: | ---: | ---: |
| **model** | 0.6340 | 0.3074 | 0.6767 | 0.2657 |
| all_negative | 0.8602 | 0.0000 | 0.6767 | 0.2657 |
| all_positive | 0.1398 | 0.2280 | 0.6767 | 0.2657 |
| majority_class | 0.8602 | 0.0000 | 0.6767 | 0.2657 |
| prevalence_random | 0.7864 | 0.1418 | 0.6767 | 0.2657 |
| threshold_half | 0.3681 | 0.2797 | 0.6767 | 0.2657 |


### Per-pathology

| Pathology | n+ | Prev. | P | R | F1 | AUROC | AUPRC | Thr. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Finding | 568 | 0.1738 | 0.1963 | 0.7394 | 0.3102 | 0.5445 | 0.1776 | 0.9750 |
| Enlarged Cardiomediastinum | 151 | 0.0462 | 0.0582 | 0.1722 | 0.0870 | 0.5537 | 0.0631 | 0.9125 |
| Cardiomegaly | 866 | 0.2649 | 0.4164 | 0.7136 | 0.5260 | 0.7387 | 0.4715 | 0.7901 |
| Lung Opacity | 1038 | 0.3175 | 0.3820 | 0.8044 | 0.5180 | 0.6407 | 0.4202 | 0.6050 |
| Lung Lesion | 123 | 0.0376 | 0.0689 | 0.3740 | 0.1163 | 0.6507 | 0.0751 | 0.8957 |
| Edema | 683 | 0.2089 | 0.4692 | 0.6032 | 0.5279 | 0.8100 | 0.5042 | 0.8665 |
| Consolidation | 220 | 0.0673 | 0.2283 | 0.1318 | 0.1671 | 0.7163 | 0.1512 | 0.9651 |
| Pneumonia | 335 | 0.1025 | 0.1870 | 0.4537 | 0.2648 | 0.6508 | 0.1811 | 0.8314 |
| Atelectasis | 731 | 0.2236 | 0.3148 | 0.6512 | 0.4244 | 0.6629 | 0.3403 | 0.7577 |
| Pneumothorax | 112 | 0.0343 | 0.2574 | 0.3125 | 0.2823 | 0.8207 | 0.1862 | 0.7009 |
| Pleural Effusion | 1074 | 0.3285 | 0.6478 | 0.7263 | 0.6848 | 0.8541 | 0.7455 | 0.8098 |
| Pleural Other | 63 | 0.0193 | 0.0193 | 1.0000 | 0.0378 | 0.5366 | 0.0221 | 0.5000 |
| Fracture | 89 | 0.0272 | 0.0272 | 1.0000 | 0.0530 | 0.4853 | 0.0274 | 0.5000 |
| Support Devices | 1125 | 0.3441 | 0.5893 | 0.7218 | 0.6488 | 0.7836 | 0.6413 | 0.1949 |


### Subgroups

| Subgroup | Samples | positive_macro_f1 | macro_auroc | macro_auprc |
| --- | ---: | ---: | ---: | ---: |
| normal_studies | 138 | n/a | n/a | n/a |
| abnormal_studies | 3131 | 0.3120 | 0.6718 | 0.2709 |


---

## Limitations

- Lexical metrics (BLEU/ROUGE/METEOR/CIDEr/BERTScore) measure surface overlap. They do not measure clinical correctness: a report can score well while inverting a negation.
- All `possible_*` error flags are lexicon heuristics over surface text. They are screening signals for triage, **not** radiologist-confirmed clinical errors, and must not be reported as clinical error rates.
- Clinical metrics (CheXbert, RadGraph, RadCliQ, RadFact) are interfaced but not implemented in this repository. Any row showing them as unavailable means the dependency is absent, not that the model scored 0.
- No metric here substitutes for radiologist review.
