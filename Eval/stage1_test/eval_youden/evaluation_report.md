# Evaluation report

Generated 2026-08-20T02:48:13.442434+00:00 from commit `11e35943e704`


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
| Threshold source | `Test/stage1_test/thresholds_youden_j.json` |
| Device | cpu |
| Git commit | `11e35943e704101786be13c9314e1cbb8a05c023` |


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
| `positive_macro_f1` | 0.8517 | [0.8422, 0.8609] |
| `positive_macro_recall` | 0.8647 | [0.8512, 0.8780] |
| `positive_macro_precision` | 0.8554 | [0.8443, 0.8663] |
| `macro_auroc` | 0.7857 | [0.7509, 0.8177] |
| `macro_auprc` | 0.8951 | [0.8836, 0.9071] |
| `positive_micro_f1` | 0.8733 | [0.8667, 0.8807] |

### All aggregates

| Metric | Value | 95% CI |
| --- | ---: | :---: |
| `accuracy` | 0.7111 | - |
| `binary_accuracy` | 0.8184 | - |
| `balanced_accuracy` | 0.7052 | - |
| `macro_precision` | 0.6725 | - |
| `macro_recall` | 0.4502 | - |
| `macro_f1` | 0.6648 | - |
| `micro_precision` | 0.7111 | - |
| `micro_recall` | 0.7111 | - |
| `micro_f1` | 0.7111 | - |
| `weighted_precision` | 0.7284 | - |
| `weighted_recall` | 0.7147 | - |
| `weighted_f1` | 0.7764 | - |
| `positive_micro_precision` | 0.8846 | - |
| `positive_micro_recall` | 0.8624 | - |
| `positive_weighted_precision` | 0.9082 | - |
| `positive_weighted_recall` | 0.8624 | - |
| `positive_weighted_f1` | 0.8818 | - |
| `macro_specificity` | 0.5580 | - |
| `macro_npv` | 0.5258 | - |
| `micro_auroc` | 0.8680 | - |
| `micro_auprc` | 0.9294 | - |

### Baseline comparison

A model whose accuracy is close to `all_negative` while its positive macro F1 is near 0 has not learned to detect findings.

| Model | Accuracy | Positive Macro F1 | Macro AUROC | Macro AUPRC |
| --- | ---: | ---: | ---: | ---: |
| **model** | 0.8184 | 0.8517 | 0.7857 | 0.8951 |
| all_negative | 0.2738 | 0.0000 | 0.7857 | 0.8951 |
| all_positive | 0.7262 | 0.8397 | 0.7857 | 0.8951 |
| majority_class | 0.8076 | 0.8200 | 0.7857 | 0.8951 |
| prevalence_random | 0.7191 | 0.7617 | 0.7857 | 0.8951 |
| threshold_half | 0.8540 | 0.8713 | 0.7857 | 0.8951 |


### Per-pathology

| Pathology | n+ | Prev. | P | R | F1 | AUROC | AUPRC | Thr. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Finding | 568 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | n/a | 1.0000 | 0.5000 |
| Enlarged Cardiomediastinum | 151 | 0.7023 | 0.8276 | 0.7947 | 0.8108 | 0.7551 | 0.8517 | 0.5316 |
| Cardiomegaly | 866 | 0.8327 | 0.9448 | 0.8499 | 0.8948 | 0.8866 | 0.9705 | 0.6608 |
| Lung Opacity | 1038 | 0.9471 | 0.9632 | 0.9085 | 0.9351 | 0.7209 | 0.9748 | 0.5438 |
| Lung Lesion | 123 | 0.8978 | 0.9130 | 0.8537 | 0.8824 | 0.6545 | 0.9355 | 0.7832 |
| Edema | 683 | 0.6913 | 0.8718 | 0.8565 | 0.8641 | 0.8800 | 0.9343 | 0.6168 |
| Consolidation | 220 | 0.6984 | 0.8652 | 0.9045 | 0.8844 | 0.8833 | 0.9395 | 0.4366 |
| Pneumonia | 335 | 0.5556 | 0.7120 | 0.8119 | 0.7587 | 0.7943 | 0.8247 | 0.5382 |
| Atelectasis | 731 | 0.9734 | 0.9874 | 0.8591 | 0.9188 | 0.7598 | 0.9907 | 0.7300 |
| Pneumothorax | 112 | 0.1335 | 0.2810 | 0.6875 | 0.3990 | 0.7757 | 0.3679 | 0.4433 |
| Pleural Effusion | 1074 | 0.7595 | 0.9307 | 0.8501 | 0.8886 | 0.9110 | 0.9687 | 0.6752 |
| Pleural Other | 63 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | n/a | 1.0000 | 0.5000 |
| Fracture | 89 | 0.9674 | 0.9674 | 1.0000 | 0.9834 | 0.6217 | 0.9831 | 0.5000 |
| Support Devices | 1125 | 0.9707 | 0.9891 | 0.8036 | 0.8867 | 0.8166 | 0.9905 | 0.1911 |


### Pathologies with undefined metrics

These are **excluded** from the macro averages rather than counted as zero.

- **No Finding**: no_negative_samples
- **Pleural Other**: no_negative_samples

### Subgroups

| Subgroup | Samples | positive_macro_f1 | macro_auroc | macro_auprc |
| --- | ---: | ---: | ---: | ---: |
| normal_studies | 138 | n/a | n/a | n/a |
| abnormal_studies | 3131 | 0.8532 | 0.7779 | 0.8962 |


---

## Limitations

- Lexical metrics (BLEU/ROUGE/METEOR/CIDEr/BERTScore) measure surface overlap. They do not measure clinical correctness: a report can score well while inverting a negation.
- All `possible_*` error flags are lexicon heuristics over surface text. They are screening signals for triage, **not** radiologist-confirmed clinical errors, and must not be reported as clinical error rates.
- Clinical metrics (CheXbert, RadGraph, RadCliQ, RadFact) are interfaced but not implemented in this repository. Any row showing them as unavailable means the dependency is absent, not that the model scored 0.
- No metric here substitutes for radiologist review.
