# Evaluation report

Generated 2026-08-21T09:06:25.019250+00:00 from commit `4c0bfa7ddc58`


---

## Experiment metadata

| Field | Value |
| --- | --- |
| Split | `unknown` |
| Studies | 3269 |
| Pathologies | 14 |
| Checkpoint | `unknown` |
| Config | `unknown` |
| Seed | 42 |
| Uncertain policy | `ignore_uncertain` |
| Threshold source | `Test/stage1_test_03/thresholds_masked_f1.json` |
| Device | cpu |
| Git commit | `4c0bfa7ddc5854e09168b52edee61bd322a23b16` |


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
| `positive_macro_f1` | 0.8815 | [0.8719, 0.8909] |
| `positive_macro_recall` | 0.9002 | [0.8895, 0.9124] |
| `positive_macro_precision` | 0.8667 | [0.8549, 0.8780] |
| `macro_auroc` | 0.7833 | [0.7508, 0.8199] |
| `macro_auprc` | 0.9060 | [0.8943, 0.9168] |
| `positive_micro_f1` | 0.9125 | [0.9069, 0.9181] |

### All aggregates

| Metric | Value | 95% CI |
| --- | ---: | :---: |
| `accuracy` | 0.7158 | - |
| `binary_accuracy` | 0.8708 | - |
| `balanced_accuracy` | 0.6772 | - |
| `macro_precision` | 0.6374 | - |
| `macro_recall` | 0.4709 | - |
| `macro_f1` | 0.6809 | - |
| `micro_precision` | 0.7158 | - |
| `micro_recall` | 0.7158 | - |
| `micro_f1` | 0.7158 | - |
| `weighted_precision` | 0.7287 | - |
| `weighted_recall` | 0.7207 | - |
| `weighted_f1` | 0.7857 | - |
| `positive_micro_precision` | 0.8975 | - |
| `positive_micro_recall` | 0.9280 | - |
| `positive_weighted_precision` | 0.9014 | - |
| `positive_weighted_recall` | 0.9280 | - |
| `positive_weighted_f1` | 0.9135 | - |
| `macro_specificity` | 0.4633 | - |
| `macro_npv` | 0.6802 | - |
| `micro_auroc` | 0.8637 | - |
| `micro_auprc` | 0.9114 | - |

### Baseline comparison

A model whose accuracy is close to `all_negative` while its positive macro F1 is near 0 has not learned to detect findings.

| Model | Accuracy | Positive Macro F1 | Macro AUROC | Macro AUPRC |
| --- | ---: | ---: | ---: | ---: |
| **model** | 0.8708 | 0.8815 | 0.7833 | 0.9060 |
| all_negative | 0.2738 | 0.0000 | 0.7833 | 0.9060 |
| all_positive | 0.7262 | 0.8397 | 0.7833 | 0.9060 |
| majority_class | 0.8076 | 0.8200 | 0.7833 | 0.9060 |
| prevalence_random | 0.7191 | 0.7617 | 0.7833 | 0.9060 |
| threshold_half | 0.8542 | 0.8739 | 0.7833 | 0.9060 |


### Per-pathology

| Pathology | n+ | Prev. | P | R | F1 | AUROC | AUPRC | Thr. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Finding | 568 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | n/a | 1.0000 | 0.5000 |
| Enlarged Cardiomediastinum | 151 | 0.7023 | 0.8231 | 0.8013 | 0.8121 | 0.7505 | 0.8500 | 0.5246 |
| Cardiomegaly | 866 | 0.8327 | 0.9046 | 0.9746 | 0.9383 | 0.9147 | 0.9793 | 0.2510 |
| Lung Opacity | 1038 | 0.9471 | 0.9488 | 0.9990 | 0.9733 | 0.7543 | 0.9791 | 0.2004 |
| Lung Lesion | 123 | 0.8978 | 0.9037 | 0.9919 | 0.9457 | 0.6800 | 0.9502 | 0.4339 |
| Edema | 683 | 0.6913 | 0.8995 | 0.7994 | 0.8465 | 0.8879 | 0.9411 | 0.5069 |
| Consolidation | 220 | 0.6984 | 0.9144 | 0.9227 | 0.9186 | 0.9143 | 0.9606 | 0.2754 |
| Pneumonia | 335 | 0.5556 | 0.7577 | 0.8030 | 0.7797 | 0.8099 | 0.8371 | 0.4612 |
| Atelectasis | 731 | 0.9734 | 0.9759 | 0.9959 | 0.9858 | 0.7400 | 0.9856 | 0.4097 |
| Pneumothorax | 112 | 0.1335 | 0.4088 | 0.5804 | 0.4797 | 0.7921 | 0.4460 | 0.6121 |
| Pleural Effusion | 1074 | 0.7595 | 0.8963 | 0.9339 | 0.9147 | 0.9198 | 0.9717 | 0.2798 |
| Pleural Other | 63 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | n/a | 1.0000 | 0.5000 |
| Fracture | 89 | 0.9674 | 0.9674 | 1.0000 | 0.9834 | 0.4532 | 0.9718 | 0.5000 |
| Support Devices | 1125 | 0.9707 | 0.9707 | 1.0000 | 0.9851 | 0.8308 | 0.9932 | 0.0518 |


### Pathologies with undefined metrics

These are **excluded** from the macro averages rather than counted as zero.

- **No Finding**: no_negative_samples
- **Pleural Other**: no_negative_samples

### Subgroups

| Subgroup | Samples | positive_macro_f1 | macro_auroc | macro_auprc |
| --- | ---: | ---: | ---: | ---: |
| normal_studies | 138 | n/a | n/a | n/a |
| abnormal_studies | 3131 | 0.8832 | 0.7780 | 0.9078 |


---

## Limitations

- Lexical metrics (BLEU/ROUGE/METEOR/CIDEr/BERTScore) measure surface overlap. They do not measure clinical correctness: a report can score well while inverting a negation.
- All `possible_*` error flags are lexicon heuristics over surface text. They are screening signals for triage, **not** radiologist-confirmed clinical errors, and must not be reported as clinical error rates.
- Clinical metrics (CheXbert, RadGraph, RadCliQ, RadFact) are interfaced but not implemented in this repository. Any row showing them as unavailable means the dependency is absent, not that the model scored 0.
- No metric here substitutes for radiologist review.
