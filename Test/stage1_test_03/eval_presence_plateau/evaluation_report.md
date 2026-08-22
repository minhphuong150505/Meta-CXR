# Evaluation report

Generated 2026-08-21T09:03:03.723326+00:00 from commit `4c0bfa7ddc58`


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
| Threshold source | `Test/stage1_test_03/thresholds_presence_marginal_plateau.json` |
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
| `positive_macro_f1` | 0.3474 | [0.3330, 0.3600] |
| `positive_macro_recall` | 0.4544 | [0.4373, 0.4706] |
| `positive_macro_precision` | 0.2944 | [0.2809, 0.3074] |
| `macro_auroc` | 0.7638 | [0.7543, 0.7733] |
| `macro_auprc` | 0.3179 | [0.3082, 0.3330] |
| `positive_micro_f1` | 0.5009 | [0.4901, 0.5107] |

### All aggregates

| Metric | Value | 95% CI |
| --- | ---: | :---: |
| `accuracy` | 0.8241 | - |
| `binary_accuracy` | 0.8175 | - |
| `balanced_accuracy` | 0.6404 | - |
| `macro_precision` | 0.6423 | - |
| `macro_recall` | 0.6294 | - |
| `macro_f1` | 0.6354 | - |
| `micro_precision` | 0.8241 | - |
| `micro_recall` | 0.8241 | - |
| `micro_f1` | 0.8241 | - |
| `weighted_precision` | 0.8574 | - |
| `weighted_recall` | 0.8241 | - |
| `weighted_f1` | 0.8348 | - |
| `positive_micro_precision` | 0.4056 | - |
| `positive_micro_recall` | 0.6549 | - |
| `positive_weighted_precision` | 0.4208 | - |
| `positive_weighted_recall` | 0.6549 | - |
| `positive_weighted_f1` | 0.5050 | - |
| `macro_specificity` | 0.8264 | - |
| `macro_npv` | 0.9276 | - |
| `micro_auroc` | 0.8056 | - |
| `micro_auprc` | 0.3912 | - |

### Baseline comparison

A model whose accuracy is close to `all_negative` while its positive macro F1 is near 0 has not learned to detect findings.

| Model | Accuracy | Positive Macro F1 | Macro AUROC | Macro AUPRC |
| --- | ---: | ---: | ---: | ---: |
| **model** | 0.8175 | 0.3474 | 0.7638 | 0.3179 |
| all_negative | 0.8602 | 0.0000 | 0.7638 | 0.3179 |
| all_positive | 0.1398 | 0.2280 | 0.7638 | 0.3179 |
| majority_class | 0.8602 | 0.0000 | 0.7638 | 0.3179 |
| prevalence_random | 0.7864 | 0.1418 | 0.7638 | 0.3179 |
| threshold_half | 0.8241 | 0.2983 | 0.7638 | 0.3179 |


### Per-pathology

| Pathology | n+ | Prev. | P | R | F1 | AUROC | AUPRC | Thr. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Finding | 568 | 0.1738 | 0.5341 | 0.4824 | 0.5069 | 0.8071 | 0.5365 | 0.5446 |
| Enlarged Cardiomediastinum | 151 | 0.0462 | 0.1131 | 0.1656 | 0.1344 | 0.6487 | 0.0848 | 0.5821 |
| Cardiomegaly | 866 | 0.2649 | 0.4066 | 0.8476 | 0.5496 | 0.7726 | 0.5092 | 0.2882 |
| Lung Opacity | 1038 | 0.3175 | 0.4388 | 0.7389 | 0.5506 | 0.7079 | 0.5080 | 0.3461 |
| Lung Lesion | 123 | 0.0376 | 0.1120 | 0.4634 | 0.1804 | 0.7686 | 0.1244 | 0.3590 |
| Edema | 683 | 0.2089 | 0.5060 | 0.6149 | 0.5552 | 0.8331 | 0.5669 | 0.5150 |
| Consolidation | 220 | 0.0673 | 0.2400 | 0.1909 | 0.2127 | 0.7511 | 0.1806 | 0.6805 |
| Pneumonia | 335 | 0.1025 | 0.2540 | 0.2836 | 0.2680 | 0.7014 | 0.2234 | 0.5904 |
| Atelectasis | 731 | 0.2236 | 0.3611 | 0.6990 | 0.4762 | 0.7376 | 0.4159 | 0.3462 |
| Pneumothorax | 112 | 0.0343 | 0.3083 | 0.3304 | 0.3190 | 0.8542 | 0.2632 | 0.7919 |
| Pleural Effusion | 1074 | 0.3285 | 0.6289 | 0.8222 | 0.7127 | 0.8784 | 0.7899 | 0.2920 |
| Pleural Other | 63 | 0.0193 | 0.1016 | 0.2063 | 0.1361 | 0.8615 | 0.0909 | 0.3375 |
| Fracture | 89 | 0.0272 | 0.0625 | 0.0899 | 0.0737 | 0.6499 | 0.0576 | 0.3015 |
| Support Devices | 1125 | 0.3441 | 0.6931 | 0.8453 | 0.7617 | 0.8869 | 0.7819 | 0.0894 |


### Subgroups

| Subgroup | Samples | positive_macro_f1 | macro_auroc | macro_auprc |
| --- | ---: | ---: | ---: | ---: |
| normal_studies | 138 | n/a | n/a | n/a |
| abnormal_studies | 3131 | 0.3534 | 0.7620 | 0.3249 |


---

## Limitations

- Lexical metrics (BLEU/ROUGE/METEOR/CIDEr/BERTScore) measure surface overlap. They do not measure clinical correctness: a report can score well while inverting a negation.
- All `possible_*` error flags are lexicon heuristics over surface text. They are screening signals for triage, **not** radiologist-confirmed clinical errors, and must not be reported as clinical error rates.
- Clinical metrics (CheXbert, RadGraph, RadCliQ, RadFact) are interfaced but not implemented in this repository. Any row showing them as unavailable means the dependency is absent, not that the model scored 0.
- No metric here substitutes for radiologist review.
