# Evaluation report

Generated 2026-08-21T01:00:02.881964+00:00 from commit `b4757127b900`


---

## Experiment metadata

| Field | Value |
| --- | --- |
| Split | `test` |
| Studies | 3269 |
| Pathologies | 14 |
| Checkpoint | `run_20260820_ft/checkpoint_best.pth (epoch 9)` |
| Config | `pretraining/configs/mimic_cxr_full.yaml @ b475712` |
| Seed | 42 |
| Uncertain policy | `ignore_uncertain` |
| Threshold source | `Test/stage1_test_02/thresholds_masked_f1.json` |
| Device | cpu |
| Git commit | `b4757127b90029e4812148ffad46d6f8b2d72c05` |


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
| `positive_macro_f1` | 0.8841 | [0.8744, 0.8934] |
| `positive_macro_recall` | 0.9017 | [0.8902, 0.9132] |
| `positive_macro_precision` | 0.8679 | [0.8555, 0.8797] |
| `macro_auroc` | 0.7985 | [0.7675, 0.8318] |
| `macro_auprc` | 0.9099 | [0.8980, 0.9205] |
| `positive_micro_f1` | 0.9159 | [0.9101, 0.9215] |

### All aggregates

| Metric | Value | 95% CI |
| --- | ---: | :---: |
| `accuracy` | 0.7253 | - |
| `binary_accuracy` | 0.8754 | - |
| `balanced_accuracy` | 0.6732 | - |
| `macro_precision` | 0.6387 | - |
| `macro_recall` | 0.4662 | - |
| `macro_f1` | 0.6842 | - |
| `micro_precision` | 0.7253 | - |
| `micro_recall` | 0.7253 | - |
| `micro_f1` | 0.7253 | - |
| `weighted_precision` | 0.7260 | - |
| `weighted_recall` | 0.7275 | - |
| `weighted_f1` | 0.7894 | - |
| `positive_micro_precision` | 0.8985 | - |
| `positive_micro_recall` | 0.9340 | - |
| `positive_weighted_precision` | 0.8993 | - |
| `positive_weighted_recall` | 0.9340 | - |
| `positive_weighted_f1` | 0.9162 | - |
| `macro_specificity` | 0.4536 | - |
| `macro_npv` | 0.7045 | - |
| `micro_auroc` | 0.8849 | - |
| `micro_auprc` | 0.9342 | - |

### Baseline comparison

A model whose accuracy is close to `all_negative` while its positive macro F1 is near 0 has not learned to detect findings.

| Model | Accuracy | Positive Macro F1 | Macro AUROC | Macro AUPRC |
| --- | ---: | ---: | ---: | ---: |
| **model** | 0.8754 | 0.8841 | 0.7985 | 0.9099 |
| all_negative | 0.2738 | 0.0000 | 0.7985 | 0.9099 |
| all_positive | 0.7262 | 0.8397 | 0.7985 | 0.9099 |
| majority_class | 0.8076 | 0.8200 | 0.7985 | 0.9099 |
| prevalence_random | 0.7191 | 0.7617 | 0.7985 | 0.9099 |
| threshold_half | 0.8661 | 0.8760 | 0.7985 | 0.9099 |


### Per-pathology

| Pathology | n+ | Prev. | P | R | F1 | AUROC | AUPRC | Thr. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Finding | 568 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | n/a | 1.0000 | 0.5000 |
| Enlarged Cardiomediastinum | 151 | 0.7023 | 0.8077 | 0.8344 | 0.8208 | 0.7487 | 0.8500 | 0.4848 |
| Cardiomegaly | 866 | 0.8327 | 0.9006 | 0.9734 | 0.9356 | 0.9138 | 0.9799 | 0.2586 |
| Lung Opacity | 1038 | 0.9471 | 0.9479 | 0.9990 | 0.9728 | 0.7721 | 0.9824 | 0.2446 |
| Lung Lesion | 123 | 0.8978 | 0.9044 | 1.0000 | 0.9498 | 0.7033 | 0.9521 | 0.3708 |
| Edema | 683 | 0.6913 | 0.8766 | 0.8843 | 0.8805 | 0.8893 | 0.9431 | 0.5088 |
| Consolidation | 220 | 0.6984 | 0.9174 | 0.9091 | 0.9132 | 0.9123 | 0.9592 | 0.3360 |
| Pneumonia | 335 | 0.5556 | 0.7529 | 0.7642 | 0.7585 | 0.8128 | 0.8390 | 0.4266 |
| Atelectasis | 731 | 0.9734 | 0.9758 | 0.9932 | 0.9844 | 0.7795 | 0.9925 | 0.4521 |
| Pneumothorax | 112 | 0.1335 | 0.4615 | 0.5357 | 0.4959 | 0.8049 | 0.4690 | 0.3916 |
| Pleural Effusion | 1074 | 0.7595 | 0.9022 | 0.9274 | 0.9146 | 0.9220 | 0.9735 | 0.3516 |
| Pleural Other | 63 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | n/a | 1.0000 | 0.5000 |
| Fracture | 89 | 0.9674 | 0.9674 | 1.0000 | 0.9834 | 0.5243 | 0.9776 | 0.5000 |
| Support Devices | 1125 | 0.9707 | 0.9745 | 0.9849 | 0.9797 | 0.8055 | 0.9919 | 0.1521 |


### Pathologies with undefined metrics

These are **excluded** from the macro averages rather than counted as zero.

- **No Finding**: no_negative_samples
- **Pleural Other**: no_negative_samples

### Subgroups

| Subgroup | Samples | positive_macro_f1 | macro_auroc | macro_auprc |
| --- | ---: | ---: | ---: | ---: |
| normal_studies | 138 | n/a | n/a | n/a |
| abnormal_studies | 3131 | 0.8858 | 0.7935 | 0.9114 |


---

## Limitations

- Lexical metrics (BLEU/ROUGE/METEOR/CIDEr/BERTScore) measure surface overlap. They do not measure clinical correctness: a report can score well while inverting a negation.
- All `possible_*` error flags are lexicon heuristics over surface text. They are screening signals for triage, **not** radiologist-confirmed clinical errors, and must not be reported as clinical error rates.
- Clinical metrics (CheXbert, RadGraph, RadCliQ, RadFact) are interfaced but not implemented in this repository. Any row showing them as unavailable means the dependency is absent, not that the model scored 0.
- No metric here substitutes for radiologist review.
