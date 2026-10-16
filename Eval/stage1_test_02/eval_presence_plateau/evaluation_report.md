# Evaluation report

Generated 2026-08-21T01:01:26.826189+00:00 from commit `fec606b8cc2e`


---

## Experiment metadata

| Field | Value |
| --- | --- |
| Split | `test` |
| Studies | 3269 |
| Pathologies | 14 |
| Checkpoint | `run_20260820_ft/checkpoint_best.pth (epoch 9)` |
| Config | `pretraining/configs/mimic_cxr_full.yaml @ fec606b` |
| Seed | 42 |
| Uncertain policy | `ignore_uncertain` |
| Threshold source | `Test/stage1_test_02/thresholds_presence_marginal_plateau.json` |
| Device | cpu |
| Git commit | `fec606b8cc2e8cd55a493fcc82bd87bcfca67d05` |


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
| `positive_macro_f1` | 0.3542 | [0.3418, 0.3647] |
| `positive_macro_recall` | 0.5373 | [0.5184, 0.5555] |
| `positive_macro_precision` | 0.2931 | [0.2816, 0.3038] |
| `macro_auroc` | 0.7643 | [0.7545, 0.7740] |
| `macro_auprc` | 0.3203 | [0.3108, 0.3362] |
| `positive_micro_f1` | 0.4752 | [0.4648, 0.4854] |

### All aggregates

| Metric | Value | 95% CI |
| --- | ---: | :---: |
| `accuracy` | 0.8307 | - |
| `binary_accuracy` | 0.7944 | - |
| `balanced_accuracy` | 0.6696 | - |
| `macro_precision` | 0.6874 | - |
| `macro_recall` | 0.6221 | - |
| `macro_f1` | 0.6747 | - |
| `micro_precision` | 0.8307 | - |
| `micro_recall` | 0.8307 | - |
| `micro_f1` | 0.8307 | - |
| `weighted_precision` | 0.8600 | - |
| `weighted_recall` | 0.8307 | - |
| `weighted_f1` | 0.8371 | - |
| `positive_micro_precision` | 0.3695 | - |
| `positive_micro_recall` | 0.6658 | - |
| `positive_weighted_precision` | 0.4259 | - |
| `positive_weighted_recall` | 0.6658 | - |
| `positive_weighted_f1` | 0.5082 | - |
| `macro_specificity` | 0.8020 | - |
| `macro_npv` | 0.9290 | - |
| `micro_auroc` | 0.8166 | - |
| `micro_auprc` | 0.4272 | - |

### Baseline comparison

A model whose accuracy is close to `all_negative` while its positive macro F1 is near 0 has not learned to detect findings.

| Model | Accuracy | Positive Macro F1 | Macro AUROC | Macro AUPRC |
| --- | ---: | ---: | ---: | ---: |
| **model** | 0.7944 | 0.3542 | 0.7643 | 0.3203 |
| all_negative | 0.8602 | 0.0000 | 0.7643 | 0.3203 |
| all_positive | 0.1398 | 0.2280 | 0.7643 | 0.3203 |
| majority_class | 0.8602 | 0.0000 | 0.7643 | 0.3203 |
| prevalence_random | 0.7864 | 0.1418 | 0.7643 | 0.3203 |
| threshold_half | 0.8307 | 0.2922 | 0.7643 | 0.3203 |


### Per-pathology

| Pathology | n+ | Prev. | P | R | F1 | AUROC | AUPRC | Thr. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Finding | 568 | 0.1738 | 0.5235 | 0.4912 | 0.5068 | 0.8081 | 0.5342 | 0.5071 |
| Enlarged Cardiomediastinum | 151 | 0.0462 | 0.0862 | 0.2450 | 0.1276 | 0.6321 | 0.0757 | 0.5119 |
| Cardiomegaly | 866 | 0.2649 | 0.3971 | 0.8487 | 0.5410 | 0.7694 | 0.5101 | 0.3500 |
| Lung Opacity | 1038 | 0.3175 | 0.4510 | 0.7042 | 0.5498 | 0.7099 | 0.5048 | 0.3335 |
| Lung Lesion | 123 | 0.0376 | 0.1106 | 0.5366 | 0.1833 | 0.7655 | 0.1216 | 0.3116 |
| Edema | 683 | 0.2089 | 0.5153 | 0.6179 | 0.5619 | 0.8353 | 0.5764 | 0.6553 |
| Consolidation | 220 | 0.0673 | 0.2618 | 0.2273 | 0.2433 | 0.7483 | 0.1918 | 0.6595 |
| Pneumonia | 335 | 0.1025 | 0.2981 | 0.2866 | 0.2922 | 0.7046 | 0.2266 | 0.5715 |
| Atelectasis | 731 | 0.2236 | 0.3595 | 0.7196 | 0.4795 | 0.7379 | 0.4105 | 0.3209 |
| Pneumothorax | 112 | 0.0343 | 0.2713 | 0.4554 | 0.3400 | 0.8585 | 0.2711 | 0.3682 |
| Pleural Effusion | 1074 | 0.3285 | 0.6378 | 0.8101 | 0.7137 | 0.8760 | 0.7823 | 0.3950 |
| Pleural Other | 63 | 0.0193 | 0.0674 | 0.7937 | 0.1242 | 0.8667 | 0.1029 | 0.1576 |
| Fracture | 89 | 0.0272 | 0.0606 | 0.2022 | 0.0933 | 0.6670 | 0.0697 | 0.2349 |
| Support Devices | 1125 | 0.3441 | 0.6829 | 0.8364 | 0.7519 | 0.8838 | 0.7745 | 0.1047 |


### Subgroups

| Subgroup | Samples | positive_macro_f1 | macro_auroc | macro_auprc |
| --- | ---: | ---: | ---: | ---: |
| normal_studies | 138 | n/a | n/a | n/a |
| abnormal_studies | 3131 | 0.3604 | 0.7628 | 0.3278 |


---

## Limitations

- Lexical metrics (BLEU/ROUGE/METEOR/CIDEr/BERTScore) measure surface overlap. They do not measure clinical correctness: a report can score well while inverting a negation.
- All `possible_*` error flags are lexicon heuristics over surface text. They are screening signals for triage, **not** radiologist-confirmed clinical errors, and must not be reported as clinical error rates.
- Clinical metrics (CheXbert, RadGraph, RadCliQ, RadFact) are interfaced but not implemented in this repository. Any row showing them as unavailable means the dependency is absent, not that the model scored 0.
- No metric here substitutes for radiologist review.
