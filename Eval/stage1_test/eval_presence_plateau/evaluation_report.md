# Evaluation report

Generated 2026-08-20T04:38:51.057564+00:00 from commit `769a56ceb13a` (**working tree dirty**)


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
| Threshold source | `Test/stage1_test/thresholds_presence_marginal_plateau.json` |
| Device | cpu |
| Git commit | `769a56ceb13ac0fe35cb74c6d457198312118ef9` |


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
| `positive_macro_f1` | 0.3397 | [0.3275, 0.3515] |
| `positive_macro_recall` | 0.5006 | [0.4811, 0.5206] |
| `positive_macro_precision` | 0.2776 | [0.2657, 0.2888] |
| `macro_auroc` | 0.7441 | [0.7341, 0.7540] |
| `macro_auprc` | 0.3004 | [0.2919, 0.3139] |
| `positive_micro_f1` | 0.4685 | [0.4585, 0.4779] |

### All aggregates

| Metric | Value | 95% CI |
| --- | ---: | :---: |
| `accuracy` | 0.7674 | - |
| `binary_accuracy` | 0.7873 | - |
| `balanced_accuracy` | 0.6454 | - |
| `macro_precision` | 0.6269 | - |
| `macro_recall` | 0.6425 | - |
| `macro_f1` | 0.6173 | - |
| `micro_precision` | 0.7674 | - |
| `micro_recall` | 0.7674 | - |
| `micro_f1` | 0.7674 | - |
| `weighted_precision` | 0.8618 | - |
| `weighted_recall` | 0.7674 | - |
| `weighted_f1` | 0.7974 | - |
| `positive_micro_precision` | 0.3601 | - |
| `positive_micro_recall` | 0.6704 | - |
| `positive_weighted_precision` | 0.4052 | - |
| `positive_weighted_recall` | 0.6704 | - |
| `positive_weighted_f1` | 0.4966 | - |
| `macro_specificity` | 0.7901 | - |
| `macro_npv` | 0.9270 | - |
| `micro_auroc` | 0.8183 | - |
| `micro_auprc` | 0.4429 | - |

### Baseline comparison

A model whose accuracy is close to `all_negative` while its positive macro F1 is near 0 has not learned to detect findings.

| Model | Accuracy | Positive Macro F1 | Macro AUROC | Macro AUPRC |
| --- | ---: | ---: | ---: | ---: |
| **model** | 0.7873 | 0.3397 | 0.7441 | 0.3004 |
| all_negative | 0.8602 | 0.0000 | 0.7441 | 0.3004 |
| all_positive | 0.1398 | 0.2280 | 0.7441 | 0.3004 |
| majority_class | 0.8602 | 0.0000 | 0.7441 | 0.3004 |
| prevalence_random | 0.7864 | 0.1418 | 0.7441 | 0.3004 |
| threshold_half | 0.7674 | 0.3173 | 0.7441 | 0.3004 |


### Per-pathology

| Pathology | n+ | Prev. | P | R | F1 | AUROC | AUPRC | Thr. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Finding | 568 | 0.1738 | 0.4792 | 0.5282 | 0.5025 | 0.8049 | 0.5185 | 0.6386 |
| Enlarged Cardiomediastinum | 151 | 0.0462 | 0.0658 | 0.1921 | 0.0980 | 0.6126 | 0.0657 | 0.5423 |
| Cardiomegaly | 866 | 0.2649 | 0.3984 | 0.8268 | 0.5377 | 0.7595 | 0.5058 | 0.4716 |
| Lung Opacity | 1038 | 0.3175 | 0.4103 | 0.7707 | 0.5355 | 0.6915 | 0.4961 | 0.4248 |
| Lung Lesion | 123 | 0.0376 | 0.0927 | 0.4309 | 0.1525 | 0.7261 | 0.1151 | 0.4455 |
| Edema | 683 | 0.2089 | 0.4840 | 0.6208 | 0.5439 | 0.8245 | 0.5435 | 0.6904 |
| Consolidation | 220 | 0.0673 | 0.2383 | 0.2091 | 0.2228 | 0.7446 | 0.1796 | 0.6830 |
| Pneumonia | 335 | 0.1025 | 0.2377 | 0.4030 | 0.2990 | 0.6993 | 0.2176 | 0.6480 |
| Atelectasis | 731 | 0.2236 | 0.3556 | 0.7278 | 0.4778 | 0.7229 | 0.4071 | 0.5388 |
| Pneumothorax | 112 | 0.0343 | 0.2720 | 0.3036 | 0.2869 | 0.8283 | 0.1919 | 0.6312 |
| Pleural Effusion | 1074 | 0.3285 | 0.6186 | 0.7989 | 0.6973 | 0.8682 | 0.7738 | 0.5658 |
| Pleural Other | 63 | 0.0193 | 0.0527 | 0.5556 | 0.0963 | 0.7867 | 0.0528 | 0.2314 |
| Fracture | 89 | 0.0272 | 0.1049 | 0.1685 | 0.1293 | 0.6651 | 0.0557 | 0.2439 |
| Support Devices | 1125 | 0.3441 | 0.6581 | 0.8436 | 0.7394 | 0.8783 | 0.7724 | 0.1244 |


### Subgroups

| Subgroup | Samples | positive_macro_f1 | macro_auroc | macro_auprc |
| --- | ---: | ---: | ---: | ---: |
| normal_studies | 138 | n/a | n/a | n/a |
| abnormal_studies | 3131 | 0.3455 | 0.7417 | 0.3069 |


---

## Limitations

- Lexical metrics (BLEU/ROUGE/METEOR/CIDEr/BERTScore) measure surface overlap. They do not measure clinical correctness: a report can score well while inverting a negation.
- All `possible_*` error flags are lexicon heuristics over surface text. They are screening signals for triage, **not** radiologist-confirmed clinical errors, and must not be reported as clinical error rates.
- Clinical metrics (CheXbert, RadGraph, RadCliQ, RadFact) are interfaced but not implemented in this repository. Any row showing them as unavailable means the dependency is absent, not that the model scored 0.
- No metric here substitutes for radiologist review.
