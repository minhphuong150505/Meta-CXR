# Evaluation report

Generated 2026-08-20T03:55:15.078577+00:00 from commit `1f2e7f0c4e5b`


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
| Threshold source | `Test/stage1_test/thresholds_presence_marginal_f1.json` |
| Device | cpu |
| Git commit | `1f2e7f0c4e5b605ed38a8a04a52aef1d4edce638` |


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
| `positive_macro_f1` | 0.3224 | [0.3109, 0.3326] |
| `positive_macro_recall` | 0.4261 | [0.4121, 0.4404] |
| `positive_macro_precision` | 0.2989 | [0.2862, 0.3117] |
| `macro_auroc` | 0.7441 | [0.7341, 0.7540] |
| `macro_auprc` | 0.3004 | [0.2919, 0.3139] |
| `positive_micro_f1` | 0.4881 | [0.4773, 0.4982] |

### All aggregates

| Metric | Value | 95% CI |
| --- | ---: | :---: |
| `accuracy` | 0.7674 | - |
| `binary_accuracy` | 0.8118 | - |
| `balanced_accuracy` | 0.6237 | - |
| `macro_precision` | 0.6269 | - |
| `macro_recall` | 0.6425 | - |
| `macro_f1` | 0.6173 | - |
| `micro_precision` | 0.7674 | - |
| `micro_recall` | 0.7674 | - |
| `micro_f1` | 0.7674 | - |
| `weighted_precision` | 0.8618 | - |
| `weighted_recall` | 0.7674 | - |
| `weighted_f1` | 0.7974 | - |
| `positive_micro_precision` | 0.3939 | - |
| `positive_micro_recall` | 0.6416 | - |
| `positive_weighted_precision` | 0.4213 | - |
| `positive_weighted_recall` | 0.6416 | - |
| `positive_weighted_f1` | 0.4942 | - |
| `macro_specificity` | 0.8214 | - |
| `macro_npv` | 0.9232 | - |
| `micro_auroc` | 0.8183 | - |
| `micro_auprc` | 0.4429 | - |

### Baseline comparison

A model whose accuracy is close to `all_negative` while its positive macro F1 is near 0 has not learned to detect findings.

| Model | Accuracy | Positive Macro F1 | Macro AUROC | Macro AUPRC |
| --- | ---: | ---: | ---: | ---: |
| **model** | 0.8118 | 0.3224 | 0.7441 | 0.3004 |
| all_negative | 0.8602 | 0.0000 | 0.7441 | 0.3004 |
| all_positive | 0.1398 | 0.2280 | 0.7441 | 0.3004 |
| majority_class | 0.8602 | 0.0000 | 0.7441 | 0.3004 |
| prevalence_random | 0.7864 | 0.1418 | 0.7441 | 0.3004 |
| threshold_half | 0.7674 | 0.3173 | 0.7441 | 0.3004 |


### Per-pathology

| Pathology | n+ | Prev. | P | R | F1 | AUROC | AUPRC | Thr. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Finding | 568 | 0.1738 | 0.4828 | 0.5194 | 0.5004 | 0.8049 | 0.5185 | 0.6459 |
| Enlarged Cardiomediastinum | 151 | 0.0462 | 0.0699 | 0.1722 | 0.0994 | 0.6126 | 0.0657 | 0.5528 |
| Cardiomegaly | 866 | 0.2649 | 0.4295 | 0.7356 | 0.5424 | 0.7595 | 0.5058 | 0.5540 |
| Lung Opacity | 1038 | 0.3175 | 0.3995 | 0.8160 | 0.5364 | 0.6915 | 0.4961 | 0.4018 |
| Lung Lesion | 123 | 0.0376 | 0.0948 | 0.4309 | 0.1554 | 0.7261 | 0.1151 | 0.4536 |
| Edema | 683 | 0.2089 | 0.4830 | 0.6252 | 0.5450 | 0.8245 | 0.5435 | 0.6869 |
| Consolidation | 220 | 0.0673 | 0.2353 | 0.2000 | 0.2162 | 0.7446 | 0.1796 | 0.6865 |
| Pneumonia | 335 | 0.1025 | 0.2483 | 0.3343 | 0.2850 | 0.6993 | 0.2176 | 0.6791 |
| Atelectasis | 731 | 0.2236 | 0.3545 | 0.7319 | 0.4777 | 0.7229 | 0.4071 | 0.5370 |
| Pneumothorax | 112 | 0.0343 | 0.2742 | 0.3036 | 0.2881 | 0.8283 | 0.1919 | 0.6352 |
| Pleural Effusion | 1074 | 0.3285 | 0.6582 | 0.7477 | 0.7001 | 0.8682 | 0.7738 | 0.6240 |
| Pleural Other | 63 | 0.0193 | 0.0400 | 0.0159 | 0.0227 | 0.7867 | 0.0528 | 0.5000 |
| Fracture | 89 | 0.0272 | n/a | 0.0000 | 0.0000 | 0.6651 | 0.0557 | 0.5000 |
| Support Devices | 1125 | 0.3441 | 0.6766 | 0.8276 | 0.7445 | 0.8783 | 0.7724 | 0.1328 |


### Pathologies with undefined metrics

These are **excluded** from the macro averages rather than counted as zero.

- **Fracture**: never_predicted_positive

### Subgroups

| Subgroup | Samples | positive_macro_f1 | macro_auroc | macro_auprc |
| --- | ---: | ---: | ---: | ---: |
| normal_studies | 138 | n/a | n/a | n/a |
| abnormal_studies | 3131 | 0.3269 | 0.7417 | 0.3069 |


---

## Limitations

- Lexical metrics (BLEU/ROUGE/METEOR/CIDEr/BERTScore) measure surface overlap. They do not measure clinical correctness: a report can score well while inverting a negation.
- All `possible_*` error flags are lexicon heuristics over surface text. They are screening signals for triage, **not** radiologist-confirmed clinical errors, and must not be reported as clinical error rates.
- Clinical metrics (CheXbert, RadGraph, RadCliQ, RadFact) are interfaced but not implemented in this repository. Any row showing them as unavailable means the dependency is absent, not that the model scored 0.
- No metric here substitutes for radiologist review.
