# Table 5 — encoder ablation F1

Status: **partial** (3/4 configurations completed).

All values use the full 3,216-study MIMIC-CXR test split, thresholds calibrated only on the 1,788-study validation split with `objective=f1`, `min_positive=20`, and `uncertain_policy=ignore_uncertain`. The five pathologies are Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Effusion.

| Configuration | Atelectasis | Cardiomegaly | Consolidation | Edema | Pleural Effusion | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|
| BioViL-T only | 0.5003 | 0.5655 | 0.2239 | 0.6070 | 0.7280 | **0.5249** |
| PubMedCLIP only ⚠ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| SwinV2 only | pending | pending | pending | pending | pending | pending |
| BioViL-T + PubMedCLIP + SwinV2 | 0.5019 | 0.5637 | 0.2239 | 0.6065 | 0.7287 | **0.5250** |

The all-three equivalence gate passed: mean F1 was 0.524952 versus the expected 0.5250 (tolerance ±0.005), with maximum per-probability difference 0.001106 from the stored best-checkpoint test artifact.

⚠ PubMedCLIP-only made no positive predictions for any of the five pathologies at the fixed validation thresholds. Its positive-score maxima ranged from 0.0459 to 0.1285 while the corresponding thresholds ranged from 0.3237 to 0.5136. The result is implausibly weak but was verified against the saved full-split predictions and is reported unchanged.

The trained checkpoint contains BioViL-T, PubMedCLIP ViT-B/32, and SwinV2. RAD-DINO and MedCLIP were not trained and are not ablation configurations. Ablation zeros inactive shared-token spans after projection, leaving the common expert tokens, MHCAC, soft/query tokens, shapes, and weights unchanged. See `table5_encoder_ablation.json` for provenance, live checkpoint-key audit, artifact paths, and pending/completed status.
