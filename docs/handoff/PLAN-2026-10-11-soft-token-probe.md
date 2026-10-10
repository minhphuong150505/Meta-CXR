# 2026-10-11 — Linear probe of the v3 soft tokens (run_20261005_paper)

Question (user): paper-mode Stage 2 reads the cues only weakly and the NLG
numbers are low -- is it the soft tokens?

Script: `scripts/probe_soft_tokens_cached.py` (new). Reads the
`.sensitive_stage1_cache` that the v3 Stage-2 run wrote from
`run_20261005_paper` phase-1c best (`qformer_embs` [32, 768], `class_logits`
[14, 3] per study). Linear multinomial heads (14 findings x 3 classes), fit on
50,000 random train studies, L2 chosen on val, scored once on test against
MHCAC's own predictions on the SAME studies with the SAME labels (D-018 rule).
Host: `~/probe_softtok_20261011/{run.log,report.json}`, rc 0.

| | test auroc_mean | Neg | Pos | Unc |
|---|---:|---:|---:|---:|
| MHCAC (same 2,780 test studies) | 0.7454 | 0.7800 | 0.7847 | 0.6162 |
| probe, pooled (mean of 32 tokens) | 0.7127 | 0.7262 | 0.7464 | 0.6301 |
| probe, all 32 tokens (24,576-d) | 0.6530 | 0.6779 | 0.6748 | 0.5715 |
| probe, pooled, shuffled train labels | 0.5607 | | | |

Positive-class AUROC per finding, MHCAC / pooled: equal on the common
findings (No Finding 0.826/0.826, Edema 0.843/0.848, Pleural Effusion
0.899/0.896, Support Devices 0.905/0.901, Cardiomegaly 0.756/0.750); lost on the
rare ones (Pleural Other 0.833/0.635, Fracture 0.684/0.488 = chance, Lung
Lesion 0.748/0.715, Consolidation 0.766/0.741, Pneumothorax 0.865/0.834).

**Token collapse:** mean cosine between the 32 tokens of one study **+0.9995**
(across studies +0.4112). The 32 queries emit essentially one vector; the LLM
receives one image vector repeated 32 times. The untrained BLIP-2 readout
measured +0.80 on 2026-09-03 (`run_20260820_ft`), so phase-1a training made it
worse, not better.

Reading:
- The soft tokens keep most of Stage 1's label-level information: pooled is
  (0.713-0.5)/(0.745-0.5) = 87% of MHCAC's above-chance AUROC, and ~all of it
  on the common findings. Missing label information is NOT the main reason
  the reports are poor.
- What is lost is (a) the rare findings and (b) everything a single vector
  cannot hold (location, extent, multiple co-occurring findings, device
  detail) -- consistent with the omission signature (BERTScore P 0.424 /
  R 0.299).
- The 32-token probe is worse than pooled: with collapsed tokens the extra
  24k dimensions add only noise to fit; read it as an optimisation artefact,
  not as "tokens carry less than their mean".
- Caveat: the shuffled-label control is 0.56, not 0.50 -- the four noise models
  scored 0.35-0.66 on val and the selection keeps the highest, so treat
  probe-vs-MHCAC gaps below ~0.05 as approximate. The per-finding pattern
  (common equal, rare lost) is the robust part.

Verdict for the user's question: partly. The soft tokens are a bottleneck by
**collapse** (one effective token), not by missing the 14-label signal.
Candidate fixes (none tried): a query-diversity / decorrelation term in phase
1a, or feeding MedGemma the encoder patch tokens through a projector instead
of the Q-Former. The oracle-cue test (ground-truth labels as cues) is still
the cheapest check of whether the LLM side can use correct information.
