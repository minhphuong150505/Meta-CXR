# Verified external numbers — every entry read from the primary source

## META-CXR original paper (the paper this repo forks)
Source: authors' README, github.com/DasithEdirisinghe/META-CXR (IEEE Xplore page empty, ResearchGate 403)
Citation: D. Edirisinghe, W. Nimalsiri, M. Hennayake, D. Meedeniya, G. Lim, "Chest X-Ray Report
Generation Using Abnormality Guided Vision Language Model", IEEE Access, vol. 13, pp. 157651-157673,
2025, doi:10.1109/ACCESS.2025.3606961
- Classification, MIMIC-CXR, 13 pathologies + No Finding, P/N/U 3-class: mean precision 0.87,
  recall 0.78, F1 0.73. Averaging scheme NOT stated in README.
- Classification, CheXpert, 5 pathologies, zero-shot: AUC 0.824, F1 0.699 (CheXzero 0.889 / 0.606)
- Report generation, MIMIC-CXR: BERTScore 0.426, CIDEr 0.291, BLEU-4 0.102, ROUGE-L 0.280, METEOR 0.173
- NOT stated: test split, JPG vs DICOM, BERTScore rescaling, metric toolkit, LLM id.
- Vision encoders per README: ResNet50, ViT, Swin.
STATUS: README only -- full paper text not yet read. Protocol unknown.

## R2Gen
Source: arXiv 2010.16056 PDF, Table 1 (p.4), Table 2 and Table 3 (p.5-6), read directly.
Citation: Z. Chen, Y. Song, T.-H. Chang, X. Wan, "Generating Radiology Reports via Memory-driven
Transformer", EMNLP 2020.
- Dataset footnote: physionet.org/content/mimic-cxr/2.0.0/ (MIMIC-CXR, not the JPG release named)
- Official MIMIC-CXR split. Test: 5,159 images, 3,269 reports, 293 patients. Train 368,960 img /
  222,758 reports. Val 2,991 img / 1,808 reports.
- Metrics: coco-caption (github tylin/coco-caption) -> pycocoevalcap. Beam 3. Mean of 5 runs.
- MIMIC-CXR test, full model: BL-1 0.353, BL-2 0.218, BL-3 0.145, BL-4 0.103, METEOR 0.142, ROUGE-L 0.277
- CE (CheXpert labeler on generated reports, 14 categories): P 0.333, R 0.273, F1 0.276
- No CIDEr reported.
- NOTE: test 3,269 reports and val 1,808 reports == this repo's Stage-1 test 3,269 / val 1,808 studies.

## MIMIC-CXR-JPG (dataset)
Source: physionet.org/content/mimic-cxr-jpg/2.1.0/ read directly.
Citation: A. Johnson, M. Lungren, Y. Peng, Z. Lu, R. Mark, S. Berkowitz, S. Horng, "MIMIC-CXR-JPG -
chest radiographs with structured labels" (version 2.1.0), PhysioNet, 2024, doi:10.13026/jsn5-t979
- DERIVED FROM MIMIC-CXR: JPG files derived from the DICOM images + structured labels from the reports.
- 377,110 JPG images; 227,827 reports (studies).
- Official split file: mimic-cxr-2.0.0-split.csv.gz (dicom_id, study_id, subject_id, split).
- Labels: mimic-cxr-2.0.0-chexpert.csv.gz and -negbio.csv.gz; 1.0 positive, 0.0 negative,
  -1.0 uncertain, blank = no mention.
=> A paper evaluating on MIMIC-CXR's OFFICIAL split scores the same studies and the same reports;
   only the image container differs. State this in the thesis with this citation.

## METransformer
Source: arxiv.org/html/2304.02211 read directly (Table 1 lower part).
Citation: Z. Wang, L. Liu, L. Wang, L. Zhou, "METransformer: Radiology Report Generation by
Transformer with Multiple Learnable Expert Tokens", arXiv:2304.02211, 2023.
- MIMIC-CXR: 377,110 images, 227,835 reports, 64,588 patients. Split 222,758 / 1,808 / 3,269 (= R2Gen).
- METransformer: BLEU-1 0.386, BLEU-2 0.250, BLEU-3 0.169, BLEU-4 0.124, ROUGE-L 0.291,
  METEOR 0.152, CIDEr 0.362
- Toolkit sentence: "standard evaluation protocol ... BLEU-4, METEOR, ROUGE-L, and CIDEr" (toolkit
  not named in the extracted text).
- Its table ALSO lists R2GenCMN / PPKED / GSK / MSAT -- SECONDHAND, do not quote as verified.

## PromptMRG
Source: arxiv.org/html/2308.12604 (v2) read directly, Table 1.
Citation: H. Jin, H. Che, Y. Lin, H. Chen, "PromptMRG: Diagnosis-Driven Prompts for Medical Report
Generation", arXiv:2308.12604, 2023/2024.
- "We follow the official split and the preprocessing from Chen et al. (2020)" -> 270,790 / 2,130 /
  3,858 samples (NOTE: 3,858 != R2Gen's own stated 3,269 reports -- image-level vs report-level).
- MIMIC-CXR: BLEU-1 0.398, BLEU-4 0.112, METEOR 0.157, ROUGE-L 0.268 (BLEU-2/3 and CIDEr not reported)
- CE via CheXbert, 14 labels, example-based: P 0.501, R 0.509, F1 0.476
- NLG toolkit not named.

## MedGemma Technical Report (the Stage-2 base model family)
Source: arxiv.org/html/2507.05201 (v4) read directly.
Citation: L. Yang et al., "MedGemma Technical Report", arXiv:2507.05201, 2025.
- MIMIC-CXR report generation, MedGemma 4B: RadGraph F1 29.5, on a 306-case set, Findings+Impression.
- MIMIC-CXR classification: macro F1 88.9 over 5 conditions (atelectasis, cardiomegaly,
  consolidation, edema, pleural effusion), Med-Gemini test set.
- MedGemma 1.5 not mentioned.
=> NOT numerically comparable with this repo (different metric, cohort, label set). Cite as context.

## CheXclusion
Source: arXiv 2003.00827 PDF, pages 3-6 read directly (Section 3-4, Table 1, Table 2).
Citation: L. Seyyed-Kalantari, G. Liu, M. McDermott, I. Y. Chen, M. Ghassemi, "CheXclusion: Fairness
gaps in deep chest X-ray classifiers", Pacific Symposium on Biocomputing (PSB) 2021; arXiv:2003.00827.
- MIMIC-CXR ("CXR"): 371,858 images, 65,079 patients, frontal+lateral, 14 labels.
- U-ZERO: all non-positive (negative, not mentioned, uncertain) -> negative. == this repo's
  study_presence framing.
- DenseNet-121 ImageNet init, multi-label BCE, 256x256, IMAGE-LEVEL, RANDOM 80-10-10 split with no
  patient overlap (NOT the official split). Threshold per label maximises F1 on val.
- Mean and 95% CI over 5 runs (different seeds).
- Table 2, CXR column, AUC: Airspace(Lung) Opacity 0.782, Atelectasis 0.837, Cardiomegaly 0.828,
  Consolidation 0.844, Edema 0.904, Effusion 0.933, Enlarged Card 0.757, Fracture 0.718,
  Lung Lesion 0.772, No Finding 0.868, Pleural Other 0.848, Pneumonia 0.748, Pneumothorax 0.903,
  Support Devices 0.927. AVERAGE 0.834 +- 0.001.

## Thesis formatting (Vietnamese convention)
Source: HCMUS Khoa Dien tu - Vien thong, "Huong dan ve khoa luan tot nghiep va seminar tot nghiep
(Nam 2022)", PDF read directly, p.1-2. (UIT-specific rules not found online.)
- A4, font size 13, line spacing 1.5, margins top/bottom/right 2 cm, left 2.5 cm, page number bottom right.
- Heading numbering 1.; 1.1; 1.1.1.
- Figures/diagrams numbered with captions; formulas clear with symbol explanations.
- Foreign author names in original form; translated technical terms keep the English term.
- References: [xx] Authors, "Title," Journal/Conference, volume, year, pages.
- Structure: cover, acknowledgements, abstract VN+EN, TOC, abbreviations, list of figures, list of
  tables, intro chapter, 2-3 content chapters, conclusion chapter, references, appendix.
