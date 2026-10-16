# Phan phoi nhan CheXpert theo split

Nguon: `mimic-cxr-2.0.0-chexpert.csv.gz` join theo `study_id` voi
`processed/full_allviews_v2/{train,val,test}.csv`, tinh o **muc study**
(khong phai muc anh). Do ngay 2026-08-15.

Anh xa cua `ReportDataset`: `1.0 -> Duong`, `0.0 -> Am`, `-1.0 -> Khong ro`,
**o trong -> IGNORE_LABEL (-100), bi mask khoi moi loss va moi metric**.

| split | study |
|---|---:|
| train | 222,758 |
| val | 1,808 |
| test | 3,269 |

## train (222,758 study)

| benh ly | duong | am | khong ro | trong | % co nhan | % duong cua (duong+am) | ty le +/- |
|---|---:|---:|---:|---:|---:|---:|---:|
| No Finding | 74,305 | 0 | 0 | 148,453 | 33.36% | 100.0% | inf |
| Enlarged Cardiomediastinum | 6,968 | 5,168 | 9,083 | 201,539 | 9.53% | 57.42% | 1.35 |
| Cardiomegaly | 43,602 | 15,613 | 5,869 | 157,674 | 29.22% | 73.63% | 2.79 |
| Lung Opacity | 50,099 | 2,993 | 3,726 | 165,940 | 25.51% | 94.36% | 16.74 |
| Lung Lesion | 6,091 | 841 | 1,114 | 214,712 | 3.61% | 87.87% | 7.24 |
| Edema | 26,093 | 25,133 | 12,777 | 158,755 | 28.73% | 50.94% | 1.04 |
| Consolidation | 10,476 | 7,817 | 4,186 | 200,279 | 10.09% | 57.27% | 1.34 |
| Pneumonia | 16,093 | 23,904 | 17,797 | 164,964 | 25.94% | 40.24% | 0.67 |
| Atelectasis | 44,718 | 1,502 | 10,041 | 166,497 | 25.26% | 96.75% | 29.77 |
| Pneumothorax | 10,171 | 41,271 | 1,106 | 170,210 | 23.59% | 19.77% | 0.25 |
| Pleural Effusion | 52,759 | 26,615 | 5,639 | 137,745 | 38.16% | 66.47% | 1.98 |
| Pleural Other | 1,933 | 126 | 743 | 219,956 | 1.26% | 93.88% | 15.34 |
| Fracture | 4,283 | 877 | 548 | 217,050 | 2.56% | 83.0% | 4.88 |
| Support Devices | 64,868 | 3,422 | 230 | 154,238 | 30.76% | 94.99% | 18.96 |

## val (1,808 study)

| benh ly | duong | am | khong ro | trong | % co nhan | % duong cua (duong+am) | ty le +/- |
|---|---:|---:|---:|---:|---:|---:|---:|
| No Finding | 582 | 0 | 0 | 1,226 | 32.19% | 100.0% | inf |
| Enlarged Cardiomediastinum | 60 | 51 | 88 | 1,609 | 11.01% | 54.05% | 1.18 |
| Cardiomegaly | 377 | 124 | 59 | 1,248 | 30.97% | 75.25% | 3.04 |
| Lung Opacity | 388 | 18 | 26 | 1,376 | 23.89% | 95.57% | 21.56 |
| Lung Lesion | 70 | 7 | 9 | 1,722 | 4.76% | 90.91% | 10.0 |
| Edema | 242 | 203 | 108 | 1,255 | 30.59% | 54.38% | 1.19 |
| Consolidation | 82 | 55 | 48 | 1,623 | 10.23% | 59.85% | 1.49 |
| Pneumonia | 128 | 166 | 144 | 1,370 | 24.23% | 43.54% | 0.77 |
| Atelectasis | 359 | 9 | 95 | 1,345 | 25.61% | 97.55% | 39.89 |
| Pneumothorax | 75 | 358 | 9 | 1,366 | 24.45% | 17.32% | 0.21 |
| Pleural Effusion | 467 | 203 | 45 | 1,093 | 39.55% | 69.7% | 2.3 |
| Pleural Other | 15 | 0 | 6 | 1,787 | 1.16% | 100.0% | inf |
| Fracture | 18 | 6 | 3 | 1,781 | 1.49% | 75.0% | 3.0 |
| Support Devices | 565 | 30 | 3 | 1,210 | 33.08% | 94.96% | 18.83 |

## test (3,269 study)

| benh ly | duong | am | khong ro | trong | % co nhan | % duong cua (duong+am) | ty le +/- |
|---|---:|---:|---:|---:|---:|---:|---:|
| No Finding | 568 | 0 | 0 | 2,701 | 17.38% | 100.0% | inf |
| Enlarged Cardiomediastinum | 151 | 64 | 204 | 2,850 | 12.82% | 70.23% | 2.36 |
| Cardiomegaly | 866 | 174 | 115 | 2,114 | 35.33% | 83.27% | 4.98 |
| Lung Opacity | 1,038 | 58 | 79 | 2,094 | 35.94% | 94.71% | 17.9 |
| Lung Lesion | 123 | 14 | 18 | 3,114 | 4.74% | 89.78% | 8.79 |
| Edema | 683 | 305 | 289 | 1,992 | 39.06% | 69.13% | 2.24 |
| Consolidation | 220 | 95 | 97 | 2,857 | 12.6% | 69.84% | 2.32 |
| Pneumonia | 335 | 268 | 350 | 2,316 | 29.15% | 55.56% | 1.25 |
| Atelectasis | 731 | 20 | 191 | 2,327 | 28.82% | 97.34% | 36.55 |
| Pneumothorax | 112 | 727 | 19 | 2,411 | 26.25% | 13.35% | 0.15 |
| Pleural Effusion | 1,074 | 340 | 130 | 1,725 | 47.23% | 75.95% | 3.16 |
| Pleural Other | 63 | 0 | 16 | 3,190 | 2.42% | 100.0% | inf |
| Fracture | 89 | 3 | 4 | 3,173 | 2.94% | 96.74% | 29.67 |
| Support Devices | 1,125 | 34 | 4 | 2,106 | 35.58% | 97.07% | 33.09 |

## Ap luc cua ham loss (tren train)

`CrossEntropyLoss(weight=w)` nhan trong so theo **nhan that**, nen tong ap luc
moi chieu = so mau x trong so lop do. `w-` luon = 1.0.

| benh ly | w+ | ap luc -> Duong | ap luc -> Am | ty le | huong |
|---|---:|---:|---:|---:|---|
| No Finding | 1.0 | 74,305 | 0 | inf | chi co the day ve Duong |
| Enlarged Cardiomediastinum | 0.86 | 5,992 | 5,168 | 1.16 | -> Duong |
| Cardiomegaly | 0.6 | 26,161 | 15,613 | 1.68 | -> Duong |
| Lung Opacity | 0.24 | 12,024 | 2,993 | 4.02 | -> Duong |
| Lung Lesion | 0.37 | 2,254 | 841 | 2.68 | -> Duong |
| Edema | 0.98 | 25,571 | 25,133 | 1.02 | -> Duong |
| Consolidation | 0.86 | 9,009 | 7,817 | 1.15 | -> Duong |
| Pneumonia | 1.22 | 19,633 | 23,904 | 1.22 | -> Am |
| Atelectasis | 0.18 | 8,049 | 1,502 | 5.36 | -> Duong |
| Pneumothorax | 2.01 | 20,444 | 41,271 | 2.00 | -> Am |
| Pleural Effusion | 0.71 | 37,459 | 26,615 | 1.41 | -> Duong |
| Pleural Other | 0.26 | 503 | 126 | 3.99 | -> Duong |
| Fracture | 0.45 | 1,927 | 877 | 2.20 | -> Duong |
| Support Devices | 0.23 | 14,920 | 3,422 | 4.36 | -> Duong |

## Doi chieu voi ket qua do tren test

checkpoint_best (epoch [8]), nguong hieu chinh tren train+val, `ignore_uncertain`.

| benh ly | ty le ap luc | Sens | **Spec** | AUROC |
|---|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 1.16 | 0.947 | **0.375** | 0.759 |
| Cardiomegaly | 1.68 | 0.965 | **0.460** | 0.901 |
| Lung Opacity | 4.02 | 1.000 | **0.017** | 0.761 |
| Lung Lesion | 2.68 | 1.000 | **0.000** | 0.721 |
| Edema | 1.02 | 0.896 | **0.675** | 0.881 |
| Consolidation | 1.15 | 0.868 | **0.737** | 0.883 |
| Pneumonia | 0.82 | 0.812 | **0.563** | 0.791 |
| Atelectasis | 5.36 | 1.000 | **0.000** | 0.730 |
| Pneumothorax | 0.5 | 0.580 | **0.795** | 0.786 |
| Pleural Effusion | 1.41 | 0.929 | **0.665** | 0.898 |
| Fracture | 2.2 | 1.000 | **0.000** | 0.442 |
| Support Devices | 4.36 | 1.000 | **0.000** | 0.821 |

Quan he don dieu: ty le ap luc cang lech ve Duong thi specificity cang gan 0.
Model khong hong -- no dang toi uu dung thu duoc giao.
