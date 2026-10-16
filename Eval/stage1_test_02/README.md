# Stage 1 — Đánh giá trên tập TEST · **phiên bản 02**

Run `run_20260820_ft` · `checkpoint_best.pth` (**epoch 9**) · config `mimic_cxr_full.yaml` @ `fec606b`

Tổng hợp 2026-08-21. So sánh trực tiếp với phiên bản 01 (`Test/stage1_test/`) trên
**cùng 3,269 study test** — nhãn khớp từng ô, nên mọi so sánh dưới đây là **ghép cặp**.

---

## 0. Phiên bản này khác gì 01

| | 01 — `run_20260819_xmpoff` | **02 — `run_20260820_ft`** |
|---|---|---|
| Vision encoder | **đóng băng hoàn toàn** (181.3M) | **mở băng phần đỉnh**: 31.85M ở `init_lr_enc` 1e-5 |
| | | ResNet50 `layer4` + `projector`, CLIP vision block 10–11 + `post_layernorm` |
| | | `keep_batchnorm_eval: true` |
| kappa (classification) | 1–4 theo nhãn | **1 cho tất cả** (inverse frequency thuần) |
| kappa_gate | 2 | **1** |
| Best epoch | 6 (giữa run) | **9 — epoch CUỐI** |
| Thời gian | 12h35m | 15h33m (+24%) |
| Epoch trung bình | 1.13 h | 1.55 h (**+37%**) |
| `max mem` | 8,526 MiB | 8,976 MiB (+5.3%) |
| Restart / kernel fault | 0 / 0 | **0 / 0** |

**Val loss vẫn đang giảm ở epoch cuối** — 1.8939 → 1.8925 → 1.8939 → 1.8913 → **1.8843**.
Phiên bản 01 thì phẳng và peak ở giữa (best 1.8882 ở ep6, và ep9 là 1.8902).
⚠ Hai giá trị loss **không so sánh trực tiếp được** vì kappa đổi thang hàm loss. Điều
so sánh được là **hình dạng đường cong**: 01 đã bão hòa, 02 thì chưa.

---

## 1. Số chính

`study_presence` + `marginal_presence`, ngưỡng calibrate trên val bằng
`--selection plateau --plateau-fraction 0.95 --min-positive 5`, bootstrap 1,000 lần, seed 42.

| Metric | **Phiên bản 02** | 95% CI | Phiên bản 01 |
|---|---:|:---:|---:|
| `macro_auroc` | **0.7643** | [0.7545, 0.7740] | 0.7441 |
| `positive_macro_f1` | **0.3542** | [0.3418, 0.3647] | 0.3397 |
| `macro_auprc` | **0.3203** | [0.3108, 0.3362] | 0.3004 |
| `positive_macro_recall` | **0.5373** | [0.5184, 0.5555] | 0.5006 |
| `positive_macro_precision` | **0.2931** | [0.2816, 0.3038] | 0.2776 |
| `macro_specificity` | 0.8020 | – | 0.7901 |
| `positive_micro_f1` | 0.4752 | [0.4648, 0.4854] | 0.4685 |

Baseline tầm thường trên cùng framing, để đối chiếu: `all_positive` **0.2280**,
`all_negative` **0.0000**, `majority_class` **0.0000**, `threshold_half` 0.2922.

---

## 2. So sánh ghép cặp với phiên bản 01 — kết quả then chốt

Bootstrap **ghép cặp** 2,000 lần trên cùng study (không phải so hai CI rời):

| | Δ (02 − 01) | 95% CI | P(Δ>0) | |
|---|---:|:---:|---:|---|
| **AUROC** | **+0.0201** | [+0.0136, +0.0263] | 1.000 | ✅ có ý nghĩa |
| **F1** | **+0.0144** | [+0.0053, +0.0236] | 1.000 | ✅ có ý nghĩa |
| **Precision** | **+0.0154** | [+0.0073, +0.0238] | 1.000 | ✅ có ý nghĩa |
| **Recall** | **+0.0368** | [+0.0205, +0.0535] | 1.000 | ✅ có ý nghĩa |

**Cả bốn cùng tăng — KHÔNG có đánh đổi.** Đây là khác biệt về bản chất so với lần
chỉnh ngưỡng ở phiên bản 01, vốn đổi 0.021 precision lấy 0.075 recall. Ở đây mô hình
thật sự tốt hơn, không phải dịch điểm vận hành.

**AUROC tăng trên 14/14 nhãn.** AUROC là đại lượng không phụ thuộc ngưỡng, nên đây là
bằng chứng trực tiếp rằng **chất lượng xếp hạng** đã cải thiện — tức trần của F1 đã
được nhấc lên, chứ không phải F1 được vay từ recall.

---

## 3. Trần precision đã dịch

Ở phiên bản 01 tôi kết luận "precision không mua được bằng ngưỡng, trần đó thuộc về
mô hình". Trần đó vừa nhấc lên:

| sàn precision fit trên val | 01 → test P | **02 → test P** | 01 → test R | **02 → test R** |
|---|---:|---:|---:|---:|
| không (F1-optimal plateau) | 0.2778 | **0.2931** | 0.5044 | **0.5358** |
| 0.40 | 0.3330 *(hụt sàn)* | **0.4227** ✓ | 0.3506 | 0.3810 |
| 0.50 | 0.3975 | **0.4584** | 0.2329 | 0.2591 |
| 0.60 | 0.3924 | **0.4906** | 0.1410 | 0.1506 |
| 0.70 | 0.4074 | **0.5240** | 0.0815 | 0.0843 |

Bão hòa từ ~0.40 lên **~0.52**. Ở sàn 0.40, phiên bản 01 còn không giao nổi sàn
(0.3330); phiên bản 02 giao được 0.4227.

⚠ Vẫn phải nói rõ: ép precision cao vẫn phá recall nặng. Sàn 0.70 chỉ còn recall 0.084.
Điểm F1-optimal vẫn là điểm vận hành nên dùng.

---

## 4. Từng bệnh lý

| Bệnh lý | Prev. | AUROC 01 | **AUROC 02** | ΔAU | F1 01 | **F1 02** | ΔF1 | P 02 | R 02 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Support Devices † | 0.344 | 0.8783 | **0.8838** | +0.0055 | 0.7394 | **0.7519** | +0.0125 | 0.6829 | 0.8364 |
| Pleural Effusion | 0.329 | 0.8682 | **0.8760** | +0.0077 | 0.6973 | **0.7137** | +0.0164 | 0.6378 | 0.8101 |
| Pleural Other | 0.019 | 0.7867 | **0.8667** | **+0.0800** | 0.0963 | 0.1242 | +0.0279 | 0.0674 | 0.7937 |
| Pneumothorax | 0.034 | 0.8283 | **0.8585** | **+0.0302** | 0.2869 | **0.3400** | **+0.0531** | 0.2713 | 0.4554 |
| Edema | 0.209 | 0.8245 | **0.8353** | +0.0108 | 0.5439 | **0.5619** | +0.0180 | 0.5153 | 0.6179 |
| No Finding † | 0.174 | 0.8049 | 0.8081 | +0.0032 | 0.5025 | 0.5068 | +0.0043 | 0.5235 | 0.4912 |
| Cardiomegaly | 0.265 | 0.7595 | **0.7694** | +0.0100 | 0.5377 | 0.5410 | +0.0033 | 0.3971 | 0.8487 |
| Lung Lesion | 0.038 | 0.7261 | **0.7655** | **+0.0394** | 0.1525 | 0.1833 | +0.0308 | 0.1106 | 0.5366 |
| Consolidation | 0.067 | 0.7446 | 0.7483 | +0.0038 | 0.2228 | 0.2433 | +0.0205 | 0.2618 | 0.2273 |
| Atelectasis | 0.224 | 0.7229 | **0.7379** | +0.0150 | 0.4778 | 0.4795 | +0.0017 | 0.3595 | 0.7196 |
| Lung Opacity | 0.318 | 0.6915 | **0.7099** | +0.0184 | 0.5355 | 0.5498 | +0.0144 | 0.4510 | 0.7042 |
| Pneumonia | 0.102 | 0.6993 | 0.7046 | +0.0053 | 0.2990 | 0.2922 | **−0.0068** | 0.2981 | 0.2866 |
| Fracture | 0.027 | 0.6651 | 0.6670 | +0.0018 | 0.1293 | 0.0933 | **−0.0360** | 0.0606 | 0.2022 |
| Enlarged Cardiomediastinum | 0.046 | 0.6126 | **0.6321** | +0.0195 | 0.0980 | 0.1276 | +0.0296 | 0.0862 | 0.2450 |

† meta-label, loại khỏi macro.

**AUROC: 14/14 tăng. F1: 12/14 tăng**, giảm ở Pneumonia (−0.0068) và Fracture (−0.0360).
Fracture chỉ có 18 mẫu dương trên val nên ngưỡng của nó nhiễu nhất; AUROC của nó vẫn
tăng nhẹ (+0.0018), tức phần giảm là do ngưỡng chứ không phải mô hình kém đi.

Bốn nhãn khỏe nhất (AUROC ≥ 0.83): Support Devices, Pleural Effusion, Pleural Other,
Pneumothorax. Yếu nhất: Enlarged Cardiomediastinum 0.6321, Fracture 0.6670.

⚠ Precision vẫn dưới 0.25 ở bốn nhãn (Pleural Other 0.067, Enlarged Cardiomediastinum
0.086, Lung Lesion 0.111, Fracture 0.061). **Chưa dùng lâm sàng được ở những nhãn này.**

---

## 5. Mẫu số cũ — chỉ để đối chiếu với bài báo gốc

| | 01 | **02** |
|---|---:|---:|
| `positive_macro_f1` | 0.8717 | 0.8841 |
| `macro_auroc` | 0.7857 | **0.7985** |
| `macro_specificity` | 0.4197 | 0.4536 |

⚠ **Không dùng F1 ở đây làm số chính.** Trên mẫu số này, chiến lược "báo dương hết"
đã đạt 0.8397 — mô hình chỉ hơn 0.044. Lý do đầy đủ ở `Test/stage1_test/README.md` mục 2.
Con số đáng đọc ở bảng này là `macro_auroc` **0.7985**, cũng tăng +0.0128.

---

## 6. Kết luận và bước tiếp theo

**Mở băng encoder có tác dụng thật và đo được.** AUROC +0.0201 trên 14/14 nhãn, cả
precision lẫn recall cùng tăng, trần precision nhấc từ 0.40 lên 0.52. Chi phí là +37%
thời gian mỗi epoch.

**Không tách được đóng góp của kappa và của mở băng** — hai thay đổi đi cùng một run.
Muốn biết riêng phần nào thì cần một run ablation nữa (~15.5 giờ). Nhưng có một lập
luận gián tiếp: kappa chỉ dịch điểm vận hành, mà ngưỡng lại được calibrate trên val
sau đó, nên phần lớn tác dụng của kappa bị hấp thụ. **AUROC tăng — thứ kappa về nguyên
tắc không đổi được — nên phần cải thiện gần như chắc chắn đến từ mở băng encoder.**

Hai việc đáng làm tiếp, theo thứ tự:

1. **Train dài hơn.** Best epoch là epoch **cuối**, val loss vẫn đang giảm. Phiên bản
   này chưa hội tụ. 15 epoch thay vì 10 là bước rẻ nhất còn lại (~23 giờ).
2. **Mở băng sâu hơn** (thêm `layer3` + CLIP block 8–9) với learning rate thấp hơn nữa.
   VRAM còn dư 44% nên còn chỗ.

Chưa làm và cần cân nhắc riêng: `lambda_mention_conditioned_cls`, vốn train thẳng đại
lượng joint mà metric này đang chấm.

---

## 7. Cấu trúc thư mục

```
Test/stage1_test_02/
├── README.md                          ← file này
├── summary_all.json                   ← aggregate + CI + bootstrap ghép cặp + sweep precision
├── version_comparison.csv             ← 01 vs 02, cả hai mẫu số
├── per_label_vs_run01.csv             ← ΔAUROC / ΔF1 từng nhãn
├── thresholds_presence_marginal_plateau.json  ← ngưỡng ĐANG DÙNG (val, plateau, minpos5)
├── thresholds_masked_f1.json          ← ngưỡng cho arm mẫu số cũ
├── predictions/
│   ├── val_best.npz / test_best.npz   ← epoch 9, CÓ mention_probabilities
├── eval_presence_plateau/             ← ★ ĐÁNH GIÁ CHÍNH (+ plots, confusion matrices)
└── eval_masked_polarity/              ← mẫu số cũ, để đối chiếu
```

Lệnh tái tạo (không cần GPU):

```bash
python scripts/calibrate_thresholds.py \
    --predictions Test/stage1_test_02/predictions/val_best.npz \
    --label-framing study_presence --score marginal_presence \
    --objective f1 --uncertain-policy ignore_uncertain \
    --min-positive 5 --selection plateau --plateau-fraction 0.95 \
    --output Test/stage1_test_02/thresholds_presence_marginal_plateau.json

python scripts/evaluate_stage1.py \
    --predictions Test/stage1_test_02/predictions/test_best.npz \
    --thresholds Test/stage1_test_02/thresholds_presence_marginal_plateau.json \
    --label-framing study_presence --score marginal_presence \
    --uncertain-policy ignore_uncertain --bootstrap-samples 1000 --split test \
    --output-dir Test/stage1_test_02/eval_presence_plateau
```

---

## 8. Bảo mật dữ liệu

`Test/` nằm trong `.gitignore`. File `.npz` chứa `sample_keys` là định danh study của
MIMIC-CXR — **dữ liệu PhysioNet credentialed**. Không commit, không push, không đưa lên
dịch vụ ngoài. Chỉ các con số tổng hợp trong file này là an toàn để trích vào luận văn.
