# Chỉ mục tính mới so với bài báo META-CXR

Cập nhật 2026-10-11. Bài báo gốc: D. Edirisinghe et al., "Chest X-Ray Report
Generation Using Abnormality Guided Vision Language Model", IEEE Access, vol. 13,
2025 (META-CXR). Mọi chi tiết "bài báo" dưới đây đã đối chiếu với văn bản PDF
(mục III–V, Eq. 22, Bảng 2–7, Hình 10–11); chi tiết bài báo không nêu thì ghi
"không nêu". Số của dự án lấy từ artifact trên máy train, nguồn ghi ở từng mục.
Không có văn bản báo cáo hay ID bệnh nhân trong file này.

"Tính mới" ở đây là **so với bài báo META-CXR**, không phải so với toàn bộ tài
liệu: nhiều kỹ thuật lấy từ công trình khác (EVOKE, GradCache, SigLIP, ...) và
được ghi nguồn. Khi đưa vào luận văn, chỉ nhóm A được viết là "cải thiện so với
bài báo"; nhóm B chỉ được viết là "khác/bổ sung", vì chưa có ablation tách riêng.

## Cách phân loại

| Loại | Ý nghĩa | Được viết trong luận văn là |
|---|---|---|
| A | Khác bài báo, **có số đo cho thấy tốt hơn**, đã đưa vào pipeline | đóng góp có bằng chứng |
| B | Khác bài báo, đã đưa vào pipeline, **chưa tách riêng đóng góp** | bổ sung/khác biệt, chưa kết luận tốt hơn |
| C | Phát hiện mới từ phân tích, không phải thay đổi mô hình | đóng góp phân tích |
| D | Đã thử, **không cải thiện** (kết quả âm) | kết quả âm, báo cáo trung thực |
| E | Đề xuất, **chưa làm**, chờ duyệt | hướng phát triển |
| K | Khác bài báo nhưng **không phải tính mới** (phần cứng, giấy phép, sửa lỗi của chính dự án, bài báo không nêu) | ghi ở phần cài đặt/giới hạn |

## Bảng chỉ mục

| ID | Hạng mục | Loại |
|---|---|---|
| TM-01 | Số chính Stage 1 dùng hai ngưỡng theo từng bệnh fit trên val | A |
| TM-02 | Giải mã beam 4 + length penalty 2 | A |
| TM-03 | Làm sạch target FINDINGS v3 (bỏ dòng tiêu đề phiếu chụp) | A |
| TM-04 | GradCache + SigLIP cho pha căn chỉnh 1a | A |
| TM-05 | Đa góc chụp: ViewFusion, MPC, StreamAdapter, view consistency | B |
| TM-06 | PubMedCLIP đọc qua post_layernorm, trừ trung bình token, tiền xử lý riêng | B |
| TM-07 | Mở khóa một phần encoder ở pha 1c | B |
| TM-08 | Chọn checkpoint theo macro recall; loại study không có nhãn CheXpert | B |
| TM-09 | Giải thích Stage 2 (attention rollout + hai cổng kiểm định) | B |
| TM-10 | Đánh giá có khoảng tin cậy bootstrap ghép cặp, mốc so sánh, nhóm con theo số ảnh | B |
| TM-11 | Soft token META-Former sụp thành một vector, sụp ở pha 1a | C |
| TM-12 | Pha 1b làm lệch đầu vào Q-Former; pha 1c phá căn chỉnh, không giúp phân loại | C |
| TM-13 | Soft token giữ bệnh phổ biến, mất bệnh khu trú/hiếm | C |
| TM-14 | Metric NLG thưởng văn phong; LLM không chuyển thông tin phân loại vào văn bản | C |
| TM-15 | Thiếu bối cảnh lâm sàng: lần chụp trước, indication | C |
| TM-16 | Ngưỡng Eq. 22 của bài báo không tái lập được, và làm giảm F1 | C |
| TM-17 | Cờ EVOKE (cắt gradient K/V ảnh phụ, τ MPC 0,5) | D |
| TM-18 | Tối đa 3 ảnh phụ | D |
| TM-19 | Cue Stage 2 theo ngưỡng hai mức | D |
| TM-20 | Finding token học được cho Stage 2 | D |
| TM-21 | Logit-adjusted loss | D |
| TM-22 | ITC ở batch nhỏ; khung nhị phân/mention gate | D |
| TM-23 | Đưa indication + FINDINGS lần chụp trước vào prompt | E |
| TM-24 | Dùng kênh ảnh gốc của MedGemma thay 32 soft token | E |
| TM-25 | Sửa sụp token META-Former; rút gọn/bỏ pha 1c; 1b còn 3 epoch | E |
| TM-26 | Đo CE bằng CheXbert | E |
| TM-27 | MedGemma 1.5 4B QLoRA thay Vicuna-7B | K |
| TM-28 | Sửa stop token của Stage 2 | K |
| TM-29 | Độ dài pha tính theo epoch; gradient checkpointing; bf16 | K |

## A — Có bằng chứng tốt hơn bài báo

**TM-01 — Quy tắc quyết định ba lớp bằng hai ngưỡng theo bệnh.** Bài báo báo
cáo số phân loại bằng argmax (Hình 10, Bảng 5/7) và chỉ dùng ngưỡng Eq. 22 cho
prompt; không nêu ngưỡng fit trên tập nào. Dự án: mỗi bệnh hai ngưỡng t1 ≤ t2
trên p_pos/(p_pos+p_neg), fit trên val (1.808 study) để tối đa weighted F1 ba
lớp; vẫn ba lớp (Negative/Uncertain/Positive). File ngưỡng
`configs/stage1_cutpoints/run_20261005_paper.json`. Test 3.269 study: weighted
P/R/F1 0,827/0,834/0,820 (argmax 0,839/0,781/0,780; bài báo 0,87/0,78/0,73), F1
5 bệnh 0,749 [0,741; 0,757] (argmax 0,669; bài báo 0,701); so với argmax, F1 5
bệnh +0,0805 [+0,0743; +0,0866], macro recall −0,0149 (bootstrap ghép cặp 1.000
lần, seed 16). Phải ghi "không phải giao thức argmax của bài báo" cạnh số chính.
Nguồn: `docs/handoff/PLAN-2026-10-10-cutpoint-headline.md`, commit `3607223`.

**TM-02 — Giải mã beam 4, length penalty 2.** Bài báo không nêu cách giải mã. Test
2.800 study, cùng adapter: beam − greedy BERTScore +0,0083 [+0,0037; +0,0129],
ROUGE-L +0,0150 [+0,0112; +0,0188], CIDEr +0,0439 [+0,0230; +0,0653]; số beam
0,357/0,098/0,137/0,265/0,128/0,369 (BLEU-1/BLEU-4/METEOR/ROUGE-L/CIDEr/
BERTScore). Chọn trên val 300 trước, chạy test một lần. ⚠ Chưa có trong
`generate_stage2_reports.py` (chạy bằng probe). Nguồn:
`docs/handoff/PLAN-2026-10-07-stage2-header-fix.md`.

**TM-03 — Làm sạch target FINDINGS (manifest v3).** Bài báo không mô tả cách lấy
FINDINGS. Dự án phát hiện 9,8% target train bắt đầu bằng dòng tiêu đề phiếu chụp
và 3,4% chỉ có dòng đó; mô hình học lại và 57% báo cáo test chỉ là dòng tiêu đề
(BLEU-1 0,056). Parser v3 bỏ dòng này: 0 báo cáo chỉ-tiêu-đề, greedy BLEU-1
0,285. ⚠ Hai run khác cohort test (2.984 và 2.800 study). Nguồn:
`docs/handoff/PLAN-2026-10-07-stage2-header-fix.md`.

**TM-04 — GradCache + SigLIP cho pha căn chỉnh META-Former (1a).** Bài báo dùng
ITC/ITM/ITG của BLIP-2, không nêu batch. Trên GPU 16 GB, ITC ở batch 8 cho kết
quả ngẫu nhiên trong cả bốn lần đo. Dự án dùng GradCache (Gao et al., 2021) với
batch 128, loss sigmoid kiểu SigLIP (Zhai et al., 2023), cache đặc trưng encoder:
lần đầu ITC vượt ngẫu nhiên — R@5 0,664/0,641, delta_nats +2,93 (val 256 cặp).
Nguồn: D-021, `docs/handoff/PLAN-2026-09-24-meta-former-3phase.md`. Lưu ý TM-11:
căn chỉnh vượt ngẫu nhiên nhưng token vẫn sụp.

## B — Khác/bổ sung, chưa tách riêng đóng góp

**TM-05 — Đa góc chụp.** Bài báo: "focused exclusively on frontal views". Dự án:
một ảnh neo (PA > AP > lateral) + tối đa 1 ảnh phụ, `ViewFusionModule` (query là
token ảnh neo, K/V là ảnh phụ, khởi tạo đồng nhất), `MultiPositiveContrastiveLoss`
và `StreamAdapter` (ý tưởng từ EVOKE, arXiv 2411.10224), `view_consistency_loss`
có lề và cổng độ tự tin. **Chưa có ablation đa góc chụp so với chỉ frontal**; các
biến thể đã thử đều không tốt hơn (TM-17, TM-18).

**TM-06 — Luồng PubMedCLIP.** Bài báo dùng tiền xử lý riêng từng encoder, không
nêu chi tiết token. Dự án: đọc patch qua `post_layernorm` rồi trừ trung bình theo
ảnh (cosine trung bình giữa patch 0,674 → −0,014), giữ token CLS; tiền xử lý
CLIP riêng (D-022). Mỗi encoder giữ số token gốc (196 + 50 + 50 = 296, khớp số
token bài báo mô tả).

**TM-07 — Mở khóa encoder ở pha 1c.** Bài báo chỉ mở "projection heads of the
visual encoders". Dự án mở thêm ResNet50 layer4 và khối 10–11 + post_layernorm của
CLIP. Bằng chứng duy nhất (+0,020 AUROC) đo trong khung nhị phân đã gỡ (D-023),
không dùng được làm bằng chứng hiện tại.

**TM-08 — Chọn checkpoint và lọc nhãn.** Bài báo không nêu tiêu chí chọn
checkpoint. Dự án chọn theo macro recall ba lớp trên val (D-024). Bài báo coi mọi
ô trống là âm tính (giống dự án, D-018), nhưng dự án loại các study không có thông
tin CheXpert nào thay vì biến thành 14 ô âm tính.

**TM-09 — Giải thích Stage 2.** Bài báo chỉ có attention map của MHCAC. Dự án thêm
attention rollout có trọng số gradient trên MedGemma, kèm hai cổng kiểm định
(ảnh lệch study: +0,2160 [+0,1860; +0,2468] NLL, n=100; ngẫu nhiên hóa trọng số:
rho giảm 0,986 → 0,295) và bộ gán nhãn câu `lexicon_v2`. Đo trên adapter cũ (đã
xóa 2026-10-05).

**TM-10 — Độ chặt của đánh giá.** Bài báo báo số điểm, không khoảng tin cậy. Dự
án: bootstrap ghép cặp theo study cho mọi so sánh, mốc so sánh (hằng số, đa số),
nhóm con theo số ảnh, kiểm tra không fit ngưỡng trên test.

## C — Phát hiện mới từ phân tích (2026-10-11)

Số đo bằng `scripts/diagnose_qformer_collapse.py`, `probe_soft_tokens_cached.py`,
`retrieval_report_baselines.py`, `analyze_stage2_generations.py`,
`prior_report_baseline.py`. Chi tiết: `docs/RESEARCH-2026-10-11-stage1-stage2-diagnosis.md`.

**TM-11 — Soft token META-Former sụp thành một vector, ngay ở pha 1a.** Cosine
trung bình giữa 32 token của một study 0,9989 sau 1a (khởi tạo BLIP-2 0,445), số
hướng hiệu dụng 1,002 (khởi tạo 2,94). Từ tầng cross-attention thứ hai, cả 32
query nhìn cùng một token ảnh (entropy 0,11, một token chiếm 0,79 attention). ITM
ở mức ngẫu nhiên suốt 1a (val 0,644–0,670, ngẫu nhiên 0,6365). Run cũ cũng vậy
(0,9999) — lỗi có tính hệ thống của thiết kế/huấn luyện, không phải một run.

**TM-12 — Pha 1b và 1c.** Q-Former bị đóng băng ở 10% đầu 1b, trong khi
StreamAdapter (không có LayerNorm phía sau ở PubMedCLIP, Swin) đẩy norm token
19,6 → 1.713 và 12 → 413, cosine giữa các token trong một ảnh 0,01 → 0,45–0,71.
Pha 1c (1 epoch, batch 8): R@5 0,65 → 0,17; val AUROC 0,778 (1b tốt nhất) → 0,771,
macro recall 0,492 → 0,486; gradient căn chỉnh lớn gấp 7–35 lần gradient phân loại
trên projector chung, cosine ≈ 0. Pha 1b đạt tốt nhất ở epoch thứ ba (chỉ số 2) trong 5
epoch; hai epoch cuối không cải thiện.

**TM-13 — Thông tin bệnh trong soft token.** Linear probe (fit 50.000 study train,
chấm 2.780 study test): AUROC trung bình 0,713 so với MHCAC 0,745 trên cùng study;
ngang MHCAC ở bệnh phổ biến (Edema 0,848/0,843, Pleural Effusion 0,896/0,899),
mất ở bệnh khu trú/hiếm (Fracture 0,488 = ngẫu nhiên, Pleural Other 0,635/0,833).

**TM-14 — Vì sao phân loại tốt hơn không làm NLG tốt hơn.** Test 2.800 study:
một báo cáo cố định cho mọi study đạt ROUGE-L 0,238, BERTScore 0,340, cao hơn
chép báo cáo train có nhãn đúng hoàn toàn (0,230; 0,334). Chép báo cáo train có
nhãn khớp dự đoán MHCAC đạt CE lexicon 0,294, cao hơn báo cáo LLM sinh (0,239):
LLM không chuyển thông tin phân loại vào văn bản. Mô hình không bao giờ nhắc
Fracture, Lung Lesion, Pleural Other, Enlarged Cardiomediastinum. Khớp với tài
liệu: PromptMRG (CE F1 0,370 → 0,444, BLEU-4 0,116 → 0,106), RaDialog (CE 26,1 →
39,4, BLEU-4 9,0 → 9,5).

**TM-15 — Thiếu bối cảnh lâm sàng.** 90,0% FINDINGS test có ngôn ngữ so sánh thời
gian; 89,6% study test có lần chụp trước với FINDINGS; 91,7% có INDICATION/HISTORY.
Mô hình không có những thông tin này và tự viết câu so sánh trong 51% báo cáo mà
bản gốc không so sánh. Bài báo không dùng các nguồn này.

**TM-16 — Ngưỡng Eq. 22.** Bài báo không nêu ngưỡng fit trên tập nào (đường ROC duy
nhất, Hình 5, vẽ trên test) và không công bố file ngưỡng. Dùng Eq. 22 fit trên val
của dự án, chấm test: weighted F1 0,702, F1 5 bệnh 0,637 — thấp hơn argmax (0,780;
0,669).

## D — Đã thử, không cải thiện

| ID | Thử nghiệm | Kết quả (test, bootstrap ghép cặp) | Nguồn |
|---|---|---|---|
| TM-17 | Cắt gradient K/V ảnh phụ + τ MPC 0,5 (như EVOKE) | F1 5 bệnh −0,0160 [−0,0193; −0,0122]; weighted F1 −0,0033 | `PLAN-2026-10-09-evoke-flags.md` |
| TM-18 | Tối đa 3 ảnh phụ | AUROC −0,0085 [−0,0132; −0,0028]; F1 5 bệnh −0,0114 | `PLAN-2026-10-06-multi-aux-views.md` |
| TM-19 | Cue Stage 2 theo ngưỡng hai mức | beam: mọi chỉ số ngang; greedy ROUGE-L −0,0030 [−0,0051; −0,0009] | `PLAN-2026-10-10-cutpoint-headline.md` |
| TM-20 | 13 finding token học được | không cải thiện; BERTScore −0,0075 so với không cue | `PLAN-2026-09-10-mention-finding-tokens.md` |
| TM-21 | Logit-adjusted loss (τ 1,0 / 0,5) | mô phỏng: τ 1,0 sụp về Uncertain; không áp dụng | `PLAN-2026-10-01-logit-adjusted-loss.md` |
| TM-22 | ITC batch 8 (4 lần); khung nhị phân + mention gate | ITC = ngẫu nhiên; khung nhị phân đã gỡ (D-023) | `PLAN-2026-08-19-itc-temp-probe.md` |

## E — Đề xuất chưa làm (chờ duyệt)

**TM-23 — Bối cảnh trong prompt Stage 2:** INDICATION/HISTORY và FINDINGS của lần
chụp trước. Bằng chứng: TM-15; RaDialog thêm indication BLEU-4 9,5 → 14,8, ROUGE-L
27,1 → 31,6; MAIRA-2 bỏ lần chụp trước lúc suy luận ROUGE-L −28,9%.

**TM-24 — Kênh ảnh gốc của MedGemma** (MedSigLIP, 256 token, đã train trên 231k
ảnh MIMIC-CXR) thay 32 soft token sụp. Bằng chứng: TM-11/TM-13; MedGemma 4B PT
RadGraph F1 29,5, MedGemma 1.5 4B 27,2 không fine-tune; MAIRA-2/LLaVA-Rad dùng toàn
bộ patch token qua MLP.

**TM-25 — Sửa Stage 1 nếu giữ META-Former:** chuẩn hóa từng luồng trước Q-Former và
sau StreamAdapter, phạt độ giống nhau giữa các query, bỏ ITM, 1a 2 epoch, thay 1c
bằng pha căn chỉnh riêng (GradCache, MHCAC đóng băng) hoặc bỏ 1c, 1b 3 epoch.

**TM-26 — CE bằng CheXbert** (thư viện `f1chexbert`, như MLRG và nhiều bài khác) thay
cho CE lexicon xấp xỉ.

## K — Khác bài báo nhưng không phải tính mới

**TM-27 — LLM.** Bài báo: Vicuna-7B LoRA khởi tạo từ RaDialog (đã fine-tune trên
MIMIC-CXR). Dự án: MedGemma 1.5 4B IT, QLoRA NF4, LoRA r8/α16 như bài báo, prompt
bài báo nguyên văn (D-027). Đổi vì phần cứng 16 GB và giấy phép.

**TM-28 — Stop token.** Sửa lỗi của chính dự án (Stage 2 bỏ qua `<end_of_turn>`,
78% token sinh sau điểm kết thúc); CIDEr val n=100 0,0107 → 0,2369.

**TM-29 — Cài đặt huấn luyện.** Bài báo 20.000 bước biểu diễn + 5.000 bước sinh,
RTX 4070 16 GB. Dự án: độ dài theo epoch (quyết định của user), RTX 5060 Ti 16 GB,
bf16, gradient checkpointing Q-Former, label smoothing ITC 0,1.

## Phần giống bài báo (để đối chiếu)

Ba encoder đóng băng cùng họ (BioViL-T ResNet50, PubMedCLIP ViT, MedCLIP Swin);
chiếu về 1408 chiều rồi nối theo trục token; che ngẫu nhiên 10% đặc trưng mỗi
encoder; META-Former khởi tạo từ Q-Former BLIP-2 với ITC/ITM/ITG; MHCAC 6 tầng,
8 common expert token, 14 expert token chuyên biệt, dropout 0,2, loss CE có trọng
số + contrastive + orthogonality + sparsity, không label smoothing; ba lớp
P/N/U, ô trống = âm tính; lịch huấn luyện ba pha (1a căn chỉnh, 1b chuyển sang
MHCAC với LR 5e-5 → 2e-4 → 1e-5 trong 5 epoch, 1c huấn luyện chung); Stage 2
chiếu đầu ra META-Former vào LLM, prompt gồm query + danh sách P/N/U theo ngưỡng
Eq. 22; LoRA r8/α16; tập chia chính thức (val 1.808, test 3.269 study); bộ chỉ số
BLEU/METEOR/ROUGE-L/CIDEr/BERTScore và F1 5 bệnh.
