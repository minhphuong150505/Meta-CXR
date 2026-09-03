> Source: `scripts/probe_soft_tokens.py` (215 dòng)
> Status: 🔬 DIAGNOSTIC — read-only, không train, không ghi checkpoint
> Last verified against source: 2026-09-03

# `scripts/probe_soft_tokens.py`

## Purpose

Trả lời một câu hỏi **trước khi** đốt ~70 giờ GPU cho arm C
(`meta_cxr_native_qformer_guided`): **32 soft token của Q-Former có mang tín
hiệu y khoa nào không?**

Câu hỏi này không hiển nhiên. Stage 1 chạy với `lambda_itc/itm/lm = 0.0`, nên
đường ảnh của Q-Former chưa từng nhận một gradient nào — cross-attention (60
tensor), query FFN (72) và `query_tokens` (1) **bit-identical** với khởi tạo
BLIP-2, vốn được fit cho đặc trưng EVA-ViT trên ảnh tự nhiên. Ở đây chúng đọc
246 token BioViL-T + PubMedCLIP trên ảnh X-quang: **khớp chiều** (đều 1408, vì
lớp FC chiếu về đúng độ rộng EVA-ViT-g) nhưng **khác không gian ngữ nghĩa**.
Nên soft token là một **phép đọc cố định** trên đặc trưng Stage-1 đã được huấn
luyện đầy đủ, và việc phép đọc đó có giữ lại gì dùng được hay không là câu hỏi
thực nghiệm, không phải điều kiến trúc bảo đảm.

## Tại sao linear probe là dụng cụ đúng

`img_proj` — lớp **duy nhất được train** giữa soft token và MedGemma — bản thân
là một ánh xạ tuyến tính. Nếu một linear probe không khôi phục được nhãn thì
`img_proj` cũng không, và kênh soft token chỉ là trang trí.

## Đo cái gì

| Chỉ số | Cách tính | Ý nghĩa |
|---|---|---|
| Macro AUROC | Logistic regression `C=0.01`, 5-fold CV trên soft token mean-pool `[N, 768]` | Tín hiệu tuyến tính còn lại |
| AUROC nhãn xáo trộn | Cùng probe, nhãn `rng.permutation` | Đối chứng rò rỉ — **phải ≈ 0,50** |
| Cosine **giữa** các study | Trung bình off-diagonal của ma trận cosine `[N, N]` trên vector pooled | ≈1,0 ⇒ mọi ảnh cho cùng một thứ, tức hằng số |
| Cosine **trong** một study | Trung bình off-diagonal `[32, 32]` mỗi study | ≈1,0 ⇒ 32 token làm việc của 1 |

Hai chỉ số cosine tồn tại vì repo này đã gặp đúng dạng hỏng đó: patch token thô
của PubMedCLIP có hướng DC cố định, cosine cặp trung bình **0,674** (BioViL lành
mạnh: **0,0017**), khiến cả luồng hoạt động như một bias hằng số.

Khung nhãn là **`study_presence`**: blank / negative / uncertain đều nghĩa là
"không hiện diện". Đây là khung duy nhất khiến con số so sánh được với macro
AUROC 0,7643 của MHCAC.

## Kết quả đã đo (2026-09-03, 1.172 study val, `run_20260820_ft/checkpoint_best`)

| | |
|---|---:|
| **Macro AUROC, 10 findings** | **0,6847** |
| Cùng probe, nhãn xáo trộn | **0,4838** |
| MHCAC tự nó (test, `marginal_presence`) | 0,7643 |
| Cosine giữa các study | **+0,4368** |
| Cosine trong một study | **+0,8022** |

**Verdict: GO.** Soft token giữ được ~90% macro AUROC của MHCAC dù cross-attention
chưa từng được huấn luyện. Nhãn mạnh nhất: Edema 0,8146 · Pneumothorax 0,7953 ·
Pleural Effusion 0,7946. Yếu nhất: Lung Lesion 0,5619.

⚠ Cosine trong-study **0,8022** nghĩa là 32 token khá dư thừa — số chiều hiệu
dụng thấp hơn 32 nhiều. Không suy sụp, nhưng phải nói ra bên cạnh kết quả arm C.

⚠ Đây là **val**, mean-pool, và tương đương `q` (không có mention gate), trong
khi 0,7643 là **test** với threshold đã calibrate và `marginal_presence`. Con số
"90%" là chỉ báo, không phải so sánh chặt.

## Hai cái bẫy đã trả giá

1. **`image_path` trong record là ĐƯỜNG DẪN TUYỆT ĐỐI** (dataset đã join
   `vis_root`), còn trong split CSV là tương đối. Join trực tiếp trả về **0
   dòng và không báo lỗi**. Script join theo **basename** (chính là `dicom_id`,
   duy nhất toàn MIMIC-CXR).
2. **Split CSV KHÔNG mang 14 cột nhãn** — chỉ có cờ `has_chexpert_label`. Nhãn
   nằm ở `mimic-cxr-2.0.0-chexpert.csv.gz`, join qua `study_id`. Vì thế có
   `--chexpert` riêng.

## Chi phí thực đo

`build_stage1_records` chạy **~50 study/s** trên RTX 5060 Ti (1.500 study trong
~30 giây), **không phải** 0,1 s/study như ước tính cũ trong `CLAUDE.md`. Toàn bộ
220k study train do đó là khoảng **73 phút**, không phải 6–9 giờ.

Cache dùng lại được: lần chạy thứ hai không dựng lại model, đọc thẳng
`.sensitive_stage1_cache/*.pt`.

## Privacy

Chỉ in **số tổng hợp** — không bao giờ in text báo cáo, identifier hay đường dẫn
ảnh. `--report` ghi JSON cùng tính chất, an toàn để dán vào tóm tắt. ⚠ Nhưng
`--output-dir` chứa `.sensitive_stage1_cache/` với `ref` (findings text) và
`image_path` — thư mục đó **là dữ liệu bệnh nhân**, để ngoài repo.

## Callers

Không có. CLI độc lập, gọi tay hoặc từ `~/run_probe.sh` trên training host.

## Callees

`training/train_eval_figure9_llm_variants_200.build_stage1_records` ·
`training/run_context.Stage1Context` · sklearn `LogisticRegression` /
`StratifiedKFold` / `roc_auc_score`

## Related documentation

[`scripts/_index.md`](_index.md) ·
[`check_itc_gate.py.doc.md`](check_itc_gate.py.doc.md) — cổng rẻ tương tự cho ITC ·
[DECISIONS.md](../_meta/DECISIONS.md)

← [Về `scripts/`](_index.md)
