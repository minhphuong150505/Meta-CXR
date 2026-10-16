# Tra cứu phương pháp mới (cuối 2025 → 10/2026) áp dụng được cho pipeline META-CXR

Ngày tra cứu: 2026-10-01. Phạm vi: các bài từ cuối 2025 đến nay, đối chiếu với các
kết quả đã đo được ghi trong `CLAUDE.md`.

⚠ Tài liệu này dựa trên tóm tắt và trích đoạn của các bài báo, chưa đọc kỹ toàn văn
từng bài. Chưa có phương pháp nào ở đây được chạy thử trên máy huấn luyện.

## Bối cảnh: dự án đang kẹt ở đâu

Theo các phép đo đã ghi trong `CLAUDE.md`:

- **Stage 2 là điểm nghẽn chính.**
  - Fine-tune chưa thắng zero-shot (arm A so với zero-shot trên cùng 3.102 study).
  - Sau khi sửa stop token, câu sinh ra **quá ngắn**: trung vị 24 từ, báo cáo tham chiếu là 52 từ.
  - Cue từ Stage 1 không giúp gì, đã xác nhận 7 lần, kể cả khi đổi kênh sang finding tokens.
  - Chỉ có metric từ vựng. Không có metric lâm sàng (không CheXbert, không RadGraph).
- **Stage 1** chạy ba lớp N/P/U (D-023) và chọn checkpoint theo `macro_recall`. Logit adjustment đã được cài nhưng không dùng (D-025).
- Phần cứng: **một RTX 5060 Ti 16 GB** (torch dùng được 15,48 GiB).

## Ưu tiên 1: Stage 2, RL với phần thưởng lâm sàng (GRPO)

Đây là xu hướng lớn nhất giai đoạn này. Fine-tune có giám sát (SFT) chỉ tối ưu
likelihood từ vựng. Các bài mới tối ưu thẳng độ đúng lâm sàng của cả báo cáo.

| Bài | Ý chính | Tài nguyên | Mức phù hợp |
|---|---|---|---|
| **OraPO** (arXiv 2509.18600) | GRPO với reward FactScore: tách các "atomic fact" lâm sàng rồi tính F-beta. Khi cả nhóm rollout được reward 0 (~30% lúc đầu), nó chạy một bước DPO nhỏ ưu tiên báo cáo gốc hơn các rollout hỏng | Qwen2.5-VL-3B, **1.000 mẫu**, 4×A10 24 GB. MIMIC-CXR: F1 0,357, recall 0,891. Không thấy công bố code | **Cao nhất.** Ít dữ liệu, máy nhỏ, gần với cấu hình MedGemma 4B QLoRA trên card 16 GB |
| **MRG-R1** (arXiv 2512.16145) | GRPO, reward MCCS: cosine giữa vector CheXbert 14 nhãn của báo cáo sinh ra và báo cáo gốc, có ngưỡng margin | HuatuoGPT-Vision-7B + LoRA, 4 rollout/mẫu, 222,8k mẫu MIMIC. CE F1 40,39 so với 27,80 của R2GenCMN | Cao, nhưng xem rào cản 2 bên dưới |
| **CXRMate-2** (arXiv 2604.18967) | SFT 5 epoch rồi GRPO 2 epoch. Reward = 0,3 RaTEScore + 0,3 CXR-BERT + 0,3 BERTScore + **0,1 ARN** (phạt n-gram lặp) | LLaMA 3.2 3B, RAD-DINO đóng băng, Q-Adapter 1.369 → 128 token, 8×MI300X. Có code | Kiến trúc gần route `meta_cxr_qformer`. Thành phần ARN nhắm đúng vấn đề lặp từ |
| **PR-controllable RRG** (arXiv 2606.21447) | Reward = λ·precision + (1−λ)·recall, kèm module PRC-AdaLN điều khiển theo λ | Nền R2Gen, 16×A100. MIMIC CE F1 0,571, tốt nhất ở λ = 0,3. Có code | Khớp chủ đề đánh đổi precision/recall của dự án, nhưng tốn tài nguyên |
| **EditGRPO** (arXiv 2509.22812) | GRPO có bước chỉnh sửa rollout sau khi sinh | Chưa đọc chi tiết | Tham khảo |
| **UniRG** (Microsoft, arXiv 2601.17151) | GRPO với reward tổng hợp gồm luật, model và LLM | Quy mô lớn | Tham khảo |

### Cách áp dụng đề xuất

1. Khởi đầu từ adapter Stage 2 hiện có (arm A `medgemma_direct`, hoặc arm C).
2. Chạy GRPO với nhóm 4 rollout trên vài nghìn study train. Theo OraPO, 1.000 mẫu đã đủ cho kết quả.
3. Reward gồm ba phần:
   - **thành phần lâm sàng**, so khớp từng finding;
   - **thành phần độ dài**, để sửa lỗi báo cáo quá ngắn (24 so với 52 từ);
   - **thành phần ARN**, phạt n-gram lặp.
4. Thêm cơ chế dự phòng của OraPO: khi tỷ lệ nhóm có reward 0 cao, chạy một bước DPO "báo cáo gốc > rollout hỏng".

### Rào cản cần quyết định

1. **Repo không có bộ gán nhãn lâm sàng** (không CheXbert, không RadGraph; xem
   `training/evaluation/clinical.py`). Không có nó thì không có reward lâm sàng, và
   cũng không đo được RL có giúp hay không. **Đây là bước phải làm trước tiên**: cài
   CheXbert hoặc RaTEScore, vừa làm reward vừa làm thước đo.
2. **MRG-R1 gộp Uncertain vào Positive (=1)**, chạm vào quy tắc "ba lớp, không nhị
   phân" (D-023). Có thể thiết kế reward giữ đủ N/P/U, ví dụ so khớp từng ô theo 3
   lớp. **Đây là quyết định của người dùng**, không tự chọn.
3. **VRAM:** GRPO cần sinh 4 rollout mỗi bước. Với báo cáo ngắn (~60–160 token) và
   MedGemma 4B NF4 thì có thể vừa 15,48 GiB, nhưng phải đo bằng smoke run trước.

## Ưu tiên 2: DPO trên chính các lỗi đã đo, rẻ hơn RL

- Tạo cặp ưu tiên: báo cáo gốc là "chosen", câu sinh ra bị lặp hoặc quá ngắn là
  "rejected". Chạy DPO trên adapter hiện có. Không cần reward model.
- Bằng chứng: bài DPO về ảo giác "prior exam" (arXiv 2406.06496) giảm loại lỗi đó
  3,2–4,8 lần mà không làm giảm độ đúng lâm sàng. **CheXPO** (arXiv 2507.06959) dùng
  cặp phản thực tế (counterfactual) cho VLM X-quang ngực.
- Ưu điểm: rẻ, ổn định, nhắm đúng các lỗi đã đo (lặp 28–85% ở chế độ stop cũ, quá ngắn ở chế độ stop mới).

## Ưu tiên 3: Stage 1, thử encoder

- **RAD-DINO**, theo bài latent fusion (arXiv 2609.09185), với encoder đóng băng hoàn toàn:
  - RAD-DINO một mình: AUROC **0,827**;
  - BioViL-T một mình: **0,805**;
  - ghép RAD-DINO + BioViL-T (hybrid fusion): **0,840**, mAP 0,467.
  - Repo đã có cờ `model.encoders.raddino`, nên ablation bật RAD-DINO thay hoặc thêm vào PubMedCLIP chủ yếu là thay đổi cấu hình.
  - ⚠ Bài này dùng **U-Ones**, tức nhị phân. Con số **không so được** với giao thức 3 lớp của dự án, chỉ cho thấy RAD-DINO là encoder mạnh hơn.
- **CheXFound** (arXiv 2502.05142): ViT-L pretrain trên ~1 triệu ảnh CXR, vượt RAD-DINO và EVA-X trên CXR-LT 24 (AUPRC 0,209). ViT-L tốn VRAM, cần đo trên card 16 GB trước.
- **RadJEPA** (arXiv 2601.15891): mới hơn, ít được kiểm chứng độc lập.

## Không khuyến nghị cho dự án này

- **Med-AR** (arXiv 2609.29156): 8×H100, 1,4 triệu ảnh nội bộ.
- **UniRG**: quy mô quá lớn.
- **ECHO** (arXiv 2604.09450): chỉ tăng tốc sinh báo cáo, không tăng chất lượng.
- **CXRMate-2, chạy toàn bộ**: 8×MI300X. Chỉ nên mượn công thức reward.

## Lưu ý về đánh giá

- **Reporting Practice Matters** (arXiv 2609.19093): chỉ riêng việc chọn báo cáo tham chiếu nào đã làm thay đổi điểm đánh giá.
- Mọi so sánh mới phải dùng cùng cohort (`--restrict-to`) và bootstrap ghép cặp, như quy ước hiện có của repo.
- Metric lâm sàng thiếu thì ghi là "không có", không ghi là 0.

## Thứ tự đề xuất

1. Thêm thước đo lâm sàng (CheXbert hoặc RaTEScore). Đây là điều kiện tiên quyết cho mọi bước sau.
2. DPO trên các lỗi lặp và quá ngắn.
3. GRPO kiểu OraPO, reward gồm lâm sàng + độ dài + ARN.
4. Song song: ablation RAD-DINO cho Stage 1.

## Nguồn

- [CXRMate-2](https://arxiv.org/html/2604.18967)
- [OraPO](https://arxiv.org/html/2509.18600)
- [MRG-R1](https://arxiv.org/html/2512.16145v2)
- [Precision Recall Controllable RRG](https://arxiv.org/html/2606.21447)
- [EditGRPO](https://arxiv.org/pdf/2509.22812)
- [UniRG (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/unirg-scaling-medical-imaging-report-generation-with-multimodal-reinforcement-learning/)
- [UniRG paper](https://arxiv.org/pdf/2601.17151)
- [CheXPO](https://arxiv.org/pdf/2507.06959)
- [DPO for hallucinated prior exams](https://arxiv.org/html/2406.06496v2)
- [MedGemma Technical Report](https://www.rivista.ai/wp-content/uploads/2025/07/2507.05201v2.pdf)
- [RAD-DINO + BioViL-T latent fusion](https://arxiv.org/pdf/2609.09185)
- [CheXFound](https://arxiv.org/pdf/2502.05142)
- [RadJEPA](https://arxiv.org/pdf/2601.15891)
- [Med-AR](https://arxiv.org/html/2609.29156v1)
- [Reporting Practice Matters](https://arxiv.org/pdf/2609.19093)
- [ECHO](https://arxiv.org/pdf/2604.09450)
