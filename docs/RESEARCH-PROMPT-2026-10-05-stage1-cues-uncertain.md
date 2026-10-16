Bạn là một nhà nghiên cứu về vision-language model cho ảnh X-quang ngực. Hãy dùng chế độ research: đọc kỹ mã nguồn, đọc bài báo gốc và nhiều bài báo liên quan, rồi viết một báo cáo nghiên cứu bằng tiếng Việt. Không đoán mò. Mọi nhận định phải dựa trên số liệu trong repo hoặc trên một bài báo cụ thể có trích dẫn (tên bài, năm, hội nghị hoặc tạp chí, link arXiv/DOI, và số liệu cụ thể trong bài đó). Chỗ nào không có bằng chứng thì ghi rõ "chưa có bằng chứng" hoặc "giả thuyết".

======================================================================
1. NGUỒN CẦN ĐỌC
======================================================================

1.1. Repository hiện tại (public): https://github.com/minhphuong150505/Meta-CXR (nhánh main; nhánh thử nghiệm feat/stage2-finding-tokens)
Đọc theo thứ tự:
- CLAUDE.md: đây là nhật ký đầy đủ nhất của dự án. Đọc kỹ các mục "Three classes, never binary", "Stage 2", "Arm A", "BETTER CUES HAVE NOT ESTABLISHED A GENERATION GAIN", "THE STOP DEFECT IS CONFIRMED", "LEARNABLE FINDING TOKENS", "The zero-shot control", "AND IT BUYS NOTHING IN STAGE 2 EITHER", và D-025 (logit-adjusted loss).
- README.md (tiếng Việt).
- struct/HOME.md và struct/project/_meta/DECISIONS.md (các quyết định D-0xx).
- stage2/prompts/ (builder.py, policies.py, templates.py, records.py): cách Stage-1 được đưa vào prompt.
- training/run_medgemma_qlora.py, training/train_eval_figure9_llm_variants_200.py, training/pipeline_modes.py, training/medgemma/soft_tokens.py: huấn luyện Stage 2 và các pipeline mode.
- mhcac/loss.py, mhcac/mhcac_12.py, model/lavis/models/blip2_models/blip2_qformer.py: loss và đầu phân loại 3 lớp.
- training/evaluation/classification_metrics.py, threshold_calibration.py, paper_protocol.py: cách tính metric theo bài báo.
- docs/handoff/PLAN-2026-09-08-stage2-root-cause.md, PLAN-2026-09-10-mention-finding-tokens.md, PLAN-2026-09-11-mention-conditioned-stage1.md, PLAN-2026-09-30-stage1-three-class-retrain.md, PLAN-2026-10-01-logit-adjusted-loss.md.

1.2. Bài báo gốc và code gốc
- D. Edirisinghe et al., "Chest X-Ray Report Generation Using Abnormality Guided Vision Language Model" (META-CXR), IEEE Access, vol. 13, 2025. Bản PDF có trong repo tại docs/paper/.
- Code gốc của tác giả: https://github.com/DasithEdirisinghe/META-CXR (lưu ý: trong code gốc toàn bộ ITC/ITM/LM của Q-Former bị comment out, prompt trong inference.py khác văn bản bài báo).

======================================================================
2. BỐI CẢNH ĐÃ ĐO (đọc lại trong CLAUDE.md để kiểm chứng, không lấy nguyên xi)
======================================================================

Pipeline hiện tại:
- Stage 1: BioViL-T + PubMedCLIP (+ MedCLIP Swin) -> view fusion -> MHCAC phân loại 14 finding x 3 lớp (Negative/Positive/Uncertain) + Q-Former 32 query. Huấn luyện 3 pha (D-020). Ô trống được tính là Negative (D-018), giống bài báo. Bắt buộc 3 lớp, không được chuyển sang nhị phân (D-023).
- Stage 2: google/medgemma-1.5-4b-it, QLoRA 4-bit, r=16, alpha=32. Các mode: medgemma_direct (chỉ ảnh), meta_cxr_qformer (soft token thay ảnh), meta_cxr_qformer_with_mhcac_prompt, meta_cxr_native_qformer_guided (ảnh gốc + 32 soft token + gợi ý P/N/U dạng văn bản). Cue rule: argmax (mặc định), paper_thresholds (Eq. 22), none.

Kết quả Stage 1 (run_20260930_3class, test n=3,269, protocol của bài báo):
- weighted P/R/F1 = 0.839 / 0.765 / 0.766 (bài báo: 0.87 / 0.78 / 0.73).
- F1 trung bình 5 finding = 0.643 (bài báo: 0.701). Dùng một cutpoint P/N fit trên val thì lên 0.744.
- AUROC one-vs-rest trung bình: Positive 0.776, Negative 0.769, Uncertain chỉ khoảng 0.57-0.59.
- Recall gộp theo lớp N/P/U = 0.802 / 0.747 / 0.046. Lớp Uncertain gần như không bao giờ được dự đoán; ở các finding hiếm thì bằng 0.
- Fit ngưỡng để tăng recall Uncertain: recall U lên 0.25-0.54 nhưng precision U chỉ 0.06-0.16.
- Logit adjustment (Menon et al., ICLR 2021), D-025: tau=1.0 khiến mô hình dự đoán Uncertain cho gần như mọi thứ (24,902 lần dự đoán so với 1,516 ô Uncertain thật), weighted F1 0.77 -> 0.32. Đã không được chọn.
- Mention-conditioned objective chỉ cải thiện calibration giữa các finding (micro AUROC +0.025), không cải thiện AUROC theo từng finding.

Kết quả Stage 2: phải đọc kỹ trong CLAUDE.md, đặc biệt các điểm sau.
- Gợi ý Stage-1 dạng văn bản: 7 lần đo độc lập, mọi khoảng tin cậy bootstrap so với "không gợi ý" đều cắt qua 0. CIDEr cao nhất thường là arm không có gợi ý. Có tính cả các ngưỡng tốt hơn (precision 0.36-0.68) và arm được huấn luyện cùng gợi ý.
- Finding token học được (13 token chứa xác suất liên tục): không giúp; D - A BERTScore -0.0075 [-0.0133, -0.0021]. Ablation cho thấy mô hình đã huấn luyện không đọc kênh này: xoá hay xáo trộn đặc trưng đều không làm thay đổi đầu ra.
- Zero-shot MedGemma so với fine-tune chỉ-ảnh (arm A), cùng 3,102 ca test: zero-shot tốt hơn có ý nghĩa ở ROUGE-1, METEOR, BERTScore và lặp từ ít hơn 3.2 lần; arm A không thắng có ý nghĩa ở metric nào. LƯU Ý: phép so này đo khi còn lỗi stop token (78% token sinh sau <end_of_turn>), và arm A chỉ huấn luyện 0.86 epoch. Sau khi sửa stop token, CIDEr tăng khoảng 22 lần nhưng report quá ngắn (24 từ so với 52 từ của bác sĩ), và phép so zero-shot vs fine-tune CHƯA được đo lại.
- Q-Former chỉ được huấn luyện thật phần ảnh từ run 3 pha (D-020). Các run Stage-2 cũ dùng checkpoint mà soft token là readout chưa học, dù linear probe vẫn giữ được khoảng 90% AUROC của MHCAC.
- Chưa có metric lâm sàng (CheXbert/RadGraph/RadCliQ): chỉ có metric từ vựng. Phải nói rõ hạn chế này.

Câu hỏi của người dùng có hai giả định cần kiểm chứng, không được mặc nhiên chấp nhận:
(a) "MedGemma không fine-tune, không cắm Stage 1 cho kết quả tốt hơn hoàn toàn": kiểm tra xem số liệu có thật sự ủng hộ chữ "hoàn toàn" không, và phân tách ảnh hưởng của lỗi stop token, epoch chưa đủ, độ dài report, và chính gợi ý Stage 1.
(b) "Cắm Stage 1 làm MedGemma lệch hướng và giảm metric": phân biệt "giảm có ý nghĩa thống kê" với "không giúp được". Số liệu repo chủ yếu cho thấy không giúp được; chỉ một vài so sánh giảm có ý nghĩa.

======================================================================
3. NHIỆM VỤ NGHIÊN CỨU
======================================================================

Vấn đề 1: Vì sao đưa kết quả Stage 1 vào MedGemma không giúp, thậm chí làm kém đi?
- Giải thích cơ chế, có bằng chứng từ repo và từ tài liệu. Các giả thuyết cần kiểm tra (không giới hạn ở đây):
  + Gợi ý dư thừa vì MedGemma đã nhìn thấy ảnh qua vision tower riêng.
  + Gợi ý nhiễu (precision Positive chỉ khoảng 0.3-0.5 trên finding hiếm), mô hình học cách bỏ qua hoặc bị dẫn sai (lỗi do anchoring / sycophancy với ngữ cảnh sai).
  + Lệch phân phối giữa train và test: lúc train có thể dùng nhãn thật hay dự đoán? Kiểm tra trong code.
  + Metric từ vựng (BLEU/ROUGE/CIDEr/BERTScore) không đo được độ đúng lâm sàng, nên gợi ý có thể đúng hơn về bệnh mà không làm tăng điểm.
  + Báo cáo tham chiếu thiếu FINDINGS hoặc rất ngắn (khoảng 6% target chỉ là một dòng kỹ thuật).
  + Ngân sách huấn luyện nhỏ (pilot 10,000 ca, ~491 optimizer update).
- Đối chiếu với các bài báo đã gặp hiện tượng tương tự hoặc đã giải quyết nó. Ví dụ hướng cần tìm (tự kiểm chứng, tự bổ sung): PromptMRG (AAAI 2024, diagnosis-driven prompt + logit adjustment), KARGEN, MAIRA-2, R2GenGPT, RaDialog, CheXagent, MedVersa, LLaVA-Rad, các nghiên cứu về LLM bị dẫn sai bởi ngữ cảnh nhiễu (retrieval noise, distractor context), contrastive/context-aware decoding, classifier-free guidance cho LLM, DPO/RL với reward lâm sàng (ví dụ RadGraph reward, CheXbert reward), các bài 2025-2026 mới nhất về report generation có dùng classifier.

Vấn đề 2: Nhãn Uncertain
- Mapping với bài báo gốc: bài báo xử lý Uncertain như thế nào ở từng bước (nhãn: NaN -> Negative, -1 -> Uncertain; chỉ dùng ảnh frontal; loss: class-weighted CE; contrastive loss có thành phần Eq. 18 giữ mẫu Uncertain cách đều tâm Positive và Negative; ngưỡng theo từng (finding, lớp) bằng khoảng cách ROC Eq. 22; prompt PNU; đánh giá: weighted P/R/F1 và AUROC theo lớp, KHÔNG báo F1/recall riêng của lớp Uncertain). Chỉ ra điểm nào repo đã làm giống, điểm nào khác (ví dụ repo dùng cả ảnh không frontal, trọng số lớp tính lại, cap 10).
- Giải thích vì sao lớp Uncertain gần như bằng 0 ở finding hiếm, có số liệu: tỉ lệ ô Uncertain theo finding, chất lượng nhãn Uncertain của CheXpert labeler (độ chính xác của labeler cho lớp uncertain so với bác sĩ, ví dụ trong bài MIMIC-CXR-JPG và CheXpert/CheXbert), và liệu Uncertain có tồn tại trong ảnh hay chỉ là cách dùng từ trong report.
- Đánh giá xem các số AUROC Uncertain trong bài báo (Fig. 5) có thật sự cao hơn repo không, và liệu bài báo có che đi vấn đề này nhờ dùng weighted metric hay không.
- Các hướng cần tìm và đánh giá (tự kiểm chứng, tự bổ sung): U-Ignore/U-Zeros/U-Ones/U-SelfTrained/U-MultiClass trong bài CheXpert (Irvin et al., AAAI 2019); label smoothing regularization cho uncertain (Pham et al., 2021); ordinal regression (Uncertain nằm giữa Negative và Positive, repo đã đo điều này trên điểm p_pos/(p_pos+p_neg)); soft label / label distribution learning; noisy-label learning; evidential deep learning / Dirichlet; conformal prediction và selective prediction (cho phép mô hình nói "không chắc" có kiểm soát sai số); nhãn tốt hơn từ CheXbert / LLM labeler; hierarchical loss.

======================================================================
4. YÊU CẦU VỀ ĐỀ XUẤT
======================================================================

Với mỗi đề xuất cho Vấn đề 1 và Vấn đề 2, trình bày ngắn gọn theo cùng một khuôn:
- Ý tưởng (2-3 câu, người không chuyên vẫn hiểu).
- Bằng chứng: bài báo nào đã làm, trên dữ liệu nào, cải thiện bao nhiêu (số cụ thể), và vì sao kết quả đó có thể áp dụng sang đây.
- Mức độ phù hợp với pipeline hiện tại: cần sửa file nào, có phá vỡ ràng buộc nào không (3 lớp bắt buộc, Stage 1 và Stage 2 tách rời, chỉ có 1 GPU RTX 5060 Ti 16 GB, một epoch Stage 2 mất khoảng 80 giờ), có dùng được checkpoint hiện có không.
- Chi phí: số giờ GPU ước lượng, dựa trên tốc độ đã đo trong CLAUDE.md.
- Đánh đổi: được gì, mất gì (ví dụ tăng recall Uncertain nhưng giảm weighted F1; tăng độ đúng lâm sàng nhưng giảm BLEU).
- Có báo cáo được trong luận văn và trong bài báo khoa học không, và cần đối chứng gì để kết luận hợp lệ (cùng cohort, paired bootstrap, nhiều seed).
- Mức độ tin cậy: cao / trung bình / thấp, kèm lý do.

Xếp hạng các đề xuất theo tỉ lệ lợi ích trên chi phí.

======================================================================
5. TÍNH MỚI
======================================================================

Đề xuất MỘT hướng có tính mới rõ rệt, đủ để làm đóng góp chính cho một bài báo, xây dựng trên chính phát hiện tiêu cực của repo (gợi ý từ classifier không giúp được LLM có mắt). Yêu cầu:
- Nêu rõ câu hỏi nghiên cứu và giả thuyết có thể bác bỏ được.
- Chứng minh tính mới: tìm kỹ xem đã có bài nào làm chưa (liệt kê các bài gần nhất và chỉ ra khác biệt). Nếu đã có người làm, nói thẳng và chọn hướng khác.
- Nối được cả hai vấn đề nếu có thể: ví dụ dùng độ không chắc chắn của Stage 1 (lớp Uncertain, calibration, conformal set) để quyết định khi nào và bằng cách nào đưa gợi ý vào LLM. Đây chỉ là ví dụ, hãy tự đánh giá và có thể đề xuất hướng khác tốt hơn.
- Thiết kế thí nghiệm tối thiểu để chứng minh: các arm, metric (bao gồm metric lâm sàng, nói rõ cái nào chưa có trong repo và cần thêm), cỡ mẫu, cách so sánh thống kê, tiêu chí thành công và tiêu chí dừng.
- Rủi ro chính và phương án dự phòng nếu kết quả âm tính (một kết quả âm tính được phân tích kỹ vẫn có giá trị báo cáo không, và báo cáo như thế nào).

======================================================================
6. ĐỊNH DẠNG ĐẦU RA
======================================================================

- Một file .docx (KHÔNG phải .md), tiếng Việt, trình bày kiểu luận văn cơ bản: font Times New Roman 13, giãn dòng 1.3-1.5, đề mục đánh số (1, 1.1, 1.1.1), bảng đơn giản có tiêu đề, hình nếu cần. Không icon, không emoji, không hộp màu, không định dạng cầu kỳ.
- Ngắn gọn nhưng đúng trọng tâm, khoảng 12-20 trang. Viết cho người đọc là sinh viên làm luận văn và giáo viên hướng dẫn: giải thích thuật ngữ khi dùng lần đầu, mỗi kết luận đi kèm số liệu.
- Cấu trúc bắt buộc:
  1. Tóm tắt (nửa trang): vấn đề, nguyên nhân chính, đề xuất chính, tính mới.
  2. Hiện trạng pipeline và số liệu đã đo (bảng ngắn, có ghi run và n).
  3. Vấn đề 1: vì sao MedGemma không fine-tune / không có gợi ý Stage 1 lại cho kết quả tốt hơn hoặc ngang bằng. Kiểm chứng giả định (a) và (b) ở mục 2. Phân tích nguyên nhân, xếp theo mức độ bằng chứng.
  4. Vấn đề 2: nhãn Uncertain. Bảng mapping bài báo gốc với repo hiện tại. Phân tích vì sao lớp Uncertain gần như bằng 0 ở finding hiếm.
  5. Các phương án cải thiện cho Vấn đề 1 (theo khuôn ở mục 4).
  6. Các phương án cải thiện cho Vấn đề 2 (theo khuôn ở mục 4).
  7. Hướng tính mới đề xuất và thiết kế thí nghiệm.
  8. Đánh đổi tổng hợp và khuyến nghị thứ tự thực hiện (bảng: phương án, chi phí GPU, lợi ích kỳ vọng, rủi ro, có báo cáo được không).
  9. Hạn chế của phân tích này.
  10. Tài liệu tham khảo (định dạng IEEE, mỗi tài liệu có link).

======================================================================
7. RÀNG BUỘC
======================================================================

- Không đưa dữ liệu bệnh nhân vào báo cáo: không subject_id, study_id, dicom_id, đường dẫn ảnh thật, hay nội dung report thật. MIMIC-CXR chịu DUA của PhysioNet.
- Không đề xuất chuyển bài toán về nhị phân (Positive vs không Positive). Mọi phân loại phải giữ 3 lớp như bài báo; nếu một phương pháp chỉ áp dụng được cho nhị phân thì nói rõ và giải thích cách mở rộng sang 3 lớp, hoặc loại bỏ.
- Không đề xuất lại những gì repo đã thử và đã cho kết quả âm tính (ví dụ dò thêm ngưỡng cue, finding token học được ở ngân sách 10,000 ca, mention gate, logit adjustment tau=1.0), trừ khi có lý do mới rõ ràng và nói rõ lý do đó.
- Phân biệt rõ ba loại khẳng định: đã đo trong repo, đã được chứng minh trong bài báo khác, và giả thuyết của bạn.
- Không coi metric từ vựng là độ chính xác lâm sàng.
