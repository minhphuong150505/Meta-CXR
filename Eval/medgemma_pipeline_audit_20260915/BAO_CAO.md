# Vì sao MedGemma nhận ảnh trực tiếp có thể hơn pipeline đã fine-tune?

Ngày điều tra: **15/09/2026**. Mã nguồn được đọc: `d90cba945b29159102df0c4fa41f9b3eed8fc71b`.

**Kết luận chính:** mô hình đã fine-tune có học, nhưng hành vi sinh của nó đang nghiêng mạnh về câu trả lời ngắn và ít đa dạng. Nhánh kết hợp MHCAC + Q-Former gặp hiện tượng này nặng hơn nhánh ảnh trực tiếp. Zero-shot hơn ở nhiều metric đo độ phủ, nhưng **không hơn mọi metric**: trên bộ đánh giá mới, CIDEr của hai nhánh fine-tune cao hơn. Vì vậy, ưu tiên hiện tại là kiểm soát đầu vào/đối chứng và khắc phục hiện tượng sinh quá ngắn; chưa có căn cứ để coi việc thêm loss Stage-1 là đòn bẩy đầu tiên.

Đọc mục 1–3 để hiểu kết quả, mục 4–7 để truy nguyên, rồi dùng [kế hoạch thử nghiệm](THU_NGHIEM_TIEP_THEO.md) để quyết định sửa gì và đo gì.

## 1. Phạm vi và độ chắc chắn của điều tra

Điều tra này đối chiếu **code đang có**, cấu hình, handoff cũ, hai bộ artifact tổng hợp trong `Eval/`, và nguồn chính thức của MedGemma/các metric. Không đọc hay đưa văn bản báo cáo bệnh nhân vào tài liệu.

| Mức bằng chứng | Ý nghĩa trong báo cáo |
|---|---|
| Xác minh từ code | Đã đọc đường gọi và cấu hình liên quan; chưa đồng nghĩa đã đo tác động trên mô hình |
| Xác minh từ artifact | Đã đọc trực tiếp JSON/CSV tổng hợp có sẵn tại máy dev; không chỉ chép diễn giải trong README |
| Thực nghiệm đã ghi nhận | Có báo cáo thực thi trước đây trong handoff; phiên này chưa chạy lại |
| Giả thuyết | Cơ chế phù hợp với quan sát, nhưng cần một can thiệp có đối chứng để kết luận |

**Giới hạn:** SSH tới máy train timeout ở cả lần thường và lần thử lại ngoài sandbox, exit 255. Vì thế chưa kiểm tra lại timestamp/checksum của checkpoint, generation summary và raw output trên host; chưa thực hiện thêm training, inference, scoring hoặc test. Không có phép đo mới trên GPU trong báo cáo này. Bộ `standard/test_main.json` không tự chứa toàn bộ provenance generation; mô tả “đã sinh lại với stop đúng” được đối chiếu với chương trình dựng báo cáo đi cùng bundle, và vẫn cần xác nhận lại trên host.

Hai nguồn kết quả cần phân biệt:

- [Bundle cũ gồm nhiều đợt thí nghiệm](../eval_bundle_20260913/README.md): có cả đầu ra **trước** và **sau** sửa stop; không thể lấy mọi hàng làm một bảng xếp hạng.
- [Bộ đánh giá luận văn — test_main](../thesis_eval_20260913/data/standard/test_main.json), [chẩn đoán độ dài](../thesis_eval_20260913/data/length_diagnostics.json), [cấu hình adapter](../thesis_eval_20260913/data/configs.json): có kết quả test mới hơn, thay đổi đáng kể cách đọc tình hình. [Chương trình dựng tài liệu](../thesis_eval_20260913/generator/build.py) ghi rằng ba nhánh test đã được sinh lại sau sửa stop.

## 2. “Không train MedGemma” có nghĩa chính xác là gì?

Ở đây zero-shot là **không fine-tune thêm trong dự án**, không phải mô hình chưa từng học ảnh X-quang. Google công bố MedGemma đã được huấn luyện trên dữ liệu y khoa, gồm ảnh và báo cáo MIMIC-CXR. Nó có đường ảnh → biểu diễn đa phương thức → ngôn ngữ đã được huấn luyện trước. Vì vậy đầu vào “chỉ có ảnh” vẫn đi qua một hệ thống có kiến thức rất mạnh. [Model card chính thức](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card).

Không suy từ việc Google dùng MIMIC-CXR thành “Google đã thấy đúng các ca test này”: artifact hiện có không cho phép kiểm tra điều đó. Đây là giới hạn provenance của base model. Hơn nữa cả ba nhánh dùng cùng base, nên riêng điều này không giải thích được chênh lệch sau fine-tune.

| Tên dùng trong báo cáo | Đầu vào thực tế | Thành phần học thêm trong dự án |
|---|---|---|
| **Z: zero-shot** | Ảnh qua vision tower/projector gốc của MedGemma + prompt | Không có adapter Stage-2 |
| **A: direct fine-tuned** | Cùng loại đường ảnh gốc + prompt | Khoảng **29,80 triệu** tham số LoRA ở tháp ngôn ngữ |
| **C: combined fine-tuned** | Ảnh gốc **cộng** 32 soft token Q-Former **cộng** cue MHCAC | LoRA như A + khoảng **1,97 triệu** tham số `img_proj` |

C ở bảng này là `meta_cxr_native_qformer_guided`. Nó **giữ ảnh gốc MedGemma**; không thể giải thích việc C kém A bằng việc “C đã bỏ vision tower gốc”. Các mode `meta_cxr_qformer*` khác mới thay ảnh bằng soft token. Xem [pipeline_modes.py](../../training/pipeline_modes.py).

**Bằng chứng chống lại nhận định “train nhưng không học”:** A có 9.500 optimizer updates đã lưu; C có 11.020 updates theo handoff. Code thật sự gọi backward/optimizer step. Phân bố đầu ra của adapter thay đổi rõ, CIDEr tăng, và pilot finding-token đã ghi nhận các tham số khởi tạo cố định thay đổi. Tuy nhiên, số tham số `requires_grad=True` tự nó chưa chứng minh mọi module nhận gradient hữu ích; cần đo gradient riêng nếu nghi một nhánh bị ngắt.

Ở Stage-2, Stage-1 được đặt eval, đóng băng, xuất feature ra cache rồi giải phóng model. **Loss sinh báo cáo không cập nhật ngược MHCAC/Q-Former** trong kiến trúc hiện tại. Nguồn: [lavis_loader.py](../../training/stage1/lavis_loader.py), [build_stage1_records và train_fine](../../training/train_eval_figure9_llm_variants_200.py).

Lưu ý tên arm bị tái sử dụng: **A/B/C/D của pilot finding-token đều có ảnh native + 32 soft token**; “pilot A, cue none” không phải A direct ở bảng trên. Gộp chúng sẽ tạo một đối chứng sai.

## 3. Kết quả mới thực sự cho thấy gì?

### 3.1 Cùng 2.772 ca test, giao thức COCO/PTB + RoBERTa-large

Nguồn trực tiếp: [standard/test_main.json](../thesis_eval_20260913/data/standard/test_main.json). Số dưới đây là giá trị corpus trong artifact, làm tròn 4 chữ số.

| Metric | Z: zero-shot | A: direct FT | C: combined FT |
|---|---:|---:|---:|
| BLEU-1 | **0,2770** | 0,1638 | 0,0920 |
| BLEU-4 | **0,0556** | 0,0452 | 0,0277 |
| METEOR 1.5, corpus | **0,1356** | 0,0964 | 0,0796 |
| ROUGE-L | **0,1995** | 0,1912 | 0,1728 |
| CIDEr | 0,0791 | 0,1439 | **0,1950** |
| BERTScore F1, RoBERTa-large, raw | 0,8525 | **0,8543** | 0,8422 |
| BERTScore F1, RoBERTa-large, rescaled | 0,1260 | **0,1370** | 0,0652 |

Do đó câu đúng là: **Z giữ độ phủ văn bản tốt hơn; fine-tune đổi cách sinh theo hướng có lợi cho CIDEr nhưng bất lợi cho nhiều metric khác.** Chưa có metric lâm sàng để xác định bên nào mô tả bệnh chính xác hơn.

Các chênh lệch ghép cặp đáng chú ý trong artifact:

- A − Z: CIDEr **+0,0647**, CI95 **[+0,0376; +0,0957]**; BLEU-4 **−0,0103**, CI95 **[−0,0138; −0,0068]**.
- C − A: CIDEr **+0,0512**, CI95 **[+0,0260; +0,0795]**; ROUGE-L **−0,0184**, CI95 **[−0,0243; −0,0122]**; BERTScore raw **−0,0121**, CI95 **[−0,0139; −0,0103]**.
- A − Z: BERTScore raw chỉ **+0,00186**, cận dưới gần 0; đây không phải lợi ích vững chắc để chọn mô hình.

CI của BLEU dùng 500 lần resample, các metric còn lại 2.000 theo artifact. Các CI này thể hiện bất định trên tập ca của **một lần train**, không bao gồm biến thiên giữa các seed.

### 3.2 Vấn đề nổi bật là đầu ra ngắn và tập trung vào ít câu mẫu

Nguồn: [length_diagnostics.json](../thesis_eval_20260913/data/length_diagnostics.json). Độ dài ở bảng này dùng bộ đếm từ `\w+` trong script đi cùng bundle; không phải token subword MedGemma.

| Chỉ báo trên cùng 2.772 ca | Z | A | C |
|---|---:|---:|---:|
| Trung vị từ sinh ra | 63 | 31 | **6** |
| Trung vị từ tham chiếu | 56 | 56 | 56 |
| Đầu ra ≤10 từ | 0% | **26,95%** | **50,90%** |
| Đầu ra ≤5 từ | 0% | 5,56% | 11,15% |
| Số chuỗi đầu ra khác nhau | 2.684 | 1.783 | 1.220 |
| Tỉ lệ của 5 chuỗi phổ biến nhất | 1,37% | 17,57% | **32,50%** |

Hai kiểm tra giúp hiểu con số:

1. Trong C, tham chiếu của nhóm sinh ≤10 từ có trung vị **56 từ**, bằng nhóm còn lại. Chỉ **6,75%** tham chiếu của cohort dài ≤15 từ. Sự ngắn đi không đơn thuần là phản ánh một tập test toàn báo cáo ngắn.
2. Có **679 ca cùng sinh ngắn ở A và C**, trong tổng 747 ca sinh ngắn ở A. Tức khoảng **90,90%** nhóm ngắn của A cũng ngắn ở C. Thành phần chung như mục tiêu SFT, prompt, dữ liệu và decoder cần được điều tra trước khi đổ toàn bộ lỗi cho MHCAC.

Đây là bằng chứng về **sụt độ phủ và đa dạng bề mặt**. Chưa được gọi tất cả báo cáo ngắn là sai bệnh: báo cáo bình thường có thể ngắn hợp lý. Phải đo thiếu sót bệnh, vị trí, mức độ bằng bộ chấm lâm sàng và kiểm tra mẫu mới biết thiệt hại thực tế.

### 3.3 Đổi bộ chấm có thể đổi cả thứ hạng BERTScore

Các artifact `repo_protocol` cùng bộ luận văn sử dụng DistilBERT raw. Chúng cho BERTScore Z/A/C lần lượt **0,8129 / 0,7955 / 0,7650**. Bộ `standard` dùng RoBERTa-large raw cho **0,8525 / 0,8543 / 0,8422**. Riêng thứ tự Z–A đã đảo ở điểm ước lượng.

Không được xem 0,85 là “85% đúng bệnh”. BERTScore là độ tương đồng embedding; lựa chọn encoder, layer và baseline ảnh hưởng giá trị. Rescale là thay đổi thang đo, không phải thêm bằng chứng lâm sàng. [Tài liệu chính thức BERTScore](https://github.com/Tiiiger/bert_score).

Nguồn đối chiếu: [repo Z](../thesis_eval_20260913/data/repo_protocol/eval_zeroshot/metrics.json), [repo A](../thesis_eval_20260913/data/repo_protocol/eval_armA/metrics.json), [repo C](../thesis_eval_20260913/data/repo_protocol/eval_armC/metrics.json). Bộ `standard` và bộ repo cũng dùng tokenizer khác, nên độ dài đếm kèm metric không được trộn trực tiếp với bảng `\w+` ở trên.

## 4. Các nguyên nhân, xếp theo bằng chứng

### 4.1 Lỗi stop token: nguyên nhân đã xác lập cho sự lặp của kết quả cũ

MedGemma dùng cả token 1 và 106 để dừng, nhưng code cũ ghi đè thành chỉ token 1. Training target kết thúc bằng token 106; model có thể kết thúc đúng rồi bị buộc tiếp tục sinh.

Probe cũ trên 25 ca: điều kiện không cue sinh 4.000 token với stop cũ, trong đó **3.112 token nằm sau end-of-turn**. Sửa stop còn 888 token, không token nào sau dấu kết thúc, không ca nào chạm cap. CIDEr theo scorer nội bộ tăng **0,0065 → 0,1683**. Đây là can thiệp cơ chế rõ ràng, mạnh hơn suy luận từ loss curve. Nguồn: [handoff root-cause](../../docs/handoff/PLAN-2026-09-08-stage2-root-cause.md), [CLAUDE.md](../../CLAUDE.md), [probe theo bộ chấm chuẩn](../thesis_eval_20260913/data/standard/stop_probe_25.json).

**Hệ quả:** không dùng các bảng trước 09/09 để kết luận hiện trạng. Không suy rằng lỗi giống nhau ở hai model thì tác động cũng giống nhau: xác suất phát token 106 của chúng khác nhau. Cũng không sửa sự sinh ngắn bằng cách bỏ token 106 lần nữa.

Bộ test mới vẫn cho C trung vị 6 từ sau sửa stop. **Stop fix sửa phần sinh thừa; không giải quyết quyết định kết thúc quá sớm của model.**

### 4.2 Mục tiêu train và cách chọn checkpoint chưa đo hành vi cần tối ưu

Code `train_fine` tối ưu cross-entropy với prefix chuẩn. Khi tính val loss, model cũng nhận prefix chuẩn. Khi sinh, nó nhận chính token mình vừa chọn. Loss thấp không đảm bảo tránh câu mẫu hoặc kết thúc sớm khi chạy tự do.

Ngoài ra:

- A chỉ giữ checkpoint update 9.500, khoảng **0,8621 epoch**; không có end-of-epoch validation, `val_loss=null` trong artifact.
- C hoàn thành một epoch nhưng với `epochs=1`, việc “chọn tốt nhất” chỉ có **một checkpoint cuối epoch được đánh giá**, không cho biết checkpoint giữa epoch có tốt hơn hay không.
- C có loss sinh báo cáo, nhưng Stage-1 đã đóng băng; loss này không trực tiếp sửa chất lượng cue hoặc khả năng alignment Q-Former.
- Cả hai dùng LoRA ở nhiều projection attention/MLP, learning rate **1e-4**; `img_proj` của C dùng **1e-3**. Đây là cấu hình thực, **chưa đủ để kết luận LR quá cao**.

Giả thuyết ưu tiên: SFT làm thay đổi mạnh phân bố ngôn ngữ và quyết định dừng, trong khi tiêu chí lựa chọn không phát hiện thiếu độ phủ. “Catastrophic forgetting”, LR lớn hoặc thiếu regularization là các cơ chế cần so sánh checkpoint/LR, chưa phải kết luận đã đo.

### 4.3 Prompt khuyến khích ngắn, trong khi target giữ nguyên báo cáo đầy đủ

Hai config thực nghiệm đặt `min_sentences=1`, `max_sentences=4`; template yêu cầu đoạn văn ngắn gọn. Đồng thời prompt cấm phát biểu so sánh khi không có prior, nhưng `temporal.target_policy=keep` giữ nguyên các target có thể chứa so sánh.

Nguồn: [config A](../../configs/experiment_native_anchor_only.yaml), [config C](../../configs/experiment_native_qformer_guided.yaml), [templates.py](../../stage2/prompts/templates.py), [builder.py](../../stage2/prompts/builder.py).

Đây là **xung đột có thể có giữa chỉ dẫn và supervision**. Chưa đo tỉ lệ target vi phạm giới hạn câu hoặc chứa temporal clause trên toàn bộ tập train. Không thể quy trung vị 6 từ chỉ cho câu “concise”, vì Z và các checkpoint khác có thể phản ứng khác. Phép đo cần làm là giữ cùng model, thay đúng chỉ dẫn về độ phủ/độ ngắn trên val, rồi kiểm tra clinical recall và xác suất dừng sớm.

### 4.4 Dữ liệu có target ngắn và lệch phân phối, nhưng không đủ giải thích toàn bộ

[data_stats.json](../thesis_eval_20260913/data/data_stats.json) ghi:

- Train: **176.312** study có FINDINGS hợp lệ; **8.298** target ≤15 token từ vựng, khoảng **4,71%**; trung vị 51.
- Val: **1.415** target hợp lệ, 83 target ≤15 token, khoảng 5,87%; trung vị 52.
- Test toàn bộ study hợp lệ: **2.984**, 200 target ≤15 token, khoảng 6,70%; trung vị 63. Đây là tập khác cohort 2.772 và khác bộ đếm ở mục 3.
- Tỉ lệ `NARRATIVE_BODY` trên **toàn bộ study**, chưa lọc valid, tăng từ 31.748/222.758 (**14,25%**) ở train lên 898/3.269 (**27,47%**) ở test. Phải cross-tab theo valid trước khi quy chênh lệch này cho tập sinh báo cáo.

Như vậy có target rất ngắn, chất lượng trích xuất cần audit và khác biệt phân phối train–test. Nhưng khoảng 4,71% target train ngắn không tự giải thích vì sao 50,90% đầu ra C ngắn. Cần xem nhóm bệnh, mức độ bất thường, extraction method và câu mẫu trên **những ca sinh ngắn**, không chỉ xóa target dưới một ngưỡng.

Ảnh native chỉ là một anchor, trong khi report có thể tổng hợp nhiều view và prior. Thiếu bằng chứng đầu vào làm một phần target không thể suy ra chắc chắn. Cho model học bắt chước cả phần đó có thể làm tăng ngôn ngữ chung hoặc phát biểu không có căn cứ; tác động cụ thể chưa được đo.

### 4.5 MHCAC thêm thông tin có nhiễu, và thông tin đúng chưa chắc bổ sung cho MedGemma

Cue lịch sử dùng xác suất dương **có điều kiện đã được nhắc tới**, dễ phát ra nhiều finding khi bỏ qua mention gate. Thực nghiệm selective cue đã ghi nhận micro precision trên test tăng từ khoảng **0,180** lên **0,680**, nhưng recall giảm, và precision được đo đối với nhãn report, không phải truth ảnh đầy đủ.

Sau sửa stop, trên cùng 100 ca val, các cue conditional/marginal/selective đều chưa chứng minh gain so với none: CI của các metric chính đều chứa 0. Pilot **được train** với cue marginal cũng không có lợi ích chắc chắn ở ngân sách ~7.854 target hợp lệ / 491 updates. Thay Stage-1 bằng checkpoint mention-conditioned và train lại hai nhánh Stage-2 cũng chưa cho gain nhất quán.

Nguồn: [bảng kết quả](../eval_bundle_20260913/stage2_generation.csv), [bootstrap](../eval_bundle_20260913/stage2_paired_bootstrap.csv), [handoff finding-token](../../docs/handoff/PLAN-2026-09-10-mention-finding-tokens.md).

Diễn giải được hỗ trợ: **chưa chứng minh cue đem thông tin hữu ích bổ sung trong các chế độ đã thử**. Không được nâng thành “MedGemma hoàn toàn không đọc cue”: mean metric gần bằng nhau không chứng minh từng đầu ra bằng nhau. Cần thêm tỉ lệ đầu ra thay đổi, thay đổi nội dung finding và can thiệp tráo cue theo ca.

Một giả thuyết nữa: Stage-1 học trên cùng train split dùng để tạo cue cho Stage-2, nên chất lượng/confidence cue train có thể lạc quan hơn cue val/test. Chưa có audit in-sample so với out-of-fold; đây là việc đo phân phối cue, không phải bằng chứng về leakage sang test.

### 4.6 Q-Former có tín hiệu nhưng chưa được alignment cho nhiệm vụ hiện tại

Stage-1 đặt `lambda_itc/itm/lm=0`. Nhánh query/cross-attention không được tối ưu bằng ba objective đó trên feature của repo. Một số tham số Q-Former dùng chung vẫn có thể thay đổi qua text teacher; vì vậy “toàn bộ Q-Former hoàn toàn bất động” cũng không chính xác.

Trong C, Stage-2 chỉ học một Linear 768→2560 trên 32 token đã cache. Không có loss Stage-2 chạy ngược vào Q-Former. Những token đó không tương đương với một visual bridge đã được huấn luyện đồng bộ cho MedGemma.

Tuy nhiên probe lịch sử có AUROC **0,6847** so với shuffled-label **0,4838**; không thể gọi feature hoàn toàn vô nghĩa. Cosine giữa các token trong cùng study khoảng **0,8022** gợi ý trùng lặp, nhưng tự nó không đo rank hiệu dụng. Can thiệp zero soft-token từng làm ROUGE-L giảm; không có bằng chứng rằng xóa nhánh sẽ tự động chữa lỗi.

**Kết luận cho đề xuất tách loss:** tách và train Q-Former riêng là thí nghiệm có thể làm sau, nhưng không sửa được việc **A direct cũng sinh ngắn**, không sửa prompt, bộ chấm hoặc stop contract. Lịch sử ITC ở mức chance không xác lập riêng batch size là nguyên nhân duy nhất; cần probe alignment có đối chứng thay vì chỉ bật lại các lambda.

## 5. Các vấn đề code cần xử lý trước thí nghiệm lớn

Những điểm dưới đây được xác minh bằng đọc code. Tác động lên từng checkpoint cũ chưa được lượng hóa.

| Mã | Phát hiện | Vì sao ảnh hưởng điều tra | Bước xử lý |
|---|---|---|---|
| C01 | Train native gọi `select_anchor_rows`; `generate_stage2_reports.native_records` lọc PA/AP rồi duyệt **từng dòng ảnh**, không group study | Cùng một report có thể được chấm nhiều lần; ảnh đưa vào eval có thể khác anchor train; cùng limit chưa phải cùng cohort | Một selector anchor dùng chung; đếm unique study và unique subject trên host |
| C02 | Train native ghi `anchor_view`; generation native ghi `view_position`; `context_from_record` chỉ đọc `anchor_view` | Prompt train có thể có dòng tư thế ảnh nhưng prompt inference mất dòng đó | Chuẩn hóa record schema, test prompt parity qua hai producer thật |
| C03 | Adapter được train với prompt v2, nhưng generation CLI nhận `--prompt-config` tùy chọn; không tự đối chiếu config/hash trong adapter | Bỏ flag vẫn sinh được bằng prompt legacy. Script pipeline lịch sử ở STEP 3 thực sự bỏ flag | Kiểm tra manifest prompt; thiếu/mismatch phải báo rõ hoặc dùng config đã lưu; giữ override có provenance cho probe |
| C04 | `build_stage1_records(split=train)` dùng dataset train; YAML bật augmentation; `model.eval()` không tắt transform dataset | Cache train có thể chứa một lần affine/jitter ngẫu nhiên; soft token/cue khác ảnh raw MedGemma thấy và khác transform val | Kiểm tra config/cache lịch sử; dùng transform export xác định khi dựng cache mới; lưu transform/seed trong fingerprint |
| C05 | `--train-limit` nhánh Stage-1 cắt **N study đầu trước lọc target**; native lấy mẫu ngẫu nhiên sau lọc target | Pilot 10k thực tế chỉ ~7.854 target; so với native 10k hợp lệ sẽ khác ngân sách và cách lấy mẫu | Chọn danh sách study hợp lệ chung trước, rồi mọi nhánh nhận cùng danh sách/thứ tự |
| C06 | Resume đọc `start_epoch=epoch+1`; checkpoint giữa epoch không chứa con trỏ đủ để khôi phục mẫu đã thấy | Một run có thể nhảy qua phần còn lại hoặc chạy vòng lặp rỗng; `status=complete` không đủ | Kiểm tra updates/samples/epoch_completed; fail khi resume không hỗ trợ, hoặc triển khai resume có sampler state đúng |
| C07 | `compute_nlg` cũ bắt lỗi METEOR/ROUGE/CIDEr và có đường trả 0; evaluator riêng có cơ chế báo unavailable | Một số metric trong summary của train có thể mang nghĩa khác metric chấm chính thức | Dùng một evaluator canonical, ghi implementation/version; không dùng 0 thay lỗi |
| C08 | `img_proj` soft-token là Linear không có bước hiệu chỉnh norm như finding-token encoder | Khả năng lệch scale với embedding đầu vào là giả thuyết kỹ thuật đáng đo | Đo RMS/norm projected token so với native/text token; chỉ đổi kiến trúc nếu đo cho thấy lệch và ablation xác nhận |

Điểm gọi để sửa:

- C01/C02/C03: [generate_stage2_reports.py](../../scripts/generate_stage2_reports.py), khoảng dòng 240–271, 363–414; [manifest.py](../../training/dataio/manifest.py), `select_anchor_rows`/`build_records`; [records.py](../../stage2/prompts/records.py).
- C04/C05: [lavis_loader.py](../../training/stage1/lavis_loader.py), `make_stage1_loader`; [ReportDataset.py](../../model/lavis/data/ReportDataset.py), khoảng dòng 440–492, 628–668; [build_stage1_records](../../training/train_eval_figure9_llm_variants_200.py), dòng 680 trở đi.
- C06/C07/C08: [động cơ Stage-2](../../training/train_eval_figure9_llm_variants_200.py), `train_fine`, `compute_nlg`, khởi tạo `img_proj`; [soft_tokens.py](../../training/medgemma/soft_tokens.py).

**Phạm vi:** C04 chưa chứng minh cache cũ thực sự được dựng với augmentation, vì người chạy có thể dùng YAML khác. C03 chứng minh đường sai có tồn tại và script lịch sử dùng nó; chưa xác minh prompt của cả ba run test mới vì thiếu generation summary tại máy dev. C01 không phủ định cohort 2.772 đã ghép với Stage-1; nó đòi hỏi chứng minh uniqueness theo study thay vì chỉ theo ảnh.

**Cue rule cũng là một phần hợp đồng checkpoint:** C full lịch sử được train với `conditional_positive`, trong khi CLI hiện mặc định `marginal_positive` cho guided mode. Sinh lại checkpoint cũ để tái hiện phải chỉ định đúng rule đã train; đổi rule là một can thiệp cần ghi tên riêng. Việc đổi default không tự làm adapter cũ được train lại với cue mới.

## 6. Các điểm dễ làm đọc sai metric và bằng chứng

### 6.1 CIDEr tăng không đồng nghĩa độ đầy đủ tăng

CIDEr dùng so khớp n-gram có trọng số và implementation có các chi tiết về độ dài. Nó không trực tiếp hỏi “có bỏ sót tổn thương không”. Ở cohort này **C đạt CIDEr cao nhất cùng lúc có đầu ra ngắn nhất**. Điều đó đủ để bác bỏ cách chọn model chỉ theo CIDEr, nhưng không đủ để nói mọi mức tăng CIDEr đều do câu mẫu: cần chấm theo nhóm output ngắn/dài và theo finding.

Bộ chấm lâm sàng hiện chưa được nối vào repo: [clinical.py](../../training/evaluation/clinical.py) chủ động raise. Nghiên cứu đối chiếu với bác sĩ cho thấy các metric có những kiểu sai khác nhau và RadGraph/BERTScore có tương quan tốt hơn BLEU trong thiết lập được khảo sát; đây không phải chứng nhận BERTScore đơn lẻ là độ đúng lâm sàng. [Yu và cộng sự](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499844/).

### 6.2 Bootstrap METEOR đang trả lời khác đại lượng ở bảng corpus

Trong `standard/test_main.json`, METEOR corpus A−Z là **0,0963666−0,1356238 = −0,0392572**, nhưng trường paired delta ghi **−0,0273786**. [build.py](../thesis_eval_20260913/generator/build.py) giải thích CI dùng **trung bình điểm từng study**. Hai đại lượng không giống nhau; đây không phải sai số làm tròn.

API METEOR COCO trả riêng score toàn corpus và danh sách score từng mẫu. [Mã nguồn gốc](https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/meteor/meteor.py).

Cách báo cáo đúng: đặt tên `meteor_corpus` và `meteor_sample_mean` riêng; CI phải gắn đúng point estimate. Nếu muốn CI của corpus METEOR, resample rồi tính lại corpus score. Không lấy CI của mean-of-study để mô tả trực tiếp hiệu corpus.

### 6.3 Cùng seed, cùng n chưa đủ thành đối chứng kiến trúc

- A và C full khác số updates đã hoàn thành; chưa đối chiếu đầy đủ train cohort/anchor ở mức study.
- Bật soft/finding token thêm tham số và có thể làm khác chuỗi RNG trước khi khởi tạo LoRA. Một seed không loại được biến thiên training.
- `cue none` trong guided mode vẫn giữ instruction của guided mode; không làm prompt tự biến thành native-only. “Bỏ block cue” là can thiệp cụ thể, không phải toàn bộ kiến trúc direct.
- CI chứa 0 không chứng minh hai nhánh tương đương hoặc model không sử dụng feature. Muốn kết luận tương đương phải đặt biên hiệu ứng nhỏ nhất đáng quan tâm trước và kiểm tra độ chính xác thống kê đủ.
- Bootstrap theo study bỏ qua tương quan giữa nhiều lần chụp của cùng một người. Bổ sung bootstrap theo subject trên host để kiểm tra độ bền; chỉ xuất số tổng hợp.

### 6.4 Những kết luận cũ cần dùng thận trọng

| Cách diễn giải cũ | Cách diễn giải có căn cứ hơn |
|---|---|
| “Fine-tune không hơn zero-shot” | Đúng với một số metric/thiết lập; bộ test mới có CIDEr tăng rõ và BERTScore phụ thuộc encoder chấm |
| “Lỗi stop đối xứng nên thứ hạng chắc giữ nguyên” | Cùng lỗi không bảo đảm cùng tác động; phải sinh lại từng nhánh |
| “Model không đọc finding-token vì các CI chứa 0” | Chưa đo được gain hoặc độ nhạy qua metric; chưa chứng minh output không đổi |
| “A dừng ở 0,86 epoch không thể giải thích kém hơn base” | Chưa có trajectory checkpoint; SFT có thể kém hơn base ở một đoạn training |
| “Q-Former thất bại chắc chắn do batch nhỏ” | Batch là giả thuyết hợp lý trong nhiều khác biệt đồng thời; chưa có ablation cô lập nguyên nhân |
| “32 token cosine cao nghĩa là đã biết rank hiệu dụng thấp bao nhiêu” | Chỉ đo cosine; cần spectrum/effective-rank hoặc diversity theo attention để kết luận rank |

Không chỉnh sửa các ghi chép lịch sử trong phiên này; báo cáo đặt những giới hạn trên để tránh dùng nhận định mạnh hơn bằng chứng.

## 7. Hướng tối ưu có cơ sở nhất

**Ưu tiên 1 — giữ một phép so đáng tin:** thống nhất study/anchor, prompt schema, tokenizer/scorer và stop; gắn generation fingerprint với metric. Chấm lại output sẵn có nếu đủ provenance, không tự động train lại.

**Ưu tiên 2 — tìm nguồn của quyết định dừng sớm:** đối chiếu Z/A/C trên cùng val panel; đo xác suất token 106 ở đầu câu trả lời, ở sau câu đầu và khi còn finding chưa nhắc. Dùng cả teacher-forced loss và free generation. So đúng prompt train với prompt legacy để tách lỗi pipeline khỏi hành vi adapter.

**Ưu tiên 3 — tối ưu A direct trước:** vì nó cũng sinh ngắn và không phụ thuộc Stage-1. Chạy pilot với dữ liệu sạch/đồng nhất, checkpoint được chấm định kỳ, LR thấp hơn như **2e-5 so với 1e-4** trong một ablation. Đây là giá trị thử nghiệm, chưa phải cấu hình đã chứng minh tốt hơn. Nếu A chưa vượt base về chất lượng nội dung, tăng phức tạp ở C làm khó truy nguyên.

**Ưu tiên 4 — chỉ mở lại kênh bổ sung khi có giá trị đo được:** tách effect soft-token/cue/prompt bằng các nhánh train có matched budget; đo tráo ảnh/cue/soft-token đúng bệnh nhân. Mục tiêu là thông tin bổ sung, không chỉ thêm tham số.

**Ưu tiên 5 — Q-Former riêng nếu vẫn cần bridge:** cố định pipeline feature sau Stage-1, kiểm tra alignment/projection bằng một probe có dương tính rõ rồi mới mở rộng batch/GradCache hoặc train bridge. Không thay checkpoint Stage-1 dưới một projector Stage-2 cũ mà coi là cùng mô hình.

Một kiến trúc ứng viên cho các thử nghiệm sau là giữ native vision làm đường chính, bổ sung tín hiệu Stage-1 qua residual gate khởi tạo đóng và train bridge trước khi mở rộng LoRA. Nhưng gate cũng có thể học bỏ qua nhánh: phải có can thiệp tráo feature và gain val để chấp nhận. **Chưa có bằng chứng để đưa kiến trúc này thành mặc định.**

Các phép thử, thứ tự phụ thuộc, tiêu chí dừng và thay đổi code cần thiết được tách thành [THU_NGHIEM_TIEP_THEO.md](THU_NGHIEM_TIEP_THEO.md).

## 8. Điều nên làm và không nên làm từ kết quả này

| Quyết định | Lý do |
|---|---|
| Giữ zero-shot như baseline bắt buộc | Base đã có năng lực y khoa; mọi fine-tune phải chứng minh giá trị tăng thêm |
| Giữ stop `[1,106]` | Đã có can thiệp thực nghiệm xác lập lỗi stop cũ |
| Theo dõi độ phủ finding và độ dài cùng metric | CIDEr riêng lẻ đang ưu ái một hành vi cần kiểm tra thêm |
| Đánh giá generation giữa epoch | A chưa được chọn trên val; C chỉ có một điểm cuối epoch |
| Sửa parity/cohort trước khi so kiến trúc | Các producer hiện không thực thi cùng một hợp đồng |
| Không dùng `min_new_tokens` để che triệu chứng | Ép dài có thể thêm chữ mà không thêm finding đúng; chỉ dùng như probe riêng nếu cần |
| Không quét thêm hàng loạt ngưỡng cue ngay | Các probe hiện chưa cho gain nhất quán ngay cả khi precision cải thiện |
| Không kết luận “thêm epoch chắc sẽ tốt” | Không có trajectory cho thấy lúc nào model bắt đầu sụp về câu ngắn |
| Không bỏ qua A để chỉ sửa MHCAC | 679 ca ngắn chung chỉ ra vấn đề ở phần chung cũng đáng kể |

## 9. Nhật ký xác minh phiên này

- Đã đọc code, YAML, JSON/CSV tổng hợp và script sinh báo cáo tại máy dev; không chạy project tại checkout dev.
- Đã đối chiếu số test mới với bộ chấm nội bộ và kiểm tra chênh lệch METEOR corpus so với mean-of-study.
- Đã tra model card Google, báo cáo kỹ thuật MedGemma 1.5, mã nguồn BERTScore/METEOR và nghiên cứu đánh giá radiology report.
- Lệnh kết nối lần hai: `ssh -o BatchMode=yes -o ConnectTimeout=12 phuong@100.116.167.90 'hostname; git -C ~/Meta-CXR status --short; git -C ~/Meta-CXR rev-parse HEAD; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader'` → **exit 255**, `Connection timed out`. Không có traceback Python, không có số VRAM mới; lệnh phía host chưa chạy. Lỗi chỉ có ở output công cụ phiên này, không tạo raw log trên host.
- Chưa `git pull` trên host vì chưa SSH được; chưa kiểm chứng host đang ở revision nào. Mọi thử nghiệm tiếp theo phải pull/check lại theo [AGENTS.md](../../AGENTS.md).
- Không sửa loss/YAML/code training, không chạy test, không khởi chạy GPU, không tạo commit. Các thay đổi người dùng đang có trong `docs/evaluator_validation.md` và `.claude/skills/thesis-docs/` được giữ nguyên.

## 10. Nguồn nghiên cứu ngoài repo

- [Google: MedGemma 1.5 model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card): base đã học y khoa/MIMIC-CXR; input native 896×896 và 256 image token; nhạy với prompt.
- [Google: MedGemma 1.5 Technical Report](https://arxiv.org/html/2604.05081v1): MedSigLIP đóng băng trong quá trình phát triển phiên bản 1.5, decoder được huấn luyện tiếp, có post-training. Việc vision tower đóng băng trong repo tự nó không chứng minh “không học ảnh”.
- [BERTScore — mã nguồn/tài liệu tác giả](https://github.com/Tiiiger/bert_score): encoder chấm, hash cấu hình, baseline rescaling.
- [COCO METEOR — mã nguồn](https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/meteor/meteor.py): score corpus và score từng mẫu là hai output riêng.
- [Yu et al., Evaluating progress in automatic chest X-ray radiology report generation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499844/): quan hệ giữa metric và lỗi bác sĩ đánh giá; giới hạn của text overlap.

Các nguồn ngoài chỉ hỗ trợ cơ chế/cách đo. Các con số hiệu năng pipeline trong báo cáo đến từ artifact của dự án, không từ điểm benchmark Google.
