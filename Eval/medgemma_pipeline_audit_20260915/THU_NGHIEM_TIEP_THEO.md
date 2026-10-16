# Kế hoạch tối ưu MedGemma sau điều tra 15/09/2026

Đi cùng [báo cáo nguyên nhân](BAO_CAO.md). **Đây là kế hoạch, chưa phải kết quả thực thi.** Mục tiêu là cải thiện nội dung báo cáo và tìm nguyên nhân nhân quả, không chỉ làm một metric tăng.

## 1. Thứ tự thực hiện

| Thứ tự | Việc cần làm | Có cần train? | Quyết định nó giải đáp |
|---|---|---|---|
| P0 | Xác minh artifact, cohort, prompt, metric | Không | Bảng hiện tại có thực sự so cùng đối tượng và đúng checkpoint không? |
| P1 | Probe đầu vào và dừng sớm trên Z/A/C | Không, chỉ inference/forward | Sinh ngắn đến từ prompt, adapter, thiếu tín hiệu ảnh hay nhánh bổ sung? |
| P2 | Kiểm tra train trên tập nhỏ + trajectory A direct | Có, ngắn | Loop có cập nhật đúng và thời điểm nào hành vi sinh xấu đi? |
| P3 | Pilot A direct với một thay đổi mỗi lần | Có, giới hạn | LR/data/selection nào giữ khả năng của base và tăng độ phủ? |
| P4 | Ablation soft-token/cue trên nền A đã ổn | Có, matched budget | Stage-1 thật sự bổ sung thông tin gì? |
| P5 | Train Q-Former/bridge riêng nếu P4 cho lý do tiếp tục | Có | Alignment có phải nút thắt và có đáng chi phí không? |

Không chạy một chuỗi tự động từ P0 đến P5. Mỗi pha cần có kết quả và quyết định kỹ thuật trước pha sau; đây là dependency của thí nghiệm, không phải yêu cầu xác nhận cho từng lệnh đọc.

## 2. P0 — tạo một bộ đối chứng có thể tin được

### 2.1 Kiểm tra trên host khi kết nối hoạt động lại

Theo `AGENTS.md`, chỉ host được chạy project. Dùng Python `~/.venvs/meta-cxr-stage1-311/bin/python`. Trước mọi phép chạy phải kiểm tra checkout rồi `git pull --ff-only`; nếu dirty/diverged, xử lý bằng checkout riêng, không reset hay ghi đè thay đổi của phiên khác.

Các lệnh đọc ban đầu:

```bash
ssh phuong@100.116.167.90
cd ~/Meta-CXR
git status --short
git pull --ff-only
git rev-parse HEAD
findmnt -no FSTYPE,OPTIONS /mnt/drive1tb
nvidia-smi
pgrep -af 'pretraining.train|run_medgemma|train_eval_figure9|generate_stage2|evaluate_stage2|explain_stage2|probe'
```

Nếu GPU bận, không khởi chạy GPU/không xếp hàng. Nếu disk không đúng UUID/driver/mode, không tự mount bằng device name. Việc kiểm tra CPU artifact có thể tiếp tục nếu không tranh tài nguyên đáng kể với job đang chạy.

Artifact cần xác minh, chỉ xuất aggregate/provenance:

- `/home/phuong/thesis_eval_20260913/gen_zeroshot/`, `gen_armA/`, `gen_armC/`: generation summaries, decoder config và checksum file dự đoán.
- `/home/phuong/ft_only_full/adapters/medgemma_qlora_medgemma_direct/`: manifest, meta, số update thực tế của checkpoint.
- `/home/phuong/ft_guided_full/`: tìm đúng adapter C được generation summary tham chiếu; không đoán theo tên thư mục nếu artifact chỉ sang chỗ khác.
- `/home/phuong/cue4_fixedstop_20260909/`, `/home/phuong/stage2_stop_probe_20260909/`: nguồn probe cũ, dùng đối chiếu cơ chế.

Không in raw text của prediction JSONL hoặc sample identifier. Không `cat` toàn bộ manifest/checkpoint nếu chưa biết schema có chứa dữ liệu nhạy cảm; chọn whitelist trường.

### 2.2 Hợp đồng dữ liệu và provenance cần xuất

Viết một script audit chạy host, giữ mapping định danh nội bộ. Output chỉ số tổng hợp:

| Nhóm | Trường cần có |
|---|---|
| Nguồn model | Base model revision, adapter hash, projector hash, Stage-1 checkpoint hash nếu dùng |
| Quá trình sinh | Commit thực thi, prompt/template hash, stop IDs, quantization/dtype, token cap, repetition controls |
| Dữ liệu | Số image rows, unique studies, unique subjects, anchor policy, số ca giao nhau/bị loại |
| Ghép cặp | Số duplicate keys, mismatch reference, mismatch anchor, missing predictions; tất cả là count |
| Bộ chấm | Tokenizer, metric implementation/version, BERTScore encoder/layer/IDF/rescale/hash |
| Độ dài | Word tokenizer được dùng, p10/p50/p90, share≤10, share≤5, top1/top5, empty, reached cap, stop reason |
| Hoàn thành train | Actual updates, actual valid targets, epoch_completed, validation checkpoints thực sự có |

**Điều kiện qua:** một cohort master, một anchor mỗi study, reference giống nhau khi ghép cặp, không duplicate, không silent missing; đầu ra metric có thể truy tới checksum generation và cấu hình model. Không dùng `n` hoặc `status=complete` thay các kiểm tra này.

Nếu artifact gốc không khớp bản copy ở dev: giữ cả hai, ghi rõ phiên bản nào dùng cho từng bảng; không ghi đè bảng cũ. Nếu thiếu generation provenance, tạo cohort val mới để probe, không gán lại ý nghĩa cho kết quả lịch sử.

### 2.3 Sửa các lỗi hạ tầng có phạm vi hẹp

Trước sửa source, đọc `struct/` của component. Mỗi thay đổi hành vi phải cập nhật `CLAUDE.md`, `README.md` tiếng Việt và trang `struct/` tương ứng trong cùng commit theo quy định repo.

1. Một selector study/anchor cho train và generation, giữ native path độc lập với LAVIS. Thêm metadata `anchor_view` nhất quán.
2. Generation kiểm tra prompt config và cue rule với metadata adapter. C full cũ cần `--cue-rule conditional_positive` khi tái hiện; default marginal hiện tại là một điều kiện khác. Override phục vụ probe phải được ghi rõ, không lặng lẽ dùng prompt/cue khác.
3. Cache export Stage-1 có transform xác định, tách khỏi augmentation training; lưu transform/seed/hash vào cache identity. Không thay cache dưới adapter cũ rồi coi là inference nguyên trạng.
4. Một evaluator canonical hoặc hai protocol được đặt tên rõ. Không bắt exception rồi trả metric 0.
5. Tách `meteor_corpus`/`meteor_sample_mean`; bootstrap đúng đại lượng. Giữ CIDEr IDF policy nhất quán trên cohort. Bổ sung khoảng tin cậy theo subject để kiểm tra độ bền.
6. Khi resume giữa epoch không được hỗ trợ, fail rõ. Nếu triển khai resume thật thì lưu permutation/offset và kiểm tra lần resume không bỏ mẫu/đọc lại mẫu ngoài chủ ý.

Tests cần có, chạy trên host: hai ảnh cùng study chỉ sinh một anchor; hai record producer tạo cùng prompt; thiếu prompt config bị phát hiện; cache rebuild xác định; metric failure không trở thành 0; CI và point estimate cùng estimand; resume giữa epoch không thể hoàn thành với 0 updates mới ngoài chủ ý. Đây là regression tests cho hành vi, không phải test chỉ lặp lại implementation.

## 3. P1 — probe cơ chế trên checkpoint hiện có

### 3.1 Panel chung

Chọn trước một panel **200–300 ca val hợp lệ** với một ảnh anchor mỗi study. Phân tầng PA/AP, độ dài tham chiếu, có/không nhãn bất thường, có/không support devices, nguồn trích xuất. Không chọn panel bằng cách nhìn output đẹp/xấu trước. Dùng cùng panel cho mọi can thiệp; báo cáo đây là development val đã được dùng trong dự án, không phải test mới.

Giữ cố định model revision, dtype, processor, decoding greedy, stop `[1,106]`, cap 160. Nếu thử thay cap, đó là một probe riêng. Không đặt minimum length để che kết thúc sớm.

### 3.2 Các can thiệp tối thiểu

| Probe | So sánh | Giữ nguyên | Dấu hiệu giúp phân biệt nguyên nhân |
|---|---|---|---|
| G1: prompt | Prompt đúng lúc train vs legacy, trên từng adapter | Trọng số, ảnh, decoder | Nếu chỉ thay về prompt đúng khôi phục độ phủ, ưu tiên sửa inference |
| G2: độ phủ | Prompt hiện tại vs prompt yêu cầu đầy đủ các finding có bằng chứng, bỏ giới hạn 1–4 câu | Checkpoint, ảnh, cue | Nếu clinical recall tăng và output bớt câu mẫu, prompt có đóng góp; thêm chữ đơn thuần chưa đủ |
| G3: ảnh | Ảnh đúng vs ảnh khác cùng loại view | Prompt và adapter | Đo tăng NLL tham chiếu + thay đổi nội dung report, đặc biệt trong nhóm sinh ngắn |
| G4: cue | C đúng cue vs bỏ block vs tráo cue giữa ca | Native image, 32 soft token, checkpoint | Đo output-change rate, finding changes, clinical recall/precision; không chỉ mean BERTScore |
| G5: soft | C đúng soft token vs tráo soft token giữa ca | Native image, cue, prompt | Có đọc tín hiệu gắn với ca không? Zero-feature thêm như probe OOD phụ |
| G6: định dạng số | Cùng adapter/base trong NF4 và bf16 | Cohort, prompt, decoder | Chênh lệch train/inference precision có góp phần gây EOS sớm không? |

G3–G5 là **can thiệp lúc inference**, không thay thế ablation train lại; một model đã học với kênh nào đó có thể phản ứng với input OOD. Tráo giữa ca cũng phá tương quan hai kênh nên phải ghi rõ.

Thứ tự để giảm chi phí: 32 ca kiểm tra wiring G1/G3; sau đó mở rộng G1 và G2 trên panel; chỉ chạy G4/G5 nếu C vẫn kém A sau parity fix. G6 dùng 32–64 ca trước, chỉ mở rộng khi thấy thay đổi rõ. Mọi số ca nhỏ là probe, không phải kết quả công bố cuối.

### 3.3 Đo riêng quyết định EOS

Trên prefix tham chiếu đã cố định, log **aggregate** xác suất/rank của token 106 ở đầu câu trả lời, sau câu đầu, giữa target và tại kết thúc thật. Đồng thời ghi vị trí dừng thực tế khi free generation. So Z/A/C trên chính các ca này.

- Nếu A/C nâng xác suất dừng sớm dù target còn nội dung, adapter đã thay đổi phân bố dừng.
- Nếu chỉ prompt legacy làm tăng EOS sớm, parity là nguyên nhân cụ thể.
- Nếu teacher-forced prediction tốt nhưng free generation sớm vào câu mẫu, ưu tiên trajectory/checkpoint và exposure bias; chưa nên kết luận dataset có quá nhiều câu ngắn.
- Nếu tráo ảnh gần như không tác động ở nhóm sinh ngắn nhưng tác động nhóm dài, điều tra riêng việc model dựa vào mẫu ngôn ngữ ở nhóm ngắn.

Không kết luận “không đọc ảnh” từ attention weight nhỏ hoặc từ một CI chứa 0. Đặt biên hiệu ứng có ích trước khi đánh giá tương đương.

## 4. P2 — xác minh loop và quan sát lúc mô hình bắt đầu lệch

### 4.1 Overfit nhỏ để kiểm tra hạ tầng

Lấy **16–32 ca train** có target hợp lệ, gồm cả báo cáo dài/ngắn và có/không bất thường. Đây là kiểm tra wiring, không đánh giá generalization.

Đo trước update đầu và sau một số update cố định:

- Số tham số nhận gradient và norm gradient theo LoRA/projector; không đòi mọi LoRA matrix có gradient ngay step đầu vì initialization có thể làm một nhóm gradient bằng 0.
- Checksum/delta norm trọng số so với **chính checkpoint khởi tạo**, không so với một model random mới.
- Số target token không bị mask; tỷ lệ EOS trong supervised target; prompt/image token phải được mask khỏi labels.
- Loss train và độ khớp target free-generation trên tập nhỏ.
- A với LoRA mới trước update phải cho hành vi phù hợp với base ở cùng dtype, eval mode; nếu đã khác lớn trước học, xử lý implementation trước.

Nếu tập nhỏ không thể học, dừng pilot lớn để kiểm tra gradient/mask/processor. Nếu học được nhưng val mất độ phủ, vấn đề chuyển sang mục tiêu/generalization/recipe.

### 4.2 Evaluation theo update, không chỉ cuối epoch

Trong pilot kế tiếp, lưu/đánh giá ở update **0, 50, 100, 250, 500** nếu ngân sách đủ. Dùng cùng panel val nhỏ và cố định. Track EOS, share≤10, top5, NLL, NLG và clinical metrics nếu đã có scorer.

Tại mỗi mốc phải ghi actual valid targets đã thấy. Một epoch là số lượt qua dataset; không dùng epoch thay số optimizer update khi so cohort khác nhau.

Thời điểm share≤10/top5 tăng đồng thời clinical recall giảm là điểm cần kiểm tra LR, dữ liệu và lựa chọn checkpoint. Nó giúp tránh chạy hết 70–80 giờ mới thấy vấn đề.

## 5. P3 — tối ưu A direct bằng pilot có kiểm soát

### 5.1 Chuẩn bị dữ liệu

Chọn **2.000 target train hợp lệ** cho sàng lọc, mở rộng **8.000 target hợp lệ** cho phương án qua gate. Lấy mẫu chung ở mức study; những ca bị loại do lỗi target phải thay bằng ca khác trước khi đóng cohort.

Audit theo nguồn extraction: header/technique-only, target rỗng sau cleaning, truncation, số câu, prior-dependent content. Không xóa báo cáo ngắn chỉ vì ngắn: một báo cáo bình thường ngắn vẫn có thể đúng. Với temporal clause, ưu tiên đánh dấu và phân tích trước; không xóa tự động mọi câu chứa từ “stable”.

Tách val dùng development, val dùng kiểm tra phương án đã chọn, và test. Nếu val trước đây đã được xem nhiều lần, công bố điều đó; một lần đánh giá trên test chính thức đã xem trước không trở thành “test chưa từng dùng”.

### 5.2 Các arm tối thiểu

| Arm | Thay đổi so với control A mới | Câu hỏi |
|---|---|---|
| A0 | Base trước train, cùng prompt/dtype | Mức cần vượt là gì? |
| A1 | Recipe hiện tại, dữ liệu/cohort đã chuẩn hóa | Tái hiện sự ngắn đi ở budget nhỏ được không? |
| A2 | Chỉ giảm LoRA LR từ 1e-4 xuống **2e-5** | Cập nhật nhẹ có giữ độ phủ/tri thức của base tốt hơn không? |
| A3 | Chỉ đổi chính sách target chất lượng theo audit, cùng LR A1 | Nhiễu target có đóng góp không? |

Giá trị LR trên là **đề xuất ablation**, chưa đo trên pipeline. Giữ rank/alpha, effective batch, số update, preprocessing và decoding giống nhau. Không gộp A2/A3 rồi quy gain cho một yếu tố. Sau sàng lọc, lặp phương án tốt ở tối thiểu 3 seed trước quyết định mở rộng.

**Tiêu chí chọn:** ưu tiên clinical completeness/correctness; nếu scorer chưa có, NLG chỉ là kết quả tạm thời, kèm kiểm tra mẫu và mọi chỉ báo độ dài/đa dạng. Không chọn theo CIDEr riêng hoặc val loss riêng. Không tăng độ dài bằng cách ép model viết tiếp sau terminator.

Ngân sách tham khảo: historical batch 2 chạy khoảng 3,3 s/iteration; 2.000 target ~1.000 iterations, gần 1 giờ **phần train mỗi arm**, 8.000 target gần 4 giờ. Đây là ước tính thô, chưa gồm load/cache/eval và chưa đo lại trên host. Phải đo smoke rồi mới ước tính chính thức.

## 6. P4 — kiểm tra Stage-1 bổ sung được gì

Chỉ sau khi có direct baseline A ổn định. Dùng cùng ảnh native, cùng target/cohort, common instruction và cùng lịch train. Tách các yếu tố:

| Arm mới | Native image | Soft token | Text cue |
|---|---|---|---|
| N | Có | Không | Không |
| N+S | Có | Có | Không |
| N+Cue | Có | Không | Có |
| N+S+Cue | Có | Có | Có |

Repo hiện chưa cung cấp một nhánh native+cue độc lập hoàn chỉnh qua producer native; cần wiring riêng, không chạy config `native_anchor_guided` rồi coi là đã có cue. Common instruction phải thật sự giống nhau; bỏ block cue nhưng vẫn giữ instruction guided chưa cô lập hoàn toàn text channel.

Kiểm tra riêng:

1. Chất lượng cue train/val/test, số cue/ca, phân phối confidence, precision/recall dưới cùng label framing. Nếu cue train quá lạc quan, cân nhắc cross-fitting trên train, không dùng test để fit.
2. Norm/RMS của soft token projected so với native/text embedding; đo ở init và sau train, không giả định norm sai chỉ vì thiếu LayerNorm.
3. Độ nhạy với tráo feature theo ca và gain clinical trên các finding MHCAC dự đoán tốt nhưng native bỏ sót.
4. Oracle cue từ nhãn report trên **val**, chỉ làm upper-bound diagnostic và gắn nhãn oracle; không báo như hệ thống chạy thực tế và không dùng để tạo lợi thế ở test.

Nếu oracle cue còn không giúp, ưu tiên cách model sử dụng cue/prompt/objective. Nếu oracle giúp nhưng predicted cue không giúp, ưu tiên chất lượng cue và sự khác biệt train–eval. Một oracle 13 nhãn cũng không chứa đầy đủ vị trí/mức độ, nên không coi đó là upper bound cho toàn bộ report.

Chỉ giữ nhánh bổ sung khi có gain đủ lớn theo tiêu chí đặt trước và không làm sụt độ phủ/factuality. Các pilot trước với CI chứa 0 không đủ để “xác nhận tương đương”, nhưng cũng không đủ để chi ngân sách full run cho cùng thiết kế mà không có giả thuyết mới.

## 7. P5 — nếu tiếp tục với Q-Former riêng

Phương án cần thử có ranh giới rõ:

1. Chốt Stage-1 classification checkpoint; đóng băng **toàn bộ** pipeline tạo feature, gồm encoder, adapter, view fusion, shared projection.
2. Export feature xác định, train Q-Former/bridge trên feature cố định. Nếu MHCAC vẫn cần text teacher cũ, giữ bản teacher tương ứng để tránh đổi hành vi MHCAC theo Q-Former mới.
3. Kiểm tra alignment trên val bằng retrieval/rank, shuffled-pair control và tác động downstream. Loss đi xuống một mình không phải gate.
4. Nếu thử contrastive batch lớn với GradCache, giữ rõ logical contrastive batch so với microbatch/gradient accumulation. Không lặp cấu hình đã về chance rồi hy vọng train lâu hơn tự sửa.
5. Train lại projector/Stage-2 tương thích với Q-Former mới. Không đưa feature mới qua `img_proj` cũ rồi chấm như một thay đổi riêng Stage-1.

Phương án này tốn thêm một chuỗi thí nghiệm. Nó hợp lý khi P4 cho thấy alignment là nút thắt, không phải mặc định chữa mọi kết quả NLG thấp.

## 8. Đầu ra bắt buộc sau mỗi pha

Mỗi pha có một handoff `docs/handoff/PLAN-<ngày>-<topic>.md` trước khi chạy và Execution report sau khi chạy, theo quy định repo. Ghi:

- Câu hỏi nhân quả, yếu tố thay đổi, yếu tố giữ nguyên.
- Lệnh chính xác và exit status; revision thực thi.
- Actual cohort/updates và checks uniqueness/reference parity.
- Bảng metric cùng CI đúng estimand; bảng EOS/độ dài/đa dạng; clinical metric unavailable nếu chưa có.
- Kết luận: chấp nhận, bác bỏ trong phạm vi thử, hoặc chưa phân biệt được; hành động tiếp theo.
- Vị trí private raw log/artifact trên host, không kèm văn bản báo cáo bệnh nhân.

Output directory và master log phải mới; kiểm tra trước shell redirection, dùng cơ chế tạo độc quyền/noclobber. Long run dùng `setsid nohup`, chỉ launch một lần sau fresh GPU/job checks. OOM là kết quả về capacity: ghi batch/accum/VRAM và dừng; không âm thầm halve batch vì điều đó thay thí nghiệm.

## 9. Bước tiếp theo cụ thể nhất

**Bắt đầu bằng P0 và G1 của P1, sau đó mới quyết định pilot A2.** Đây là đường ngắn nhất để phân biệt một adapter học lệch với một phép inference dùng sai hợp đồng. Kết quả cần có trước khi train dài là: cohort/prompt đúng, model thực sự dùng ảnh, và đã thấy trajectory cho biết lúc nào output bắt đầu co về câu mẫu.
