> Source: `training/explainability/sentence_attribution.py` (326 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-30

# `training/explainability/sentence_attribution.py`

## Purpose

Tách báo cáo sinh ra thành câu, gắn nhãn, gộp token→câu, tính `mean_token_nll`
và **parse coverage**.

Thư viện chuẩn cộng `safety.claims`. Không torch, không model, không tokenizer —
token đến dưới dạng chuỗi đã decode, NLL đến dưới dạng dãy float.

## Hai labeler — `lexicon_v2` là mặc định từ 2026-08-30

| | `lexicon_v1` | `lexicon_v2` (mặc định) |
|---|---|---|
| Nhãn | 14 CheXpert | 14 + synonym bổ sung + 9 nhãn ngoài |
| `parse_coverage` toàn val | 0.4827 | **0.6477** |
| Tầng | luôn `chexpert_14` | `chexpert_14` **hoặc** `extended` |

⚠ **Chỉ 61% mức tăng là kiểm chứng được.** Trong 1.285 câu v2 gắn thêm: 782
thuộc 14 nhãn (Stage 1 dự đoán được, +0.100 coverage) và 473 nằm ngoài
(không gì dự đoán, +0.065). Đọc trường `tier` trước khi coi một nhãn là claim
đã kiểm.

⚠ **`safety/claims.py` KHÔNG bị sửa** — 14 nhãn ở đó khớp 1-1 với đầu phân loại
Stage 1 và `safety/pipeline.py` đối chiếu claim với chính nó. Thêm nhãn ở đó sẽ
sinh claim không có gì để kiểm. Synonym gốc lấy **theo tham chiếu**, không copy.

⚠ 9 nhãn mở rộng là **đề xuất**, cần bác sĩ duyệt trước khi công bố.

`--labeler lexicon_v1` tái lập mọi kết quả ghi trước 2026-08-30; mọi artifact
đều ghi labeler đã dùng.

## ⚠ Nguồn nhãn: KHÔNG phải labeler đã huấn luyện

Repository này **không** implement labeler lâm sàng đã huấn luyện nào.
`training/evaluation/clinical.py` cố ý ném lỗi thay vì trả về điểm bịa, và chính
sách đó áp cả ở đây.

Nhãn câu đến từ `LexiconSentenceLabeler` — bộ khớp từ đồng nghĩa + cue phân cực
tất định trên 14 nhãn bất thường của repo, adapter mỏng bọc
`safety.claims.LexiconClaimParser`. Nó báo tên mình là `lexicon_v1` và **không
bao giờ được trình bày như một labeler đã huấn luyện**.

`SentenceLabeler` là Protocol để thay thế sau. **Không có adapter cho labeler đã
huấn luyện trong branch này**, có chủ ý.

## ⚠ `parse_coverage` phải xuất hiện cạnh mọi kết quả

Giới hạn của bộ khớp từ vựng là thứ chịu lực, nên nó được **đo**, không phải giả
định. `parse_coverage` = tỉ lệ câu sinh ra được ít nhất một nhãn, mang trên **cả
hai cấp**: từng study (`StudyAttribution.parse_coverage`) và toàn dataset
(`dataset_parse_coverage`).

Một run có coverage 0.3 đã gắn nhãn ba câu trên mười, và **mọi kết luận ở mức
câu rút ra từ nó đều bị chặn bởi con số đó**. Trích nó bên cạnh mọi kết quả.

### Gộp theo CÂU, không theo study

`dataset_parse_coverage` trả `parse_coverage` gộp theo câu, và
`mean_study_parse_coverage` bên cạnh. Trung bình các phân số study cho một study
1 câu cân bằng với một study 12 câu. Test chốt trường hợp hai con số lệch nhau
(0.5 so với 0.667) để lý do này không bị quên.

## ⚠ Câu không parse được: GIỮ, không lọc bỏ

Chúng vẫn nhận attribution map — model có sinh ra chúng, và nó nhìn vào đâu vẫn
là một sự thật — nhưng mang `spatially_meaningful=False`, đồng thời liệt kê
trong `unparsed`. Người đọc không được phép nhầm một câu chưa gắn nhãn thành một
finding có cơ sở.

## `locate_sentences` — không sinh ra bộ tách câu thứ hai

`safety.claims.split_sentences` là bộ tách của repo, nhưng nó `strip()` từng
mảnh nên mất offset — thứ cần để ánh xạ token lên câu. Thay vì viết lại phép
tách (rồi để hai bộ trôi khỏi nhau), hàm này **gọi** nó rồi đi lại chuỗi gốc để
khôi phục span. Pin bằng `test_locate_sentences_agrees_with_the_existing_splitter`
trên 5 loại input.

Con trỏ `cursor` là bắt buộc: `str.find` không có cursor sẽ trả về lần xuất hiện
đầu tiên hai lần khi báo cáo có câu lặp lại, và gộp hai span lên nhau.

## `align_tokens_to_sentences` — chỗ lỗi im lặng nhất module

Token nào thuộc câu nào được quyết bằng **độ chồng lấn ký tự**. Token vắt qua
ranh giới thuộc về câu chồng lấn nhiều hơn; hoà thì về câu trước, nên phép gán
tất định. Token không chồng lấn câu nào (khoảng trắng giữa câu) không thuộc câu nào.

Sai một đơn vị ở đây hoàn toàn im lặng: mọi con số phía sau vẫn tính ra, chỉ là
một câu mang độ bất định và heatmap của câu khác. Vì thế `attribute_sentences`
**từ chối** khi `len(token_nll) != len(token_texts)`, và vòng lặp dựng record
dùng `zip(..., strict=True)`.

## Main items

| Item | Vai trò |
|---|---|
| `SentenceLabel(finding, polarity)` | Một finding + phân cực P/N/U |
| `SentenceLabeler` | Protocol; `name` được ghi nguyên văn vào mọi record |
| `LexiconSentenceLabeler` | `name = "lexicon_v1"` |
| `SentenceRecord` | index, text, char span, token_indices, labels, `spatially_meaningful`, `mean_token_nll` |
| `StudyAttribution` | sentences, labeler, `parse_coverage`, `unparsed` |
| `dataset_parse_coverage(studies)` | Gộp theo câu + trung bình theo study |
| `locate_sentences(text)` | `(sentence, start, end)` |
| `align_tokens_to_sentences(token_texts, spans)` | Một tuple chỉ số token cho mỗi câu |
| `attribute_sentences(text, token_texts, token_nll, labeler)` | Entry point |

## `mean_token_nll = None` ≠ `0.0`

`None` nghĩa là **chưa đo**; `0.0` nghĩa là model rất chắc chắn. Cùng quy ước mà
`safety.claims.Claim` dùng cho các score của nó.

## Error / edge cases

- `token_nll` lệch độ dài `token_texts` → `ValueError` nêu rõ hậu quả misalignment.
- Đối tượng không thoả Protocol → `TypeError`.
- Text rỗng → 0 câu, `parse_coverage = 0.0` (không phải 1.0).
- Câu không có token nào → `mean_token_nll = None`.

## Calls / Called by

- Calls: `safety.claims` (`split_sentences`, `LexiconClaimParser`,
  `unparsed_sentences`). Có import shim hai nhánh theo quy ước `training/`.
- Called by: `attention_capture.py` (chưa tồn tại).

## Related tests

`tests/explainability/test_sentence_attribution.py` — 28 test.
`test_the_only_labeler_name_ever_emitted_is_on_the_allowlist` viết theo kiểu
**allowlist** chứ không đi tìm chuỗi cấm, để tên của labeler đã huấn luyện mà dự
án KHÔNG implement cũng không xuất hiện trong source. Labeler mới phải được thêm
vào allowlist một cách có chủ ý — đó là điểm của nó.
