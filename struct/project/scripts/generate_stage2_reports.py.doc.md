> Source: `scripts/generate_stage2_reports.py` (412 dòng)
> Status: ✅ ACTIVE — sinh báo cáo, ghi `.jsonl` cho `evaluate_stage2.py`
> Last verified against source: 2026-09-08

# `scripts/generate_stage2_reports.py`

## Purpose

Sinh FINDINGS bằng MedGemma và ghi ra đúng định dạng `.jsonl` mà
`evaluate_stage2.py` đọc. Dùng lại chính engine Stage-2 của dự án
(`VariantLLM`) chứ không viết một bản song song, nên prompt, chat template và
generation kwargs **giống hệt** lúc train.

## Hai nguồn record, chọn bằng `--pipeline-mode`, không bao giờ trộn

| Nguồn | Mode | Cách dựng record | Cần gì |
|---|---|---|---|
| Split CSV | `medgemma_direct` | pandas đọc CSV, join `--image-root` | `--manifest`, `--image-root` |
| Stage 1 | mọi `meta_cxr_*` | `build_stage1_records` load checkpoint Stage-1, sinh 32 soft token + cue P/N/U của MHCAC | `--checkpoint-root`, `--adapter` |

Nhánh Stage-1 **không dùng** `--manifest` / `--image-root`: record của nó đã mang
`image_path` **tuyệt đối** (dataset đã join `vis_root`). Đây chính là cái bẫy đã
ghi ở [`probe_soft_tokens.py.doc.md`](probe_soft_tokens.py.doc.md) — join thêm
một lần nữa sẽ ra đường dẫn sai.

Import LAVIS nằm **trong** `stage1_records()`, không ở module scope, để giữ bất
biến của `tests/test_native_independence.py`.

## ⚠ Hai cohort KHÁC NHAU — `--limit` giống nhau không làm chúng so sánh được

| | nhánh native | nhánh Stage-1 |
|---|---|---|
| Lọc view | chỉ PA/AP (`--frontal-only`) | không lọc |
| Chọn mẫu | `DataFrame.sample(random_state=seed)` | thứ tự dataset, lọc theo `generation_mask` |

So arm A với arm C trên "3.102 study mỗi bên" sẽ lặp lại đúng sai lầm dự án đã
mắc hai lần. **`--restrict-to <earlier.jsonl>`** là cách làm cho phép so sánh:
nó giữ lại đúng những `sample_key` đã có trong file cũ.

`sample_key` = `blake2b(dicom_id, 12)`, tính từ **stem của tên file ảnh**, nên
**giống nhau trên cả hai nhánh** và giống hệt giá trị các lần chạy native cũ đã
ghi — file `.jsonl` cũ vẫn join được.

## Bốn refusal, tất cả xảy ra TRƯỚC khi import torch

`validate_invocation()` chạy ngay sau khi resolve mode. Trước đây một cờ thiếu
chỉ lộ ra **sau** khi đã import torch/transformers/nltk — trên training host là
mất một phút GPU cho mỗi lần gõ nhầm.

| Điều kiện | Vì sao là lỗi cứng, không phải cảnh báo |
|---|---|
| native thiếu `--manifest`/`--image-root` | không có nguồn record |
| `meta_cxr_*` thiếu `--adapter` | **`img_proj` khởi tạo ngẫu nhiên** — sinh ra báo cáo trôi chảy nhưng mô tả nhiễu, không hề báo lỗi |
| `--adapter` không có `img_proj.pt` | như trên. `load_img_proj_if_present()` **im lặng return** khi thiếu file, nên không thể dựa vào nó |
| `native_qformer` thiếu `--prompt-config` | placeholder soft token do prompt builder v2 sinh; instruction legacy không sinh cái nào → phép thay thế tìm thấy 0 vị trí và lặng lẽ chạy như native MedGemma thuần |

`both_for_ablation` cũng bị từ chối: hai arm sẽ ghi đè cùng một
`generated_test.jsonl` — đúng loại va chạm đã khiến hai tiến trình chia chung một
output path ngày 2026-08-19.

`SOFT_TOKEN_IMAGE_MODES` được nhân bản trong module này để kiểm tra sớm mà không
cần import nặng; `main()` so nó với `SOFT_TOKEN_MODES` thật ngay sau import và
`RuntimeError` nếu lệch, còn
`tests/test_generate_stage2_reports.py::test_the_local_soft_token_set_matches_the_engine`
ghim hai bên ở nơi import được.

## `--stage1-cache-dir` — tiết kiệm ~73 phút

`build_stage1_records` cache vào `<dir>/.sensitive_stage1_cache`. Trỏ cờ này vào
**thư mục output của lần train** thì lần sinh báo cáo dùng lại đúng pass encode
đó thay vì mã hoá lại toàn split.

Cache chỉ trúng khi khớp `cohort_id`, tức cùng `--checkpoint-root`,
`--stage1-run` **và** cùng `sample_limit`. Vì thế script luôn gọi
`build_stage1_records` với `sample_limit=None` và áp `--limit` **sau đó**, trên
danh sách record — thu hẹp ở tầng dưới sẽ trượt cache.

## Chống lặp — `--no-repeat-ngram-size`, `--repetition-penalty`

Cả hai **mặc định TẮT**. Mọi con số Stage-2 đã ghi đều sinh ra khi không có
chúng, nên bật mặc định sẽ âm thầm làm kết quả cũ không tái lập được.
`VariantLLM.generate` nhận chúng dạng keyword-only và giữ `generate_kwargs`
y hệt cũ khi không truyền.

**n=5 chọn từ dữ liệu, không phải theo thói quen.** Tỷ lệ báo cáo chứa ít nhất
một n-gram lặp, đo trên 3.102 study test:

| n | báo cáo THẬT | arm A sinh ra | nhận xét |
|---|---:|---:|---|
| 3 | **22,0%** | 59,3% | phạt oan 1/5 báo cáo thật |
| 4 | 6,7% | 51,5% | |
| **5** | **2,4%** | **44,6%** | **điểm vận hành, ~19:1** |
| 6 | 1,2% | 41,9% | |

Chặn dưới 5 là tính tiền bác sĩ vì cách hành văn bình thường — một báo cáo viết
"there is no evidence of ..." hai lần là chuyện thường.

⚠ `--repetition-penalty` rescale **mọi** token đã xuất hiện, kể cả thuật ngữ
lâm sàng mà báo cáo lặp lại một cách chính đáng. Ưu tiên `--no-repeat-ngram-size`.

## `--cue-rule` — cách dự đoán MHCAC thành cue P/N/U

Chỉ áp dụng cho các mode `meta_cxr_*`. Đo trên file dự đoán của
`run_20260820_ft`, macro 13 nhãn, framing `study_presence`, ngưỡng fit trên val:

| Giá trị | Precision | Recall | Cue/study |
|---|---:|---:|---:|
| `conditional_positive` (mặc định) | 0,1887 | 0,8097 | 8,46 |
| `mention_gated` | 0,2357¹ | 0,4364¹ | 2,30¹ |
| `marginal_positive` | **0,4069** | 0,3135 | **1,46** |
| `none` | — | — | 0 |

¹ chỉ trên val. Thực tế mỗi study có **~1,6** bệnh.

`conditional_positive` sắp xếp cả 13 nhãn chỉ dựa trên `q` — thang trả lời
*"NẾU bác sĩ có nhắc thì dương hay âm?"*. Vì 79,5% ô nhãn để trống và bị mask
khỏi loss, đầu này chưa từng thấy "không xuất hiện trong báo cáo", nên với bệnh
hiếm nó suy biến thành hằng số "luôn nói có".

`marginal_positive` đặt ngưỡng `sigmoid(m) · q_pos` **theo từng nhãn** và **chỉ
phát nhóm positive** — không khẳng định bệnh nào vắng mặt, vì cue âm sai có thể
dập tắt một bệnh có thật và dự án chưa đo gì về chất lượng cue âm. Ngưỡng ở
`configs/stage2_cue_thresholds_marginal_pfit.json`.

`none` phát nhóm rỗng nhưng **giữ nguyên hình dạng prompt guided và soft token**
— dùng để tách riêng ảnh hưởng của cue. Nhóm rỗng là một dự đoán hợp lệ; record
thiếu hẳn key mới là thứ prompt builder từ chối.

⚠ **Đổi `--cue-rule` làm đổi luôn `cohort_id`**, nên cache Stage-1 sẽ dựng lại.
Đó là chủ đích: `pred_groups` nằm trong cache, và dùng lại cache của quy tắc
khác sẽ cho cue sai mà không báo lỗi.

⚠ **Cue tốt hơn KHÔNG cải thiện được Stage 2.** Đo 2026-09-08 trên đúng 100
study val, một checkpoint arm C: `none` − `conditional` là +0,0115
[+0,0025; +0,0217] BERTScore (có ý nghĩa), còn `marginal` − `conditional` là
+0,0073 [−0,0008; +0,0166] (**chưa xác lập**). Xem `CLAUDE.md`.

## Một sửa lỗi kèm theo

`--frontal-only` trước đây là `action="store_true", default=True`, nghĩa là
truyền hay không truyền đều ra `True` — **không có cách nào tắt bộ lọc**. Nay là
`BooleanOptionalAction`, nên `--no-frontal-only` hoạt động; mặc định giữ nguyên.

## Privacy

Text báo cáo sinh ra và text tham chiếu đều là dẫn xuất PhysioNet. Output đi qua
`_assert_private_output_location`, tên file không mang identifier, mỗi dòng khoá
bằng `sample_key` blake2. **`.jsonl` và `.sensitive_stage1_cache/` là dữ liệu
bệnh nhân — để ngoài repo.**

## Callers

`scripts/pipeline_stage2_rest.sh` STEP 3/5 (arm A, `medgemma_direct`, ngầm định).

## Callees

`training/pipeline_modes.resolve_pipeline_modes` ·
`training/train_eval_figure9_llm_variants_200.VariantLLM` /
`build_stage1_records` · `training/run_context.Stage1Context` ·
`stage2.prompts.load_prompt_config` ·
`scripts/evaluate_explanation._assert_private_output_location`

## Related documentation

[`scripts/_index.md`](_index.md) ·
[`evaluate_stage2.py.doc.md`](evaluate_stage2.py.doc.md) — đọc `.jsonl` này ·
[`probe_soft_tokens.py.doc.md`](probe_soft_tokens.py.doc.md) ·
[DECISIONS.md](../_meta/DECISIONS.md)

← [Về `scripts/`](_index.md)
## Correction: withheld is not normal (2026-09-08)

Selective marginal artifacts from `calibrate_cue_precision.py` carry per-label
`positive_enabled`. Use `--cue-rule marginal_positive --threshold-path <artifact>`
with a matching guided prompt. The shared record builder rejects incomplete
selective artifacts or a mismatching rule; disabled labels abstain, never become
negative cues. Generation benefit remains unverified pending the matched probe.

`none` hiện ghi cue_state=not_provided và bỏ structured block; rỗng từ marginal/gated là abstained, cũng không phát câu normal. Đây là thay đổi so với bảng metric lịch sử ở trên: none cũ vẫn phát normal summary, không tương đương probe xoá block. Non-default rule yêu cầu matching guided --prompt-config. Training cũng nhận --cue-rule và truyền cùng rule cho cả ba split.
## Stop-token provenance

`generation_summary.json` includes effective `eos_token_id` (scalar/list),
preserving model chat terminators instead of overriding them with tokenizer EOS.
