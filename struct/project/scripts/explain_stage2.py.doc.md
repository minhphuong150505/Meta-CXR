> Source: `scripts/explain_stage2.py` (~330 dòng)
> Status: ✅ ACTIVE — đã chạy đầu-cuối trên GPU 2026-08-30
> Last verified against source: 2026-08-30

# `scripts/explain_stage2.py`

## Purpose

Sản phẩm của lớp XAI Stage 2: mỗi study một dòng JSONL, mỗi study một `.npz`
bản đồ quy kết, cộng một `summary.json` cho cả lần chạy.

## ⚠ Cổng triệt tiêu chạy TRƯỚC và có quyền huỷ cả lần chạy

Nếu thay ảnh của study khác vào mà báo cáo **không** khó dự đoán hơn một cách
có ý nghĩa thống kê, thì mọi bản đồ sinh ra sau đó đều vô nghĩa — hoặc span thị
giác định vị sai, hoặc model không dùng ảnh. Lệnh **dừng** thay vì đổ đầy một
thư mục artifact trông có vẻ hợp lý.

`--skip-ablation-gate` có tồn tại, cảnh báo to, và được ghi vào `summary.json`
dưới khoá `ablation_gate_skipped`. `--no-gradient-weight` cũng vậy. Cả hai mặc
định ở thiết lập nghiêm.

Đo thật 2026-08-30, 8 study test: `mismatched_image` mean **+0.1429**,
95% CI **[+0.0284, +0.2668]**, 88% study tệ đi, `established=True`.

## Bảo mật — đi theo `evaluate_explanation.py`, không viết lại

| | |
|---|---|
| Đích ghi | Dùng lại `_assert_private_output_location`: từ chối path trong repo mà `git check-ignore` không xác nhận |
| Tên file bản đồ | **Tuần tự** — `study_00007.npz`. Không bao giờ chứa định danh. Có test đọc source chặn `{study.dicom_id}` và các biến thể |
| JSONL | Mang `sample_key` (blake2, 24 hex) thay cho id thật |
| Phép nối ngược | Chỉ ghi khi có `--write-key-map`, và đó là artifact nhạy cảm nhất lệnh này tạo ra |
| Độ phân giải bản đồ | **Lưới gốc 16×16**. Có test khẳng định source không chứa `imsave`/`savefig`/`.png`/`Image.fromarray` — ảnh upsample là ảnh bệnh nhân |

Kiểm chứng trên đầu ra thật: không có `keymap*`, không có khoá `dicom_id`,
không có path dạng `/p1X/pXXXXXXXX/` trong JSONL.

## Cấu trúc bản ghi

```
record  : schema_version, sample_key, split, attribution_map, attribution_grid,
          visual_span, rollout_method, labeler, parse_coverage,
          unparsed_sentences, sentences[]
sentence: index, text, char_start, char_end, token_indices, labels,
          spatially_meaningful, mean_token_nll, attribution_index,
          gradient_weighted
npz     : maps [n_sentences, 16, 16] float32 (mỗi bản đồ tổng = 1.0), grid [2]
```

## Lần chạy đầy đủ trên val — n=1513

2026-08-30, **23.1 phút**, 0 lỗi, 1.513/1.513 study frontal val có findings hợp
lệ, **7.786 câu**, 15 MB đầu ra. Peak **11.31 GiB / 15.48**; mọi study dùng
`shared`, fallback per-sentence không lần nào phải kích hoạt.

| | n | kết quả |
|---|---:|---|
| ablation, ảnh study khác | **100** | **+0.1788 [+0.1400, +0.2185]**, 83% tệ đi, established |
| randomization, rho cuối | 1 | **-0.0030**, degrades |
| `parse_coverage` (gộp theo câu) | **7.786** | **0.483** |
| `spatially_meaningful` | 7.786 | 3.758 = 48.3% |
| câu/study | 1.513 | min 1, median 5, max 14 |
| `mean_token_nll` | 7.777 | median 2.451, p5 0.547, p95 8.624 |

⚠ **154/1.513 study có coverage bằng 0** — không câu nào được gắn nhãn.

⚠ Các số smoke n=6 (coverage 0.606, ablation +0.1868) **không phải kết quả**.

## Bộ chẩn đoán đi kèm — `diagnose_parse_coverage.py`

⚠ **Bộ phân loại của nó KHÔNG được runner dùng.** `explain_stage2.py` lấy nhãn
từ `LexiconSentenceLabeler`; `classify()` chỉ phân loại các câu *đã* không gắn
được nhãn. Sửa từ khoá phân loại **không** làm đổi `parse_coverage`, bản đồ hay
gate — nên đừng chạy lại 23 phút GPU sau khi sửa nó.

Toàn val, n=1.513 study / 7.786 câu / 4.028 câu unparsed, chạy trong **1 giây**:

| nhóm | % của unparsed |
|---|---:|
| `normal` | **41.0%** |
| `technical` | **26.9%** |
| `unclassified` | 15.8% |
| `outside_14` | 10.8% |
| `missed_14` | **5.4%** |

Bộ chẩn đoán ra **đúng** cohort và **đúng** coverage 0.4827 như GPU runner, qua
một đường code độc lập — một phép kiểm chéo cho cả hai.

⚠ `missed_14` nằm giữa **5.4% (chặt) và 10.4% (nới)**: 202/636 câu
`unclassified` chứa từ chỉ một trong 14 nhãn mà lexicon thiếu synonym. Tức
217–419 câu trên 7.786 = **2,8–5,4% tổng số câu**. Đó là con số để quyết định
có đầu tư labeler tốt hơn hay không.

## ⚠ `parse_coverage` phải đi kèm mọi kết luận mức câu

Ghi ở **cả hai cấp** và in ra khi kết thúc. Đo thật trên 6 study / 33 câu:
gộp theo câu **0.606**, trung bình theo study **0.674** — hai số khác nhau đúng
như thiết kế, và số gộp mới là số nên trích. Một study trong lô đó có coverage
**0.125** (1 trong 8 câu có nhãn); câu không nhãn vẫn có bản đồ nhưng mang
`spatially_meaningful: false`.

## Cách chạy

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
python scripts/explain_stage2.py \
  --manifest <.../processed/full_allviews_v2/test.csv> \
  --image-root <thư mục chứa trực tiếp files/> \
  --output-dir ~/xai_out --split test --limit 6 --ablation-studies 8 --verbose
```

Peak VRAM đo được **10.7 GiB / 15.5** — cao hơn 9.9 của một forward đơn vì
`retain_graph` được giữ qua các câu trong cùng một study.

## Calls / Called by

- Calls: `training.explainability.{attention_capture, projection, rollout,
  sentence_attribution}`; `scripts.evaluate_explanation` (chỉ lấy chốt chặn
  bảo mật); `pandas`/`numpy`/`torch` import trễ trong hàm.
- Called by: người dùng. Không module nào import nó.

## Related tests

`tests/explainability/test_explain_runner.py` — 14 test CPU. Chúng kiểm phần
phải đúng **trước khi** cần tới GPU: chốt chặn bảo mật, khuôn tên file, ngữ
nghĩa tham số, và các trường bắt buộc trong bản ghi.
