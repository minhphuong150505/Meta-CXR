# `scripts/analyze_stage2_generations.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-11 · 🔬 chẩn đoán

## Purpose
Phân tích lỗi báo cáo Stage 2 so với tham chiếu, **chỉ số tổng hợp**: độ dài, số đầu
ra khác nhau, tỉ lệ đầu ra phổ biến nhất (mức "khuôn mẫu"), tỉ lệ ngôn ngữ so sánh
thời gian ("unchanged", "compared with prior"...), và **CE xấp xỉ bằng lexicon**
(`safety.claims.LexiconClaimParser` gán nhãn cả báo cáo sinh lẫn tham chiếu; P/R/F1
nhắc dương tính theo từng bệnh, nhắc đến bệnh bất kể chiều).

⚠ CE lexicon là thước đo **tương đối** để so các hệ trên cùng study, không phải CE
CheXbert; không báo như số CE tuyệt đối.

```bash
python scripts/analyze_stage2_generations.py --pred tên=<file.jsonl> [--pred ...] --report <private>/x.json
```
Mỗi dòng JSONL cần `sample_key`, `pred`, `ref`.
