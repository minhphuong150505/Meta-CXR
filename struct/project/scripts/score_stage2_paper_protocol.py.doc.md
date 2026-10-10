# `scripts/score_stage2_paper_protocol.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-11 · ✅ ACTIVE (chạy trên máy train)

## Purpose
Chấm báo cáo Stage 2 theo **giao thức bài báo** (bỏ xuống dòng, nltk word_tokenize,
chữ thường; pycocoevalcap BLEU-1..4 / METEOR / ROUGE-L / CIDEr; BERTScore F1
`distilroberta-base` có rescale) cộng **CE lexicon** (tương đối), và **so sánh ghép
cặp** với một file mốc (`--baseline`) trên các `sample_key` chung: BERTScore, ROUGE-L,
CIDEr, BLEU-4 theo study và CE lexicon macro F1, bootstrap 2.000 lần, seed 16. Đưa vào
git phần chấm trước đây chỉ có trên máy train (`~/paper_nlg_gpu.py`, `~/pair_cmp.py`).
Nhận dòng JSONL `pred`/`ref` hoặc `generated`/`reference`.
