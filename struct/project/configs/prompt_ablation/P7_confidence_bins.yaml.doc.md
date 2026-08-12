> Source: `configs/prompt_ablation/P7_confidence_bins.yaml` (25 dòng)
> Status: 🧪 ABLATION — requires calibrated probabilities
> Last verified against source: 2026-08-12

# `P7_confidence_bins.yaml`

Biến thể P5 với `uncertainty_policy: probability_bins`. Nó yêu cầu
`uncertain_probabilities` đã calibrate trên validation; builder raise nếu thiếu.
Không chạy bằng probability không có provenance hoặc lấy từ test split.

← [`prompt_ablation/`](_index.md) · [HOME](../../../HOME.md)
