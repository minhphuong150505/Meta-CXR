> Source: `model/lavis/data/ReportDataset.py:810-890`
> Status: 🟡 CONDITIONAL — legacy caption evaluator

# `MIMICEvalCap`

## Responsibility

Adapter caption evaluation giữ ground truth theo image id và gọi BLEU/ROUGE/
METEOR/CIDEr scorer kiểu COCO trong `evaluate(res)`.

## Lifecycle

`__init__(gts, img_id_map)` chuẩn hóa key; `preprocess` làm sạch text;
`evaluate(res)` dựng scorer list, compute score và trả dict.

## Status / risk

Stage-1 production classification hook hiện dùng `training/evaluation` và không
đi qua class này. Class vẫn nằm trong file LAVIS cho caption workflow/compatibility,
nên gắn CONDITIONAL thay vì unused. Metric implementation cần optional packages.

← [`ReportDataset.py`](../../ReportDataset.py.doc.md) · [HOME](../../../../../../HOME.md)
