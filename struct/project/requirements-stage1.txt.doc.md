> Source: `requirements-stage1.txt` (35 dòng)
> Status: ✅ ACTIVE — Stage-1 lock file
> Last verified against source: 2026-08-12

# `requirements-stage1.txt`

## Purpose

Pin environment Stage 1/test core cho Python 3.10/3.11. Bao gồm torch stack,
LAVIS dependencies, preprocessing/evaluation packages và pytest.

## Important pins

`torch==2.5.1`, `torchvision==0.20.1`, `transformers==4.53.2`,
`numpy==1.26.4`, `pandas==2.2.3`, `timm==0.9.16`, `omegaconf==2.3.0`.

Comment source yêu cầu cài torch từ official CUDA index phù hợp trước, target đã
test là CUDA 12.4 trên RTX 5060 Ti 16 GB. File pin package nhưng không pin CUDA wheel index.

## Relationship

`requirements.txt` include file này. `requirements-stage2.txt` cũng include toàn
bộ file này rồi thêm QLoRA packages; vì Q-Former path vẫn cần Stage 1.

← [HOME](../HOME.md)
