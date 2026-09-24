"""Per-encoder random token masking (META-CXR paper, Implementation section).

"As a regularization technique, we randomly mask 10% of the features from each
encoder." Applied here to each encoder's own token sequence, independently per
sample, during training only: ``round(ratio * P)`` of the ``P`` tokens of every
image are zeroed.

It replaces ``Blip2Qformer._create_mask``, which was never called, and which
would have masked the same positions for the whole batch and over the
concatenated sequence rather than per encoder.

Torch only, so it is testable on a CPU box.
"""

from __future__ import annotations

import torch


def num_masked(num_tokens: int, ratio: float) -> int:
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"feature mask ratio must be in [0, 1), got {ratio}")
    return int(round(ratio * num_tokens))


def mask_encoder_tokens(
    tokens: torch.Tensor, ratio: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Zero ``round(ratio * P)`` random tokens of each sample of ``[B, P, D]``.

    Positions are drawn independently per sample. Returns ``tokens`` itself when
    nothing is masked, so the default path allocates nothing.
    """
    if tokens.ndim != 3:
        raise ValueError(f"expected [B, P, D] tokens, got {tuple(tokens.shape)}")
    batch, count, _ = tokens.shape
    k = num_masked(count, ratio)
    if k == 0 or batch == 0:
        return tokens
    scores = torch.rand(batch, count, device=tokens.device, generator=generator)
    drop = scores.argsort(dim=1)[:, :k]
    keep = torch.ones(batch, count, dtype=tokens.dtype, device=tokens.device)
    keep.scatter_(1, drop, 0)
    return tokens * keep.unsqueeze(-1)


def mask_aux_tokens(
    tokens: torch.Tensor, ratio: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Same rule for auxiliary views shaped ``[B, N, P, D]``."""
    if tokens.ndim != 4:
        raise ValueError(f"expected [B, N, P, D] tokens, got {tuple(tokens.shape)}")
    b, n, p, d = tokens.shape
    if b * n == 0:
        return tokens
    return mask_encoder_tokens(tokens.reshape(b * n, p, d), ratio, generator).reshape(b, n, p, d)
