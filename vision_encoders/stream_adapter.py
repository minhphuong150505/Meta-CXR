"""Trainable capacity on top of frozen encoder features.

Both classes here exist because ``MultiPositiveContrastiveLoss`` had **no
trainable parameter upstream of it** and was therefore a constant: its value sat
at 3.9929 / 3.9949 / 3.9941 / 3.9943 across four epochs of a real run while
carrying 0.1 of the loss weight. It was computed on tensors stashed before the
fusion module, and those tensors are raw outputs of frozen encoders, with the
auxiliary branch additionally under ``torch.no_grad()``.

``StreamAdapter`` fixes that by putting a small residual block between the frozen
encoder and everything downstream. ``ContrastiveProjectionHead`` is the standard
SimCLR-style ``g(.)``: the contrastive objective gets its own space instead of
competing for the representation MHCAC reads.
"""

import torch.nn as nn
import torch.nn.functional as F


class StreamAdapter(nn.Module):
    """Residual bottleneck applied per encoder stream, on the main path.

    Sits immediately after the frozen encoder output and **before** both the
    pre-fusion stash and ``ViewFusionModule``, so the contrastive loss finally
    has something to train while the fused path sees the same tensors.

    The last Linear is zero-initialised, which makes the block an exact identity
    at step 0 — the same trick ``ViewFusionBlock`` uses, and for the same reason:
    turning this on must not perturb a freshly built model before it has learned
    anything.

    ⚠ It adds parameters to the **inference** path. A checkpoint trained without
    it cannot be resumed into a model that has it.
    """

    def __init__(self, dim, bottleneck_ratio=4):
        super().__init__()
        dim = int(dim)
        if dim <= 0:
            raise ValueError("dim must be positive")
        hidden = max(1, dim // int(bottleneck_ratio))
        self.down = nn.Linear(dim, hidden)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens):
        """``[..., N, D] -> [..., N, D]``, residual."""
        return tokens + self.up(self.activation(self.down(tokens)))


class ContrastiveProjectionHead(nn.Module):
    """SimCLR-style ``g(.)``: pooled stream vector -> L2-normalised 256-d.

    One head per encoder, shared between the anchor and every auxiliary view.
    Kept separate from the representation MHCAC consumes so the contrastive
    objective cannot drag the classification features around directly; it is
    training-only and has no role at inference.
    """

    def __init__(self, dim, hidden_dim=512, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, pooled):
        """``[..., D] -> [..., output_dim]``, unit norm."""
        return F.normalize(self.net(pooled), dim=-1)


def pool_stream(tokens, has_cls_token):
    """Reduce one encoder's token sequence to a single vector.

    PubMedCLIP carries a real global token, so its CLS is the right summary;
    BioViL has none and its own global output *is* the patch mean, so a mean is
    the right summary there. Pooling the concatenated 246 tokens instead would
    weight BioViL 196/246 against PubMedCLIP 50/246 purely by token count, and
    would mix two different feature spaces.

    tokens: ``[..., N, D]``  ->  ``[..., D]``
    """
    if has_cls_token:
        return tokens[..., 0, :]
    return tokens.mean(dim=-2)
