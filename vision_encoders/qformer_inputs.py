"""What the META-Former (Q-Former) reads, and a penalty against its 32 queries collapsing.

Both pieces are opt-in (2026-10-15, ``docs/handoff/PLAN-2026-10-15-qformer-refit.md``) and
exist because the 32 soft tokens of the reported Stage-1 model are one vector
(within-study cosine 0.9989 after phase 1a, ``scripts/diagnose_qformer_collapse.py``):
in the first two cross-attention layers every query attends PubMedCLIP's CLS token.

* :class:`QformerInputNorm` -- one LayerNorm per encoder stream on the Q-Former's copy of
  the shared visual tokens. BLIP-2's Q-Former was pretrained on LayerNorm-ed ViT tokens
  (``ln_vision``); here only BioViL passes a LayerNorm, before its projection, so the
  PubMedCLIP and Swin tokens reach the cross-attention at whatever scale the StreamAdapter
  gives them (phase 1b raised it 19.6 -> 1,713 and 12 -> 413). MHCAC keeps reading the
  un-normalised tokens; this is the one sanctioned per-span operation on
  :class:`~vision_encoders.shared_visual_tokens.SharedVisualTokens` that only one branch
  sees, and it normalises, it does not re-project.
* :func:`query_diversity_loss` -- ``log(Q / PR) / log(Q)`` of the per-study participation
  ratio of the queries' IMAGE-DEPENDENT parts (each query's batch mean removed). 1 when the
  32 queries carry one direction of image information, 0 when all Q carry equal shares of
  independent directions. Removing the per-query batch mean is what stops a constant
  per-query offset from passing for diversity, and the participation ratio (not a mean of
  pairwise cosines) is what stops "one informative query + 31 near-constant ones" from
  passing: near-constant queries contribute ~no variance and so ~nothing to the ratio.

torch only, so both are testable on a CPU box.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

QFORMER_INPUT_NORM_MODES = ("per_stream",)
QFORMER_INPUT_NORM_PREFIX = "qformer_input_norm"


def resolve_qformer_input_norm(value) -> str | None:
    """``None`` (absent key, the historical path) or ``"per_stream"``."""
    if value is None:
        return None
    mode = str(value)
    if mode not in QFORMER_INPUT_NORM_MODES:
        raise ValueError(
            f"model.qformer_input_norm must be one of {QFORMER_INPUT_NORM_MODES} or absent; "
            f"got {value!r}"
        )
    return mode


class QformerInputNorm(nn.Module):
    """``LayerNorm(visual_dim)`` per stream, applied to the stream's span of the shared tokens."""

    def __init__(self, stream_names: Iterable[str], visual_dim: int):
        super().__init__()
        self.stream_names = tuple(stream_names)
        if not self.stream_names:
            raise ValueError("QformerInputNorm needs at least one stream")
        self.norms = nn.ModuleDict({name: nn.LayerNorm(int(visual_dim)) for name in self.stream_names})

    def forward(self, shared) -> torch.Tensor:
        if set(shared.spans) != set(self.stream_names):
            raise ValueError(
                f"shared tokens carry streams {sorted(shared.spans)}; this norm was built for "
                f"{list(self.stream_names)}"
            )
        parts, cursor = [], 0
        for name in self.stream_names:
            span = shared.spans[name]
            if span.start != cursor:
                raise ValueError(f"stream {name!r} span {span} is not contiguous with the previous one")
            parts.append(self.norms[name](shared.tokens[:, span]))
            cursor = span.stop
        if cursor != shared.tokens.shape[1]:
            raise ValueError(f"spans cover {cursor} of {shared.tokens.shape[1]} tokens")
        return torch.cat(parts, dim=1)


def batch_centred_participation_ratio(states: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-study participation ratio of ``states - states.mean(0)``; ``states`` is ``[B, Q, D]``.

    ``(sum s^2)^2 / sum s^4`` over the singular values ``s`` of each study's ``[Q, D]`` matrix of
    image-dependent parts, computed from its Gram matrix ``G``: ``sum s^2 = tr G`` and
    ``sum s^4 = ||G||_F^2``. In ``[1, min(Q, D)]`` (0 for an all-zero matrix); returns ``[B]``.
    """
    if states.ndim != 3:
        raise ValueError(f"states must be [B, Q, D]; got {tuple(states.shape)}")
    x = states.float()
    centred = x - x.mean(dim=0, keepdim=True)
    gram = centred @ centred.transpose(1, 2)
    trace = gram.diagonal(dim1=1, dim2=2).sum(-1)
    frob2 = gram.pow(2).sum(dim=(1, 2))
    return trace.pow(2) / frob2.clamp_min(eps)


def query_diversity_loss(states: torch.Tensor) -> torch.Tensor:
    """``mean_b log(Q / PR_b) / log(Q)``, in ``[0, 1]``; see the module docstring.

    A batch of one study has no image-dependent part to measure: the loss is then a zero that
    still carries the graph (no gradient, no error), which is what the last GradCache chunk of
    an odd-sized batch gets.
    """
    if states.ndim != 3:
        raise ValueError(f"states must be [B, Q, D]; got {tuple(states.shape)}")
    batch, queries = int(states.shape[0]), int(states.shape[1])
    if batch < 2 or queries < 2:
        return states.float().sum() * 0.0
    pr = batch_centred_participation_ratio(states).clamp(min=1.0, max=float(queries))
    return ((math.log(queries) - pr.log()) / math.log(queries)).mean()


def assert_input_norm_keys_match(model_keys: Iterable[str], checkpoint_keys: Iterable[str],
                                 where: str = "checkpoint") -> None:
    """Fail loudly when the model and a checkpoint disagree on ``qformer_input_norm``.

    Built without the norm, a refit checkpoint's norm tensors would be silently dropped (the
    Q-Former then reads un-normalised tokens it never saw in training); built with it, an old
    checkpoint would leave the norm at its LayerNorm initialisation. Both give fluent Stage-2
    reports from the wrong soft tokens, so neither may pass quietly.
    """
    prefix = QFORMER_INPUT_NORM_PREFIX + "."
    in_model = any(k.startswith(prefix) for k in model_keys)
    in_ckpt = any(k.startswith(prefix) for k in checkpoint_keys)
    if in_ckpt and not in_model:
        raise ValueError(
            f"{where} carries {QFORMER_INPUT_NORM_PREFIX}.* but the model was built without it; "
            "pass the refit's config option model.qformer_input_norm=per_stream"
        )
    if in_model and not in_ckpt:
        raise ValueError(
            f"the model has {QFORMER_INPUT_NORM_PREFIX} but {where} does not; it would run at its "
            "LayerNorm initialisation. Use the merged refit checkpoint or drop the option"
        )
