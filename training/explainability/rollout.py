"""Gradient-weighted attention rollout, as pure tensor algebra.

DELIBERATELY DEPENDENCY-FREE BEYOND ``torch``. This module imports no model, no
``transformers``, and nothing else from this repository. Attention matrices go
in, an attribution vector comes out. That is what makes the arithmetic testable
against hand-computed answers on a CPU box, which is the only machine the
planning session has -- see ``tests/explainability/test_rollout.py``.

The formula implemented
=======================

``METHOD_CHEFER`` (default) is the self-attention rule of Chefer et al. 2021,
*Generic Attention-model Explainability for Interpreting Bi-Modal and
Encoder-Decoder Transformers*:

    Abar[l] = mean_h ( relu( grad(A[l]) (*) A[l] ) )     -- fuse_heads
    R[0]    = I
    R[l]    = R[l-1] + Abar[l] @ R[l-1]                  -- rollout

where ``(*)`` is elementwise. Note the order: the elementwise product is
clamped at zero and only THEN averaged over heads, which is what the reference
implementation does. Clamping after the mean lets a strongly negative head
cancel a positive one and gives visibly different maps.

``METHOD_ABNAR`` is the older gradient-free rollout of Abnar & Zuidema 2020:

    Ahat[l] = rownorm( 0.5 * A[l] + 0.5 * I )
    R       = Ahat[L] @ ... @ Ahat[1]

Both are offered because the gradient-weighted path may not survive a backward
pass through NF4-quantised weights. The documented fallback is to call
:func:`fuse_heads` with ``gradient=None``, which keeps the Chefer product but
drops the gradient term; ``METHOD_ABNAR`` is the more conservative second
fallback. Neither fallback is silent: :func:`rollout` records what it did in
the returned :class:`RolloutTrace`.

Orientation convention
======================

``R[i, j]`` is the attributed influence of key position ``j`` on query position
``i``. Attention rows are queries. In a causal decoder ``A`` is lower
triangular, which rollout handles without special-casing: a soft token that
precedes every generated token is reachable from all of them.

.. warning::
   This module was written from the repository's own reading of Chefer et al.
   The project's design document (Section 4) was not available to the session
   that wrote it. Check these three equations against that section before
   quoting any number produced by them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

METHOD_CHEFER = "chefer"
METHOD_ABNAR = "abnar"
METHODS = (METHOD_CHEFER, METHOD_ABNAR)

REDUCE_MEAN = "mean"
REDUCE_SUM = "sum"
REDUCE_MAX = "max"
REDUCTIONS = (REDUCE_MEAN, REDUCE_SUM, REDUCE_MAX)


@dataclass(frozen=True)
class RolloutTrace:
    """What the rollout actually did, so a fallback can never be silent."""

    method: str
    num_layers: int
    sequence_length: int
    gradient_weighted: bool
    row_normalized: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "num_layers": self.num_layers,
            "sequence_length": self.sequence_length,
            "gradient_weighted": self.gradient_weighted,
            "row_normalized": self.row_normalized,
        }


def _validate_attention(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [L, H, S, S], got {tuple(tensor.shape)}")
    if tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError(f"{name} must be square in its last two dims, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 0 or tensor.shape[1] == 0 or tensor.shape[-1] == 0:
        raise ValueError(f"{name} must be non-empty in every dimension, got {tuple(tensor.shape)}")
    value = tensor.detach().to(torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def stack_layers(
    per_layer: Sequence[torch.Tensor],
    batch_index: int = 0,
) -> torch.Tensor:
    """Turn a per-layer tuple of ``[B, H, S, S]`` into one ``[L, H, S, S]``.

    ``output_attentions=True`` returns exactly that tuple, so this is the seam
    between a live model and the rest of this module. It stays here, rather
    than in ``attention_capture``, because it is pure tensor work and therefore
    belongs on the testable side of the line.
    """
    if not isinstance(per_layer, Sequence) or isinstance(per_layer, (str, bytes)):
        raise TypeError("per_layer must be a sequence of tensors")
    if len(per_layer) == 0:
        raise ValueError("per_layer must contain at least one layer")

    selected = []
    for layer_index, layer in enumerate(per_layer):
        if not torch.is_tensor(layer):
            raise TypeError(f"layer {layer_index} is not a tensor")
        if layer.ndim != 4:
            raise ValueError(
                f"layer {layer_index} must have shape [B, H, S, S], got {tuple(layer.shape)}"
            )
        if not 0 <= batch_index < layer.shape[0]:
            raise IndexError(
                f"batch_index {batch_index} outside [0, {layer.shape[0]}) at layer {layer_index}"
            )
        selected.append(layer[batch_index])

    shapes = {tuple(item.shape) for item in selected}
    if len(shapes) != 1:
        raise ValueError(f"layers disagree on shape: {sorted(shapes)}")
    return torch.stack(selected, dim=0)


def fuse_heads(
    attention: torch.Tensor,
    gradient: torch.Tensor | None = None,
    *,
    row_normalize: bool = False,
) -> torch.Tensor:
    """Collapse ``[L, H, S, S]`` to ``[L, S, S]``: weight, clamp, then average.

    ``gradient=None`` is the documented gradient-free fallback. Attention
    probabilities are already non-negative, so the clamp is then a no-op and
    this reduces to a plain head average -- which is the honest description of
    what that fallback computes, and why it is weaker.

    ``row_normalize`` rescales each query row to sum to one after clamping.
    Off by default: it changes the relative weight of layers in the Chefer
    product, so a map computed with it is not comparable to one computed
    without it.
    """
    value = _validate_attention(attention, "attention")
    weighted = value
    if gradient is not None:
        grad = _validate_attention(gradient, "gradient")
        if grad.shape != value.shape:
            raise ValueError(
                "gradient and attention must have identical shapes: "
                f"{tuple(grad.shape)} vs {tuple(value.shape)}"
            )
        weighted = value * grad

    # Clamp BEFORE the head average, matching the reference implementation.
    # Averaging first would let a negative head cancel a positive one.
    fused = weighted.clamp(min=0.0).mean(dim=1)

    if row_normalize:
        totals = fused.sum(dim=-1, keepdim=True)
        fused = torch.where(totals > 0, fused / totals, torch.zeros_like(fused))
    return fused


def _rollout_chefer(fused: torch.Tensor) -> torch.Tensor:
    length = fused.shape[-1]
    result = torch.eye(length, dtype=fused.dtype, device=fused.device)
    for layer in fused:
        result = result + layer @ result
    return result


def _rollout_abnar(fused: torch.Tensor) -> torch.Tensor:
    length = fused.shape[-1]
    identity = torch.eye(length, dtype=fused.dtype, device=fused.device)
    result = None
    for layer in fused:
        residual = 0.5 * layer + 0.5 * identity
        totals = residual.sum(dim=-1, keepdim=True)
        residual = torch.where(totals > 0, residual / totals, torch.zeros_like(residual))
        result = residual if result is None else residual @ result
    return result


def rollout(
    fused: torch.Tensor,
    *,
    method: str = METHOD_CHEFER,
    subtract_identity: bool = False,
    gradient_weighted: bool | None = None,
    row_normalized: bool = False,
) -> tuple[torch.Tensor, RolloutTrace]:
    """Propagate per-layer fused attention into one ``[S, S]`` matrix.

    ``subtract_identity`` removes each position's trivial contribution to
    itself. It defaults to off because it is IRRELEVANT to this project's
    query: the generated tokens and the soft-token span are disjoint, so the
    identity contributes exactly zero to ``R[generated, soft]``. Turn it on
    only when reading the diagonal block.

    ``gradient_weighted`` and ``row_normalized`` are recorded, not enforced --
    :func:`fuse_heads` already consumed the gradient by the time the fused
    tensor arrives here. Pass what you actually did so the returned trace
    cannot claim a gradient-weighted map when the fallback ran.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if not torch.is_tensor(fused):
        raise TypeError("fused must be a torch.Tensor")
    if fused.ndim != 3:
        raise ValueError(f"fused must have shape [L, S, S], got {tuple(fused.shape)}")
    if fused.shape[-1] != fused.shape[-2]:
        raise ValueError(f"fused must be square in its last two dims, got {tuple(fused.shape)}")
    if fused.shape[0] == 0 or fused.shape[-1] == 0:
        raise ValueError(f"fused must be non-empty, got {tuple(fused.shape)}")
    if not torch.isfinite(fused).all():
        raise ValueError("fused contains non-finite values")

    value = fused.to(torch.float32)
    result = _rollout_chefer(value) if method == METHOD_CHEFER else _rollout_abnar(value)

    if subtract_identity:
        length = result.shape[-1]
        result = result - torch.eye(length, dtype=result.dtype, device=result.device)

    trace = RolloutTrace(
        method=method,
        num_layers=int(fused.shape[0]),
        sequence_length=int(fused.shape[-1]),
        # ``None`` means the caller did not say; record it as unweighted rather
        # than guessing generously.
        gradient_weighted=bool(gradient_weighted),
        row_normalized=bool(row_normalized),
    )
    return result, trace


def _validate_positions(
    positions: Sequence[int],
    length: int,
    name: str,
) -> torch.Tensor:
    if isinstance(positions, torch.Tensor):
        index = positions.detach().reshape(-1).to(torch.long)
    else:
        if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
            raise TypeError(f"{name} must be a sequence of integers or a tensor")
        index = torch.tensor([int(item) for item in positions], dtype=torch.long)
    if index.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if int(index.min()) < 0 or int(index.max()) >= length:
        raise IndexError(
            f"{name} contains a position outside [0, {length}): "
            f"min={int(index.min())} max={int(index.max())}"
        )
    if index.unique().numel() != index.numel():
        raise ValueError(f"{name} contains duplicate positions")
    return index


def span_attribution(
    rollout_matrix: torch.Tensor,
    query_positions: Sequence[int],
    key_positions: Sequence[int],
    *,
    reduce: str = REDUCE_MEAN,
    normalize: bool = True,
) -> torch.Tensor:
    """Attribution of one key span, read from the queries that consume it.

    Returns one value per entry of ``key_positions``, in the order given.

    ``normalize`` divides by the total so the vector sums to one, which is what
    makes two sentences of different lengths comparable. A span that receives
    no attributed mass returns all zeros rather than NaN -- that is a real
    outcome (the sentence did not use the image) and must not be laundered into
    a uniform distribution.
    """
    if reduce not in REDUCTIONS:
        raise ValueError(f"reduce must be one of {REDUCTIONS}, got {reduce!r}")
    if not torch.is_tensor(rollout_matrix):
        raise TypeError("rollout_matrix must be a torch.Tensor")
    if rollout_matrix.ndim != 2 or rollout_matrix.shape[0] != rollout_matrix.shape[1]:
        raise ValueError(
            f"rollout_matrix must be square [S, S], got {tuple(rollout_matrix.shape)}"
        )

    length = int(rollout_matrix.shape[0])
    queries = _validate_positions(query_positions, length, "query_positions")
    keys = _validate_positions(key_positions, length, "key_positions")

    block = rollout_matrix.to(torch.float32)[queries][:, keys]
    if reduce == REDUCE_MEAN:
        values = block.mean(dim=0)
    elif reduce == REDUCE_SUM:
        values = block.sum(dim=0)
    else:
        values = block.max(dim=0).values

    # Rollout entries are non-negative by construction (clamped, and both
    # recurrences are sums/products of non-negatives), but a caller that passed
    # ``subtract_identity=True`` over an overlapping span could break that.
    values = values.clamp(min=0.0)

    if normalize:
        total = values.sum()
        values = values / total if float(total) > 0.0 else torch.zeros_like(values)
    return values
