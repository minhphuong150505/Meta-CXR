"""Late multi-view fusion: one MHCAC pass per view, logits pooled over views.

Opt-in since 2026-10-26 (``model.view_fusion.mode: late``; absent = ``early`` =
every recorded run). Plan and reading rule:
``docs/handoff/PLAN-2026-10-26-late-view-fusion.md``.

Why: early fusion (``view_fusion.py``) rewrites the anchor's tokens before MHCAC,
and every study -- single-view ones included -- also passes the StreamAdapters
and the MPC objective. Measured: fusion ON is weaker than OFF on single-view
studies. In late mode the anchor path is OFF's path (no adapter, no fusion block,
no MPC); MHCAC classifies the anchor and each real auxiliary view with the same
weights, and :class:`LateViewPooling` pools the per-view logits per finding.

Guaranteed by construction: a study with one real view gets exactly its anchor
logits (softmax over one element is 1). Not guaranteed: MHCAC is also trained on
the auxiliary (mostly lateral) views, so single-view accuracy can still move.

Depends on torch only, so every invariant is testable on a CPU box
(``tests/test_late_view_fusion.py``).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

VIEW_FUSION_MODES = ("early", "late")

# Additive mask value for padded view slots. Finite, as in view_fusion.MASK_NEG,
# so the softmax stays NaN-free; the anchor slot is always real.
MASK_NEG = -1e4


def resolve_view_fusion_mode(
    mode,
    *,
    multi_view: bool,
    lambda_mpc: float = 0.0,
    lambda_view_consistency: float = 0.0,
    lambda_explanation: float = 0.0,
    lambda_explanation_strong: float = 0.0,
) -> str:
    """``model.view_fusion.mode`` -> ``"early"`` | ``"late"``; refuses what late cannot serve.

    Absent (``None``) is ``"early"``, the historical path. ``"late"`` needs
    ``multi_view: true`` (that is what makes the dataset load the auxiliary view,
    so without it a "late" run would silently train as OFF), and it refuses the
    terms that read early fusion's pre-fusion streams (MPC, view consistency)
    and the explanation loss, which Grad-CAMs a single token sequence.
    """
    resolved = "early" if mode is None else str(mode)
    if resolved not in VIEW_FUSION_MODES:
        raise ValueError(
            f"model.view_fusion.mode must be one of {VIEW_FUSION_MODES}; got {mode!r}"
        )
    if resolved != "late":
        return resolved
    if not multi_view:
        raise ValueError(
            "model.view_fusion.mode: late needs model.multi_view: true in the same "
            "phase (it is what loads the auxiliary view); without it the run would "
            "train as fusion OFF"
        )
    refused = {
        "model.loss.lambda_mpc": lambda_mpc,
        "model.loss.lambda_view_consistency": lambda_view_consistency,
        "model.loss.lambda_explanation": lambda_explanation,
        "model.loss.lambda_explanation_strong": lambda_explanation_strong,
    }
    live = [key for key, value in refused.items() if float(value) > 0]
    if live:
        raise ValueError(
            f"model.view_fusion.mode: late does not support {live}; set them to 0.0 "
            "(MPC and view consistency read early fusion's pre-fusion streams, the "
            "explanation loss Grad-CAMs one token sequence)"
        )
    return resolved


class LateViewPooling(nn.Module):
    """Per-finding attention over a study's real views, applied to MHCAC logits.

    For view ``v`` and finding ``f`` with logits ``z[v, f]`` (3 classes)::

        s[v, f] = w_f . [log_softmax(z[v, f]), onehot(view_type[v])] + b_f
        a[:, f] = softmax over the REAL views of s[:, f]
        out[f]  = sum_v a[v, f] * z[v, f]

    ``w`` and ``b`` are zero-initialised, so at step 0 the output is the mean of
    the real views' logits. Padded slots are masked out of the softmax and their
    logits are never read (``torch.where``), so junk there cannot leak in. A
    study whose only real view is the anchor returns the anchor logits exactly.
    ``num_findings * (num_classes + num_view_types + 1)`` parameters (112 here).
    """

    def __init__(self, num_findings: int = 14, num_classes: int = 3, num_view_types: int = 4):
        super().__init__()
        self.num_findings = int(num_findings)
        self.num_classes = int(num_classes)
        self.num_view_types = int(num_view_types)
        self.weight = nn.Parameter(
            torch.zeros(self.num_findings, self.num_classes + self.num_view_types)
        )
        self.bias = nn.Parameter(torch.zeros(self.num_findings))

    def weights(self, anchor_logits, aux_logits, aux_mask, anchor_view_id=None, aux_view_ids=None):
        """Pooling weights ``[B, 1 + N, F]``, anchor first; zero on padded slots."""
        logits, valid, view_ids = self._stack(
            anchor_logits, aux_logits, aux_mask, anchor_view_id, aux_view_ids
        )
        log_probs = F.log_softmax(logits.float(), dim=-1)                 # [B, V, F, C]
        onehot = F.one_hot(view_ids, self.num_view_types).to(log_probs.dtype)
        onehot = onehot[:, :, None, :].expand(-1, -1, self.num_findings, -1)
        features = torch.cat([log_probs, onehot], dim=-1)                  # [B, V, F, C+T]
        scores = (features * self.weight.float()).sum(-1) + self.bias.float()
        scores = scores.masked_fill(~valid[:, :, None], MASK_NEG)
        return torch.softmax(scores, dim=1) * valid[:, :, None].to(scores.dtype)

    def forward(self, anchor_logits, aux_logits, aux_mask, anchor_view_id=None, aux_view_ids=None):
        """anchor_logits [B, F, C], aux_logits [B, N, F, C], aux_mask [B, N] -> [B, F, C]."""
        if aux_logits is None or aux_logits.shape[1] == 0 or aux_mask is None:
            return anchor_logits
        logits, valid, _ = self._stack(
            anchor_logits, aux_logits, aux_mask, anchor_view_id, aux_view_ids
        )
        weights = self.weights(anchor_logits, aux_logits, aux_mask, anchor_view_id, aux_view_ids)
        safe = torch.where(valid[:, :, None, None], logits, torch.zeros_like(logits))
        pooled = (weights[..., None].to(safe.dtype) * safe).sum(dim=1)
        # A study with no real auxiliary view: its weight on the anchor is exactly
        # 1, but 1.0 * z is not bit-identical to z under every dtype, so return z.
        has_aux = valid[:, 1:].any(dim=1)
        return torch.where(has_aux[:, None, None], pooled, anchor_logits)

    def _stack(self, anchor_logits, aux_logits, aux_mask, anchor_view_id, aux_view_ids):
        if anchor_logits.ndim != 3 or anchor_logits.shape[1:] != (self.num_findings, self.num_classes):
            raise ValueError(
                f"anchor_logits must be [B, {self.num_findings}, {self.num_classes}]; "
                f"got {tuple(anchor_logits.shape)}"
            )
        batch, views = int(aux_logits.shape[0]), int(aux_logits.shape[1])
        if tuple(aux_logits.shape) != (batch, views, self.num_findings, self.num_classes) \
                or batch != anchor_logits.shape[0]:
            raise ValueError(
                f"aux_logits must be [B, N, {self.num_findings}, {self.num_classes}] with "
                f"B={anchor_logits.shape[0]}; got {tuple(aux_logits.shape)}"
            )
        device = anchor_logits.device
        aux_mask = aux_mask.to(device=device, dtype=torch.bool)
        if tuple(aux_mask.shape) != (batch, views):
            raise ValueError(f"aux_mask must be [{batch}, {views}]; got {tuple(aux_mask.shape)}")
        logits = torch.cat([anchor_logits[:, None], aux_logits.to(anchor_logits.dtype)], dim=1)
        valid = torch.cat([torch.ones(batch, 1, dtype=torch.bool, device=device), aux_mask], dim=1)
        if anchor_view_id is None:
            anchor_view_id = torch.zeros(batch, dtype=torch.long, device=device)
        if aux_view_ids is None:
            aux_view_ids = torch.zeros(batch, views, dtype=torch.long, device=device)
        view_ids = torch.cat(
            [
                anchor_view_id.to(device=device, dtype=torch.long).reshape(batch, 1),
                aux_view_ids.to(device=device, dtype=torch.long).reshape(batch, views),
            ],
            dim=1,
        )
        # Padded slots may carry any id (the collater fills them); clamp so
        # one_hot never sees an out-of-range value on a slot that is masked anyway.
        view_ids = torch.where(valid, view_ids, torch.zeros_like(view_ids))
        if bool(((view_ids < 0) | (view_ids >= self.num_view_types)).any()):
            raise ValueError(f"view type ids must be in 0..{self.num_view_types - 1}")
        return logits, valid, view_ids


def late_classification_loss(cls_loss_fn, pooled_logits, anchor_logits, labels,
                             sample_mask=None, anchor_weight: float = 0.5):
    """``(1 - w) CE(pooled) + w CE(anchor)``, w = ``model.view_fusion.late_anchor_weight``.

    A single-view study has ``pooled == anchor``, so it gets exactly the OFF
    loss CE(anchor); the anchor path is supervised on every study. ``w`` is
    fixed by the plan (0.5), not tuned.
    """
    w = float(anchor_weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"late_anchor_weight must be in [0, 1]; got {w}")
    return (
        (1.0 - w) * cls_loss_fn(pooled_logits, labels, sample_mask=sample_mask)
        + w * cls_loss_fn(anchor_logits, labels, sample_mask=sample_mask)
    )
