import torch
import torch.nn as nn
import torch.nn.functional as F


def logit_difference_squared(logits, labels, sample_mask=None):
    """Return the logit-difference score and disease-positive sample mask."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [B,A,C] and labels [B,A]")
    if logits.shape[:2] != labels.shape or logits.shape[-1] < 2:
        raise ValueError(
            f"logit/label shape mismatch: {tuple(logits.shape)} vs "
            f"{tuple(labels.shape)}"
        )

    labels = labels.to(device=logits.device)
    positive = labels == 1
    if sample_mask is not None:
        sample_mask = torch.as_tensor(
            sample_mask, dtype=torch.bool, device=logits.device
        ).reshape(-1)
        if sample_mask.numel() != logits.shape[0]:
            raise ValueError("sample_mask must contain one value per batch item")
        positive = positive & sample_mask[:, None]

    logits_fp32 = logits.float()
    differences = logits_fp32[..., 1] - logits_fp32[..., 0]
    score = (differences.square() * positive.to(differences.dtype)).sum(dim=1)
    return score, positive.any(dim=1)


def single_label_score(logits, label_index, sample_selector):
    """(logit_pos - logit_neg)^2 for ONE abnormality, summed over selected rows.

    The pooled :func:`logit_difference_squared` sums every positive finding into
    a single scalar, so its Grad-CAM answers "where is the evidence for anything
    this study has?". That is the right question for a lung prior and the wrong
    one for an expert box, which is drawn around one named finding. Taking the
    gradient of this score instead yields a CAM for that finding alone.

    Only the rows in ``sample_selector`` contribute, so one backward pass covers
    every study in the batch that carries a box for ``label_index``.
    """
    if logits.ndim != 3 or logits.shape[-1] < 2:
        raise ValueError("expected logits [B,A,C] with at least 2 classes")
    label_index = int(label_index)
    if not 0 <= label_index < logits.shape[1]:
        raise ValueError(
            f"label_index {label_index} outside [0, {logits.shape[1]})"
        )
    selector = torch.as_tensor(
        sample_selector, dtype=torch.bool, device=logits.device
    ).reshape(-1)
    if selector.numel() != logits.shape[0]:
        raise ValueError("sample_selector must contain one value per batch item")

    logits_fp32 = logits.float()
    difference = (
        logits_fp32[:, label_index, 1] - logits_fp32[:, label_index, 0]
    )
    return (difference.square() * selector.to(difference.dtype)).sum()


def _cam_from_gradients(activations, gradients, grid_hw):
    height, width = (int(grid_hw[0]), int(grid_hw[1]))
    if height <= 0 or width <= 0:
        raise ValueError("grid_hw must contain two positive integers")
    if activations.ndim != 3:
        raise ValueError("activations must have shape [B,N,C]")
    if activations.shape != gradients.shape:
        raise ValueError("activations and gradients must have identical shapes")
    if activations.shape[1] != height * width:
        raise ValueError(
            f"activation token count {activations.shape[1]} does not match "
            f"grid {height}x{width}"
        )

    activations_fp32 = activations.float()
    gradients_fp32 = gradients.float()
    channel_weights = gradients_fp32.mean(dim=1, keepdim=True)
    cam = F.relu((channel_weights * activations_fp32).sum(dim=-1))
    return cam.reshape(activations.shape[0], height, width)


def grad_cam(score, activations, grid_hw, create_graph=True):
    """Compute a Grad-CAM map for a per-sample score."""
    if score.numel() != activations.shape[0]:
        raise ValueError("score must contain one value per batch item")
    gradients = torch.autograd.grad(
        outputs=score.sum(),
        inputs=activations,
        create_graph=create_graph,
    )[0]
    return _cam_from_gradients(activations, gradients, grid_hw)


def explanation_loss(cam, mask, top_k=0.5, eps=1e-6):
    """Measure the fraction of top-k soft CAM mass outside the target mask."""
    if cam.ndim != 3 or mask.shape != cam.shape:
        raise ValueError("cam and mask must have identical [B,H,W] shapes")
    if not 0.0 < top_k <= 1.0:
        raise ValueError("top_k must be in (0, 1]")
    if eps <= 0:
        raise ValueError("eps must be positive")

    flat = cam.float().flatten(1)
    minimum = flat.amin(dim=1, keepdim=True)
    maximum = flat.amax(dim=1, keepdim=True)
    normalized = (flat - minimum) / (maximum - minimum + eps)
    threshold = torch.quantile(normalized, 1.0 - top_k, dim=-1).detach()
    positive = normalized * (normalized >= threshold[:, None]).to(normalized.dtype)

    mask_flat = mask.to(device=cam.device, dtype=positive.dtype).flatten(1)
    inside_mass = (positive * mask_flat).sum(dim=1)
    total_mass = positive.sum(dim=1)
    return 1.0 - inside_mass / (total_mass + eps)


def resize_mask_to_grid(mask, grid_hw):
    """Pool a high-resolution binary mask onto a CAM grid."""
    if mask.ndim != 3:
        raise ValueError("mask must have shape [B,H,W]")
    height, width = (int(grid_hw[0]), int(grid_hw[1]))
    if height <= 0 or width <= 0:
        raise ValueError("grid_hw must contain two positive integers")
    pooled = F.adaptive_avg_pool2d(mask.float().unsqueeze(1), (height, width))
    return (pooled.squeeze(1) > 0).float()


class ExplanationLoss(nn.Module):
    """Two explanation terms with different evidentiary status.

    ``weak`` — one pooled CAM per study against a CheXmask **lung** mask. The
    mask is anatomical and identical for all fourteen findings, so the only
    claim it can support is "the model looks inside the lungs". Broad coverage
    (~93% of studies).

    ``strong`` — one CAM **per (study, finding)** against an MS-CXR expert
    **bounding box** drawn around that named finding. This is the only term that
    can support "the model looks at the pathology". Coverage is tiny: 823 train
    and 138 test studies.

    They are returned separately, never summed here, because they deserve
    different weights and because collapsing them would let the plentiful weak
    signal drown the scarce strong one and still look like it was learning
    localisation.

    Efficiency: the strong term takes one backward pass per *distinct finding*
    present among the boxed rows of the batch — typically one or two — not one
    per abnormality. Studies without a box cost nothing.
    """

    def __init__(self, top_k=0.5, strong_top_k=None):
        super().__init__()
        if not 0.0 < top_k <= 1.0:
            raise ValueError("top_k must be in (0, 1]")
        self.top_k = float(top_k)
        strong_top_k = self.top_k if strong_top_k is None else float(strong_top_k)
        if not 0.0 < strong_top_k <= 1.0:
            raise ValueError("strong_top_k must be in (0, 1]")
        self.strong_top_k = strong_top_k

    @staticmethod
    def _collect_streams(streams, batch_size):
        names, activations, grids = [], [], []
        for name, (stream, grid_hw) in streams.items():
            if stream.shape[0] != batch_size:
                raise ValueError(
                    f"stream {name!r} batch size does not match logits"
                )
            names.append(name)
            activations.append(stream)
            grids.append(grid_hw)
        return names, activations, grids

    def _weak_term(self, logits, labels, names, activations, grids, mask,
                   valid_mask, zero):
        score, valid = logit_difference_squared(
            logits, labels, sample_mask=valid_mask
        )
        if not valid.any():
            return zero, {name: zero for name in names}

        gradients = torch.autograd.grad(
            outputs=score.sum(), inputs=activations, create_graph=True
        )
        valid_float = valid.to(dtype=torch.float32)
        valid_count = valid_float.sum()
        per_stream = {}
        for name, stream, gradient, grid_hw in zip(
            names, activations, gradients, grids, strict=True
        ):
            cam = _cam_from_gradients(stream, gradient, grid_hw)
            grid_mask = resize_mask_to_grid(mask.to(device=cam.device), grid_hw)
            sample_losses = explanation_loss(cam, grid_mask, top_k=self.top_k)
            per_stream[name] = (sample_losses * valid_float).sum() / valid_count
        return torch.stack(list(per_stream.values())).mean(), per_stream

    def _strong_term(self, logits, names, activations, grids, bbox_masks,
                     bbox_valid, zero):
        """Per-(study, finding) CAM against expert boxes.

        ``bbox_masks``: [B, A, H, W]  ``bbox_valid``: [B, A] bool
        """
        if bbox_masks is None or bbox_valid is None:
            return zero
        bbox_valid = bbox_valid.to(device=logits.device, dtype=torch.bool)
        if bbox_valid.shape != logits.shape[:2]:
            raise ValueError(
                f"bbox_valid must be [B,A]; got {tuple(bbox_valid.shape)} "
                f"against logits {tuple(logits.shape[:2])}"
            )
        if bbox_masks.shape[:2] != bbox_valid.shape:
            raise ValueError("bbox_masks and bbox_valid disagree on [B,A]")
        if not bbox_valid.any():
            return zero

        # One backward per distinct finding, not per abnormality.
        label_indices = torch.nonzero(bbox_valid.any(dim=0), as_tuple=False)
        losses = []
        weights = []
        for label_tensor in label_indices.flatten():
            label_index = int(label_tensor.item())
            rows = bbox_valid[:, label_index]
            score = single_label_score(logits, label_index, rows)
            gradients = torch.autograd.grad(
                outputs=score, inputs=activations, create_graph=True
            )
            rows_float = rows.to(dtype=torch.float32)
            row_count = rows_float.sum()
            per_stream = []
            for stream, gradient, grid_hw in zip(
                activations, gradients, grids, strict=True
            ):
                cam = _cam_from_gradients(stream, gradient, grid_hw)
                grid_mask = resize_mask_to_grid(
                    bbox_masks[:, label_index].to(device=cam.device), grid_hw
                )
                sample_losses = explanation_loss(
                    cam, grid_mask, top_k=self.strong_top_k
                )
                per_stream.append(
                    (sample_losses * rows_float).sum() / row_count
                )
            losses.append(torch.stack(per_stream).mean())
            weights.append(row_count)

        if not losses:
            return zero
        # Weight each finding by how many boxed studies carried it, so the mean
        # is over (study, finding) pairs rather than over findings.
        weight_tensor = torch.stack(weights).to(dtype=torch.float32)
        return (
            torch.stack(losses) * weight_tensor
        ).sum() / weight_tensor.sum()

    def forward(self, logits, labels, streams, mask, valid_mask,
                bbox_masks=None, bbox_valid=None):
        """Return ``(weak, strong, per_stream_weak)``."""
        zero = logits.sum() * 0.0
        if not streams:
            return zero, zero, {}

        names, activations, grids = self._collect_streams(
            streams, logits.shape[0]
        )
        weak, per_stream = self._weak_term(
            logits, labels, names, activations, grids, mask, valid_mask, zero
        )
        strong = self._strong_term(
            logits, names, activations, grids, bbox_masks, bbox_valid, zero
        )
        return weak, strong, per_stream


def explanation_lambda(epoch, lambda_max, warmup_start_epoch, warmup_epochs):
    """Return the approved half-to-full linear explanation-loss warmup."""
    lambda_max = float(lambda_max)
    warmup_start_epoch = int(warmup_start_epoch)
    warmup_epochs = int(warmup_epochs)
    if epoch < warmup_start_epoch or lambda_max <= 0.0:
        return 0.0
    if warmup_epochs <= 0 or epoch >= warmup_start_epoch + warmup_epochs:
        return lambda_max

    progress = 0.5 + 0.5 * (
        (epoch - warmup_start_epoch) / warmup_epochs
    )
    return lambda_max * progress
