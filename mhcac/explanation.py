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
    def __init__(self, top_k=0.5):
        super().__init__()
        if not 0.0 < top_k <= 1.0:
            raise ValueError("top_k must be in (0, 1]")
        self.top_k = float(top_k)

    def forward(self, logits, labels, streams, mask, valid_mask):
        score, valid = logit_difference_squared(
            logits, labels, sample_mask=valid_mask
        )
        zero = logits.sum() * 0.0
        if not streams:
            return zero, {}

        names = []
        activations = []
        grids = []
        for name, (stream, grid_hw) in streams.items():
            if stream.shape[0] != logits.shape[0]:
                raise ValueError(
                    f"stream {name!r} batch size does not match logits"
                )
            names.append(name)
            activations.append(stream)
            grids.append(grid_hw)

        if not valid.any():
            return zero, {name: zero for name in names}

        gradients = torch.autograd.grad(
            outputs=score.sum(),
            inputs=activations,
            create_graph=True,
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
