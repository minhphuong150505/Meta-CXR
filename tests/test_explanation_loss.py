import pytest
import torch
import torch.nn as nn

from mhcac.explanation import (
    ExplanationLoss,
    explanation_lambda,
    explanation_loss,
    grad_cam,
    logit_difference_squared,
    resize_mask_to_grid,
)
from mhcac.mhcac_12 import AbnormalityClassificationModel
from vision_encoders.shared_visual_tokens import SharedVisualTokens


def test_logit_difference_is_zero_without_positive_labels():
    logits = torch.randn(2, 14, 3)
    labels = torch.zeros(2, 14, dtype=torch.long)

    score, valid = logit_difference_squared(logits, labels)

    assert torch.equal(score, torch.zeros_like(score))
    assert torch.equal(valid, torch.tensor([False, False]))


def test_logit_difference_uses_only_the_positive_disease():
    logits = torch.randn(1, 14, 3)
    labels = torch.zeros(1, 14, dtype=torch.long)
    labels[0, 6] = 1
    logits[0, 6, 0] = -1.5
    logits[0, 6, 1] = 2.0

    score, valid = logit_difference_squared(logits, labels)

    assert score.item() == pytest.approx((2.0 - (-1.5)) ** 2)
    assert valid.item() is True


def test_explanation_loss_inside_and_outside_mask_limits():
    cam = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    inside = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    outside = 1.0 - inside

    inside_loss = explanation_loss(cam, inside, top_k=0.5)
    outside_loss = explanation_loss(cam, outside, top_k=0.5)

    assert inside_loss.item() == pytest.approx(0.0, abs=1e-5)
    assert outside_loss.item() == pytest.approx(1.0, abs=1e-6)


def test_top_k_keeps_half_the_pixels_and_preserves_soft_gradient():
    cam_values = torch.arange(16, dtype=torch.float32, requires_grad=True)
    cam = cam_values.reshape(1, 4, 4)
    mask = torch.zeros_like(cam)
    mask[:, -1, :] = 1.0

    flat = cam.flatten(1)
    normalized = (flat - flat.amin(dim=1, keepdim=True)) / (
        flat.amax(dim=1, keepdim=True) - flat.amin(dim=1, keepdim=True) + 1e-6
    )
    threshold = torch.quantile(normalized, 0.5, dim=-1).detach()
    kept = normalized >= threshold[:, None]
    positive = normalized * kept.to(normalized.dtype)

    assert kept.sum().item() == pytest.approx(cam.numel() * 0.5, abs=1)
    positive_grad = torch.autograd.grad(
        positive.sum(), cam_values, retain_graph=True
    )[0]
    assert torch.count_nonzero(positive_grad).item() > 0

    loss = explanation_loss(cam, mask, top_k=0.5).sum()
    loss_grad = torch.autograd.grad(loss, cam_values)[0]
    assert torch.count_nonzero(loss_grad).item() > 0


def test_grad_cam_matches_channel_weight_definition():
    activations = torch.tensor(
        [[[1.0, 2.0], [2.0, 1.0], [3.0, 0.0], [4.0, -1.0]]],
        requires_grad=True,
    )
    score = activations.square().sum(dim=(1, 2))

    cam = grad_cam(score, activations, (2, 2))
    gradients = 2.0 * activations
    weights = gradients.mean(dim=1, keepdim=True)
    expected = torch.relu((weights * activations).sum(dim=-1)).reshape(1, 2, 2)

    assert cam.dtype == torch.float32
    assert torch.allclose(cam, expected)


def test_explanation_gradient_reaches_tiny_module_parameter():
    class TinyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial = nn.Parameter(torch.tensor([0.2, 0.5, 1.0, 2.0]))

        def forward(self):
            activations = self.spatial.reshape(1, 4, 1)
            coefficients = torch.tensor(
                [1.0, 2.0, 3.0, 4.0], device=activations.device
            )
            positive_logit = (activations[..., 0] * coefficients).sum(dim=1)
            zero = positive_logit * 0.0
            first_disease = torch.stack(
                (zero, positive_logit, zero), dim=-1
            ).unsqueeze(1)
            other_diseases = zero.reshape(1, 1, 1).expand(1, 13, 3)
            logits = torch.cat((first_disease, other_diseases), dim=1)
            return logits, activations

    module = TinyModule()
    logits, activations = module()
    labels = torch.zeros(1, 14, dtype=torch.long)
    labels[:, 0] = 1
    mask = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]])
    loss_fn = ExplanationLoss(top_k=0.5)

    loss, per_stream = loss_fn(
        logits,
        labels,
        {"tiny": (activations, (2, 2))},
        mask,
        torch.tensor([True]),
    )
    gradient = torch.autograd.grad(loss, module.spatial)[0]

    assert set(per_stream) == {"tiny"}
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(0, 0.0), (1, 0.0), (2, 0.125), (3, 0.1875), (4, 0.25), (9, 0.25)],
)
def test_explanation_lambda_matches_approved_warmup(epoch, expected):
    value = explanation_lambda(
        epoch,
        lambda_max=0.25,
        warmup_start_epoch=2,
        warmup_epochs=2,
    )

    assert value == pytest.approx(expected)


def test_resize_mask_to_grid_returns_binary_grid():
    mask = torch.zeros(1, 4, 4)
    mask[0, 0, 0] = 1.0

    resized = resize_mask_to_grid(mask, (2, 2))

    assert resized.shape == (1, 2, 2)
    assert torch.equal(resized, torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]))


def test_mhcac_only_keeps_cam_streams_while_capture_is_enabled():
    model = AbnormalityClassificationModel(
        embed_dim=8,
        num_heads=2,
        num_abnormalities=2,
        num_classes=3,
        num_layers=1,
        num_commmon_tokens=2,
        visual_dim=8,
        txt_dim=8,
        target_patch_count=4,
        num_text_teacher_layers=0,
        use_cnn=False,
    ).eval()
    shared = SharedVisualTokens(
        tokens=torch.randn(1, 4, 8),
        spans={"pubmedclip": slice(0, 4)},
    )

    model(shared)
    assert model._last_cam_streams is None

    model.capture_streams = True
    model(shared)
    captured = model._last_cam_streams
    assert set(captured) == {"pubmedclip"}
    assert captured["pubmedclip"][0].requires_grad
    assert captured["pubmedclip"][1] == (2, 2)

    model.capture_streams = False
    model(shared)
    assert model._last_cam_streams is None


def test_explanation_loss_cannot_run_without_a_live_graph():
    """Pins why Blip2Qformer.forward gates the term on torch.is_grad_enabled().

    Grad-CAM differentiates the score with respect to the visual activations, so
    it needs a graph.  RunnerBase.eval_epoch is decorated ``@torch.no_grad()``
    and validation reaches the same ``forward()`` as training, so computing this
    term there raises instead of returning a number.  If this assertion ever
    stops holding, re-read that gate before deleting it: the crash it prevents
    lands on the first scored epoch, roughly a day into a run.
    """
    tokens = torch.randn(1, 4, 8, requires_grad=True)
    head = nn.Linear(8, 6)
    labels = torch.zeros(1, 2, dtype=torch.long)
    labels[0, 0] = 1

    with torch.no_grad():
        logits = head(tokens).mean(dim=1).reshape(1, 2, 3)
        with pytest.raises(RuntimeError, match="does not require grad"):
            ExplanationLoss(top_k=0.5)(
                logits,
                labels,
                {"biovil": (tokens, (2, 2))},
                torch.ones(1, 4, 4),
                torch.ones(1, dtype=torch.bool),
            )
