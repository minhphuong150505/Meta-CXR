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

    loss, strong, per_stream = loss_fn(
        logits,
        labels,
        {"tiny": (activations, (2, 2))},
        mask,
        torch.tensor([True]),
    )
    assert strong.item() == 0.0, "no boxes supplied -> strong term must be zero"
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


# --------------------------------------------------------------------------
# Strong (MS-CXR, per-pathology) term
# --------------------------------------------------------------------------

class _TwoLabelModule(nn.Module):
    """Logits for 2 findings driven by disjoint halves of a 2x2 activation grid.

    Finding 0 is driven by the LEFT column, finding 1 by the RIGHT column, so a
    per-finding CAM must light up a different half depending on which finding is
    scored. A pooled CAM cannot tell them apart -- which is the whole point of
    the strong term.

    Grad-CAM weights a channel by its gradient averaged over *tokens*, so a
    single-channel activation can only ever yield a flat CAM. Two channels are
    the minimum that can localise: channel 0 lives on the left column, channel 1
    on the right, and each finding reads one channel.
    """

    def __init__(self):
        super().__init__()
        # [B=1, N=4, C=2]; token order over a 2x2 grid is (0,0) (0,1) (1,0) (1,1).
        spatial = torch.zeros(1, 4, 2)
        spatial[0, [0, 2], 0] = 1.0        # channel 0 -> left column
        spatial[0, [1, 3], 1] = 1.0        # channel 1 -> right column
        self.spatial = nn.Parameter(spatial)

    def forward(self):
        activations = self.spatial
        pos0 = activations[..., 0].sum(dim=1)
        pos1 = activations[..., 1].sum(dim=1)
        zero = pos0 * 0.0
        finding0 = torch.stack((zero, pos0, zero), dim=-1).unsqueeze(1)
        finding1 = torch.stack((zero, pos1, zero), dim=-1).unsqueeze(1)
        return torch.cat((finding0, finding1), dim=1), activations


def _strong_inputs(box_label, box_grid):
    """Build [B,A,H,W] boxes with one finding boxed on a 2x2 grid."""
    bbox_masks = torch.zeros(1, 2, 2, 2)
    bbox_masks[0, box_label] = box_grid
    bbox_valid = torch.zeros(1, 2, dtype=torch.bool)
    bbox_valid[0, box_label] = True
    return bbox_masks, bbox_valid


def test_strong_term_is_zero_without_boxes():
    module = _TwoLabelModule()
    logits, activations = module()
    labels = torch.zeros(1, 2, dtype=torch.long)
    labels[0, 0] = 1

    _, strong, _ = ExplanationLoss(top_k=0.5)(
        logits, labels, {"tiny": (activations, (2, 2))},
        torch.ones(1, 2, 2), torch.ones(1, dtype=torch.bool),
        bbox_masks=torch.zeros(1, 2, 2, 2),
        bbox_valid=torch.zeros(1, 2, dtype=torch.bool),
    )
    assert strong.item() == 0.0, strong


def test_strong_term_is_disease_specific():
    """A box on the half that drives the boxed finding must score better.

    If the strong CAM were still the pooled one, both placements would give the
    same loss and this test would fail -- that is exactly the regression it
    exists to catch.
    """
    module = _TwoLabelModule()
    logits, activations = module()
    labels = torch.ones(1, 2, dtype=torch.long)

    left_box = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    right_box = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    # Finding 0 is driven by the LEFT column.
    masks_aligned, valid = _strong_inputs(0, left_box)
    masks_wrong, _ = _strong_inputs(0, right_box)

    kwargs = dict(
        logits=logits, labels=labels,
        streams={"tiny": (activations, (2, 2))},
        mask=torch.ones(1, 2, 2), valid_mask=torch.ones(1, dtype=torch.bool),
    )
    _, aligned, _ = ExplanationLoss(top_k=0.5)(
        **kwargs, bbox_masks=masks_aligned, bbox_valid=valid
    )
    _, wrong, _ = ExplanationLoss(top_k=0.5)(
        **kwargs, bbox_masks=masks_wrong, bbox_valid=valid
    )
    assert aligned.item() < wrong.item(), (aligned.item(), wrong.item())


def test_strong_term_backprops_and_handles_two_findings():
    module = _TwoLabelModule()
    logits, activations = module()
    labels = torch.ones(1, 2, dtype=torch.long)

    bbox_masks = torch.zeros(1, 2, 2, 2)
    bbox_masks[0, 0] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    bbox_masks[0, 1] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    bbox_valid = torch.ones(1, 2, dtype=torch.bool)

    _, strong, _ = ExplanationLoss(top_k=0.5)(
        logits, labels, {"tiny": (activations, (2, 2))},
        torch.ones(1, 2, 2), torch.ones(1, dtype=torch.bool),
        bbox_masks=bbox_masks, bbox_valid=bbox_valid,
    )
    assert torch.isfinite(strong)
    gradient = torch.autograd.grad(strong, module.spatial)[0]
    assert gradient is not None and torch.isfinite(gradient).all()


def test_strong_term_rejects_shape_mismatch():
    module = _TwoLabelModule()
    logits, activations = module()
    labels = torch.ones(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match=r"bbox_valid must be \[B,A\]"):
        ExplanationLoss(top_k=0.5)(
            logits, labels, {"tiny": (activations, (2, 2))},
            torch.ones(1, 2, 2), torch.ones(1, dtype=torch.bool),
            bbox_masks=torch.zeros(1, 5, 2, 2),
            bbox_valid=torch.zeros(1, 5, dtype=torch.bool),
        )


def test_separate_top_k_for_strong_and_weak():
    loss_fn = ExplanationLoss(top_k=0.2, strong_top_k=0.5)
    assert loss_fn.top_k == 0.2 and loss_fn.strong_top_k == 0.5
    assert ExplanationLoss(top_k=0.3).strong_top_k == 0.3
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError):
            ExplanationLoss(top_k=0.5, strong_top_k=bad)
