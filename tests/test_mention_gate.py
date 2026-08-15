"""The mention gate is the model's only way to say "nothing to report here".

Without it the model must pick Positive/Negative/Uncertain for all fourteen
findings on every image, including the 79.5% of cells the radiologist never
wrote about. Measured consequence on the test split before this head existed:
10.8 of 14 findings called Positive per study, and every single study labelled
`No Finding = Positive` while 8.8 other findings were flagged on it.

Each failure mode here is silent -- the loss stays finite and falls either way.
"""

import pytest
import torch

from mhcac.loss import MentionGateLoss


def test_weights_are_capped():
    """Three labels have raw ratios of 26-157; uncapped they own the loss."""
    loss = MentionGateLoss(3, pos_weights=[2.0, 26.7, 157.0], weight_cap=10.0)

    assert loss.pos_weight.tolist() == [2.0, 10.0, 10.0]


def test_a_wrongly_silent_gate_costs_more_than_a_talkative_one():
    """kappa > 1 must make hiding a mentioned finding the expensive error."""
    loss = MentionGateLoss(1, pos_weights=[4.0])
    target_mentioned = torch.ones(1, 1)
    target_silent = torch.zeros(1, 1)
    confident_silence = torch.full((1, 1), -4.0)
    confident_speech = torch.full((1, 1), 4.0)

    missed = loss(confident_silence, target_mentioned)   # hid a real finding
    spurious = loss(confident_speech, target_silent)     # spoke unnecessarily

    assert missed > spurious


def test_studies_without_a_chexpert_record_do_not_train_the_gate():
    """Their blank pattern is unknown, not empty."""
    loss = MentionGateLoss(3, pos_weights=[1.0, 1.0, 1.0])
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.zeros(4, 3)
    mask = torch.tensor([True, True, False, True])

    loss(logits, targets, sample_mask=mask).backward()

    assert torch.count_nonzero(logits.grad[2]) == 0, "masked study must not train"
    assert torch.count_nonzero(logits.grad[0]) > 0


def test_an_all_masked_batch_stays_finite_and_differentiable():
    loss = MentionGateLoss(2, pos_weights=[1.0, 1.0])
    logits = torch.randn(3, 2, requires_grad=True)

    value = loss(logits, torch.zeros(3, 2), sample_mask=torch.zeros(3, dtype=torch.bool))
    value.backward()

    assert float(value.detach()) == 0.0
    assert logits.grad is not None


def test_shape_and_weight_length_errors_are_loud():
    with pytest.raises(ValueError, match="one value per abnormality"):
        MentionGateLoss(14, pos_weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="must be positive"):
        MentionGateLoss(2, pos_weights=[1.0, 0.0])
    loss = MentionGateLoss(3, pos_weights=[1.0] * 3)
    with pytest.raises(ValueError, match="shape mismatch"):
        loss(torch.randn(2, 3), torch.zeros(2, 4))
    with pytest.raises(ValueError, match="expected 3 abnormalities"):
        loss(torch.randn(2, 4), torch.zeros(2, 4))


def test_gate_head_emits_one_logit_per_abnormality():
    """MHCAC must return the gate logits alongside the classification logits."""
    from mhcac.mhcac_12 import AbnormalityClassificationModel, StreamLayout
    from vision_encoders.shared_visual_tokens import SharedVisualTokens

    model = AbnormalityClassificationModel(
        embed_dim=32, num_heads=4, num_layers=2, num_commmon_tokens=14,
        visual_dim=1408, num_text_teacher_layers=0,
        stream_layouts={"biovil": StreamLayout(196)},
    )
    tokens = SharedVisualTokens(
        tokens=torch.randn(2, 196, 1408), spans={"biovil": slice(0, 196)}
    )

    logits, _, _, _, _, mention_logits = model(tokens)

    assert logits.shape == (2, 14, 3)
    assert mention_logits.shape == (2, 14)
    mention_logits.sum().backward()
    assert model.mention_heads[0].weight.grad is not None
