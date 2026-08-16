"""The mention gate is the model's only way to say "nothing to report here".

Without it the model must pick Positive/Negative/Uncertain for all fourteen
findings on every image, including the 79.5% of cells the radiologist never
wrote about. Measured consequence on the test split before this head existed:
10.8 of 14 findings called Positive per study, and every single study labelled
`No Finding = Positive` while 8.8 other findings were flagged on it.

Each failure mode here is silent -- the loss stays finite and falls either way.
"""

import inspect

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


def test_gate_targets_are_not_left_in_the_dataframe():
    """Fourteen extra columns cost 16x per __getitem__, so they must be dropped.

    `self.annotation.iloc[i]` on a mixed-dtype frame consolidates every block
    into one object Series, so its cost scales with the column count. Adding the
    fourteen int8 mention columns took a study from 42.5 ms to 687 ms, cut the
    loader from 92.9 studies/s to about 17, and put GPU utilisation back to 15%
    -- with no error anywhere, just a run four times longer than it should be.
    The targets live in a positional numpy array instead.
    """
    pytest.importorskip("torchvision")
    try:
        from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset
    except Exception as exc:  # private configs/env_config.yaml
        pytest.skip(f"ReportDataset not importable here ({type(exc).__name__})")

    source = inspect.getsource(MIMIC_CXR_Dataset.__init__)
    assert "*self.mention_cols," in source, (
        "mention columns must be dropped from self.annotation after the matrix "
        "is built; leaving them in is a silent 16x slowdown"
    )
    assert "_mention_matrix" in source
    item = inspect.getsource(MIMIC_CXR_Dataset.__getitem__)
    assert "self._mention_matrix[" in item, "read the array, not the DataFrame row"
    assert "ann[self.mention_cols]" not in item


# --------------------------------------------------------------------------
# Mention-conditioned classification (hierarchical gate x classifier)
# --------------------------------------------------------------------------

from mhcac.loss import (  # noqa: E402
    MentionConditionedClassificationLoss,
    mention_marginal_log_probs,
)


def test_silence_suppresses_a_confident_positive():
    """The whole point: a closed gate must veto the classifier, not sit next to it.

    Before the hierarchy the two heads were independent — the gate could say
    "never mentioned" while the classifier said "Positive" and nothing
    reconciled them. Measured cost on the 2026-08-15 run: macro specificity
    0.2637, four labels at ~0.
    """
    conditional = torch.tensor([[[-6.0, 6.0, 0.0]]])   # screams Positive
    mention = torch.tensor([[-8.0]])                   # says: never mentioned

    log_p = mention_marginal_log_probs(conditional, mention)
    probs = log_p.exp()

    assert torch.allclose(probs.sum(-1), torch.ones(1, 1), atol=1e-6), probs
    assert probs[0, 0, 1].item() < 1e-3, probs[0, 0, 1].item()
    assert probs.argmax(-1).item() == 0, "must fall back to Negative"


def test_open_gate_lets_the_classifier_through():
    conditional = torch.tensor([[[-6.0, 6.0, 0.0]]])
    mention = torch.tensor([[8.0]])                    # definitely mentioned

    probs = mention_marginal_log_probs(conditional, mention).exp()

    assert torch.allclose(probs.sum(-1), torch.ones(1, 1), atol=1e-6)
    assert probs.argmax(-1).item() == 1
    assert probs[0, 0, 1].item() > 0.99


def test_unmentioned_target_trains_only_the_mention_head():
    conditional = torch.tensor([[[-6.0, 6.0, 0.0]]], requires_grad=True)
    mention = torch.tensor([[-8.0]], requires_grad=True)
    loss_fn = MentionConditionedClassificationLoss(num_abnormalities=1)

    loss = loss_fn(
        conditional,
        mention,
        labels=torch.tensor([[-100]]),            # class unknown
        mention_targets=torch.tensor([[0.0]]),    # not mentioned
    )
    loss.backward()

    assert mention.grad.item() > 0, (
        "gradient descent must push the logit further toward silence"
    )
    assert torch.count_nonzero(conditional.grad).item() == 0, (
        "an unmentioned cell must not train the conditional classifier"
    )


def test_mentioned_positive_trains_both_heads_in_opposite_directions():
    conditional = torch.tensor([[[2.0, -2.0, 0.0]]], requires_grad=True)
    mention = torch.tensor([[-3.0]], requires_grad=True)
    loss_fn = MentionConditionedClassificationLoss(num_abnormalities=1)

    loss = loss_fn(
        conditional,
        mention,
        labels=torch.tensor([[1]]),               # Positive
        mention_targets=torch.tensor([[1.0]]),    # mentioned
    )
    loss.backward()

    assert mention.grad.item() < 0, "must open the gate"
    assert conditional.grad[0, 0, 1].item() < 0, "must raise the Positive logit"
    assert conditional.grad[0, 0, 0].item() > 0, "must lower the Negative logit"


def test_ignored_class_still_trains_the_mention_head():
    """Whether a finding was written about is known even when its polarity is not."""
    conditional = torch.tensor([[[1.0, 1.0, 1.0]]], requires_grad=True)
    mention = torch.tensor([[-1.0]], requires_grad=True)

    loss = MentionConditionedClassificationLoss(num_abnormalities=1)(
        conditional,
        mention,
        labels=torch.tensor([[-100]]),
        mention_targets=torch.tensor([[1.0]]),    # mentioned, class masked out
    )
    loss.backward()

    assert mention.grad.item() < 0
    assert torch.count_nonzero(conditional.grad).item() == 0


def test_marginals_reject_shape_mismatch():
    with pytest.raises(ValueError, match=r"mention_logits must be \[B, A\]"):
        mention_marginal_log_probs(torch.zeros(1, 2, 3), torch.zeros(1, 5))
    with pytest.raises(ValueError, match="must be"):
        mention_marginal_log_probs(torch.zeros(2, 3), torch.zeros(2, 3))


def test_sample_mask_drops_unlabelled_studies():
    conditional = torch.zeros(2, 1, 3, requires_grad=True)
    mention = torch.zeros(2, 1, requires_grad=True)
    loss = MentionConditionedClassificationLoss(num_abnormalities=1)(
        conditional,
        mention,
        labels=torch.tensor([[1], [1]]),
        mention_targets=torch.tensor([[1.0], [1.0]]),
        sample_mask=torch.tensor([True, False]),
    )
    loss.backward()
    assert torch.count_nonzero(mention.grad[1]).item() == 0
    assert torch.count_nonzero(mention.grad[0]).item() > 0


def test_wrong_silence_costs_more_than_speaking_up():
    """Hiding a finding the radiologist DID write about must not be cheap.

    Unweighted, the hierarchy charges both directions identically (1.00x) even
    though 79.5% of cells are blank, so silence is the majority answer. The
    separate gate BCE it replaces charged a wrong silence 4-10x more.
    """
    cond = torch.zeros(1, 1, 3)
    silent, spoke = torch.tensor([[-3.0]]), torch.tensor([[3.0]])
    mentioned, blank = torch.tensor([[1.0]]), torch.tensor([[0.0]])
    labels = torch.tensor([[-100]])

    flat = MentionConditionedClassificationLoss(num_abnormalities=1)
    a = flat(cond, silent, labels, mentioned).item()
    b = flat(cond, spoke, labels, blank).item()
    assert abs(a / b - 1.0) < 1e-6, "unweighted must stay symmetric"

    weighted = MentionConditionedClassificationLoss(
        pos_weights=[2.959], num_abnormalities=1
    )
    a_w = weighted(cond, silent, labels, mentioned).item()
    b_w = weighted(cond, spoke, labels, blank).item()
    assert abs(a_w / b_w - 2.959) < 1e-3, (a_w, b_w)
    assert abs(b_w - b) < 1e-6, "speaking up must be untouched by the weight"


def test_mention_weight_is_capped_and_validated():
    capped = MentionConditionedClassificationLoss(
        pos_weights=[50.0], num_abnormalities=1, weight_cap=10.0
    )
    assert capped.pos_weight.item() == 10.0

    with pytest.raises(ValueError, match="one value per abnormality"):
        MentionConditionedClassificationLoss(pos_weights=[1.0, 2.0], num_abnormalities=1)
    with pytest.raises(ValueError, match="must be positive"):
        MentionConditionedClassificationLoss(pos_weights=[0.0], num_abnormalities=1)


def test_weighting_leaves_the_conditional_class_term_calibrated():
    """Only the mention term is weighted; q must be untouched."""
    cond = torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True)
    mention = torch.tensor([[5.0]])          # gate wide open, mention term ~0
    flat = MentionConditionedClassificationLoss(num_abnormalities=1)(
        cond, mention, torch.tensor([[1]]), torch.tensor([[1.0]])
    )
    weighted = MentionConditionedClassificationLoss(
        pos_weights=[10.0], num_abnormalities=1
    )(cond, mention, torch.tensor([[1]]), torch.tensor([[1.0]]))
    # The class term is identical; only the (tiny) mention term is scaled.
    assert weighted.item() > flat.item()
    assert abs((weighted - flat).item()) < 0.1, "class term must not be rescaled"
