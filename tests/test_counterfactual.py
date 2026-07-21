"""Counterfactual audit, exercised on CPU with deterministic fake backends.

The point of the audit is to catch a generator that scores well on NLG metrics
while ignoring the radiograph. These tests therefore include both an
image-dependent fake and a language-prior fake, and assert the audit separates
them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.evaluation import perturbations as pert  # noqa: E402
from training.evaluation.counterfactual import (  # noqa: E402
    AuditSample,
    CounterfactualConfig,
    CounterfactualEvaluator,
    assert_shareable,
    lexical_change,
    privacy_violations,
)

C, H, W = 1, 32, 32


def make_image(seed: int) -> torch.Tensor:
    return torch.randn(C, H, W, generator=torch.Generator().manual_seed(seed))


class ImageDependentBackend:
    """Describes the image, so any perturbation changes the wording."""

    def generate(self, images, prompts):
        out = []
        for image in images:
            if image is None:
                out.append("no image was provided so nothing can be described")
                continue
            mean = float(image.mean())
            top = float(image[:, : H // 2, :].mean())
            out.append(
                f"the mean density is {mean:.3f} and the upper zone density is {top:.3f} "
                f"with peak {float(image.max()):.3f} and trough {float(image.min()):.3f}"
            )
        return out


class LanguagePriorBackend:
    """Ignores the image entirely -- the failure mode being hunted."""

    def generate(self, images, prompts):
        return ["no acute cardiopulmonary process is identified" for _ in images]


class FakeClinicalBackend:
    name = "fake_clinical_for_tests"

    def compare(self, original, perturbed):
        return {
            "clinical_change": 0.0 if original == perturbed else 1.0,
            "label_change": [] if original == perturbed else ["Pneumonia"],
        }


def make_samples(n: int) -> list[AuditSample]:
    return [
        AuditSample(sample_key=f"key{i}", image=make_image(i), prompt="write findings")
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# perturbations
# --------------------------------------------------------------------------


def test_blank_image_is_all_zeros_and_shape_preserving():
    image = make_image(0)
    out = pert.blank_image(image)
    assert out.shape == image.shape
    assert torch.all(out == 0)


def test_constant_image_preserves_mean_but_removes_structure():
    image = make_image(0)
    out = pert.constant_image(image)
    # Not exactly 0: fp32 std over a constant tensor lands around 1e-9.
    assert float(out.std()) == pytest.approx(0.0, abs=1e-6)
    assert float(out.mean()) == pytest.approx(float(image.mean()), abs=1e-5)


def test_shuffled_pixels_preserves_the_histogram():
    image = make_image(0)
    out = pert.shuffled_pixels(image, seed=1)
    torch.testing.assert_close(out.flatten().sort().values, image.flatten().sort().values)
    assert not torch.equal(out, image)


def test_shuffled_patches_preserves_the_multiset_of_values():
    image = make_image(0)
    out = pert.shuffled_patches(image, seed=1, patch=8)
    torch.testing.assert_close(out.flatten().sort().values, image.flatten().sort().values)
    assert not torch.equal(out, image)


def test_perturbations_are_deterministic_given_a_seed():
    image = make_image(0)
    assert torch.equal(pert.shuffled_pixels(image, 5), pert.shuffled_pixels(image, 5))
    assert not torch.equal(pert.shuffled_pixels(image, 5), pert.shuffled_pixels(image, 6))


def test_region_occlusion_zeros_only_the_requested_corner():
    image = torch.ones(C, H, W)
    out = pert.region_occlusion(image, fraction=0.25, corner="top_left")
    assert torch.all(out[:, : H // 2, : W // 2] == 0)
    assert torch.all(out[:, H // 2 :, W // 2 :] == 1)


def test_patch_larger_than_image_raises():
    with pytest.raises(ValueError, match="smaller than one"):
        pert.shuffled_patches(make_image(0), seed=1, patch=64)


def test_wrong_rank_image_raises():
    with pytest.raises(ValueError, match=r"\[C, H, W\]"):
        pert.blank_image(torch.randn(4, C, H, W))


def test_donor_is_never_the_sample_itself():
    pool = [pert.Donor(f"key{i}", make_image(i)) for i in range(4)]
    for i in range(4):
        assert pert.pick_random_donor(f"key{i}", pool, seed=i).sample_key != f"key{i}"


def test_single_sample_cohort_cannot_swap():
    pool = [pert.Donor("only", make_image(0))]
    with pytest.raises(ValueError, match="no donor available"):
        pert.pick_random_donor("only", pool, seed=0)


# --------------------------------------------------------------------------
# lexical change
# --------------------------------------------------------------------------


def test_identical_reports_have_zero_change():
    assert lexical_change("clear lungs", "clear lungs") == 0.0


def test_disjoint_reports_have_maximal_change():
    assert lexical_change("clear lungs", "large effusion") == 1.0


def test_two_empty_reports_are_not_reported_as_changed():
    assert lexical_change("", "") == 0.0


# --------------------------------------------------------------------------
# evaluator
# --------------------------------------------------------------------------


def test_image_dependent_model_scores_above_the_threshold():
    evaluator = CounterfactualEvaluator(
        backend=ImageDependentBackend(),
        config=CounterfactualConfig(perturbations=pert.SELF_CONTAINED),
    )
    result = evaluator.evaluate(make_samples(3))

    assert result["mean_visual_reliance_score"] > 0.1
    assert result["language_prior_dependent"] is False


def test_language_prior_model_is_flagged():
    """A model that ignores the image must not pass the audit."""
    evaluator = CounterfactualEvaluator(
        backend=LanguagePriorBackend(),
        config=CounterfactualConfig(perturbations=pert.SELF_CONTAINED),
    )
    result = evaluator.evaluate(make_samples(3))

    assert result["mean_visual_reliance_score"] == 0.0
    assert result["language_prior_dependent"] is True


def test_random_swap_changes_an_image_dependent_report():
    evaluator = CounterfactualEvaluator(
        backend=ImageDependentBackend(),
        config=CounterfactualConfig(perturbations=(pert.RANDOM_IMAGE_SWAP,)),
    )
    result = evaluator.evaluate(make_samples(4))

    for row in result["records"]:
        assert row["donor_sample_key"] is not None
        assert row["donor_sample_key"] != row["sample_key"]
        assert row["lexical_text_change"] > 0


def test_hard_negative_swap_uses_the_configured_partner():
    samples = make_samples(3)
    evaluator = CounterfactualEvaluator(
        backend=ImageDependentBackend(),
        config=CounterfactualConfig(
            perturbations=(pert.HARD_NEGATIVE_SWAP,),
            hard_negatives={"key0": "key2", "key1": "key0", "key2": "key1"},
        ),
    )
    result = evaluator.evaluate(samples)
    mapping = {row["sample_key"]: row["donor_sample_key"] for row in result["records"]}
    assert mapping == {"key0": "key2", "key1": "key0", "key2": "key1"}


def test_hard_negative_without_configuration_raises():
    evaluator = CounterfactualEvaluator(
        backend=ImageDependentBackend(),
        config=CounterfactualConfig(perturbations=(pert.HARD_NEGATIVE_SWAP,)),
    )
    with pytest.raises(ValueError, match="no hard negative configured"):
        evaluator.evaluate(make_samples(2))


def test_unknown_perturbation_is_rejected_at_config_time():
    with pytest.raises(ValueError, match="unknown perturbation"):
        CounterfactualConfig(perturbations=("rotate_180",))


def test_empty_cohort_raises():
    evaluator = CounterfactualEvaluator(backend=ImageDependentBackend())
    with pytest.raises(ValueError, match="no samples"):
        evaluator.evaluate([])


# --------------------------------------------------------------------------
# clinical fields must not be faked
# --------------------------------------------------------------------------


def test_clinical_fields_are_none_and_explained_without_a_backend():
    evaluator = CounterfactualEvaluator(
        backend=ImageDependentBackend(),
        config=CounterfactualConfig(perturbations=(pert.BLANK_IMAGE,)),
    )
    result = evaluator.evaluate(make_samples(2))

    for row in result["records"]:
        assert row["clinical_change"] is None
        assert row["label_change"] is None
        assert any("no clinical backend" in note for note in row["notes"])
        # The lexical number must never be copied into a clinical field.
        assert row["lexical_text_change"] != row["clinical_change"]
    assert result["clinical_backend"] is None


def test_clinical_backend_is_used_when_supplied():
    evaluator = CounterfactualEvaluator(
        backend=ImageDependentBackend(),
        config=CounterfactualConfig(perturbations=(pert.BLANK_IMAGE,)),
        clinical_backend=FakeClinicalBackend(),
    )
    result = evaluator.evaluate(make_samples(2))

    assert result["clinical_backend"] == "fake_clinical_for_tests"
    for row in result["records"]:
        assert row["clinical_change"] == 1.0
        assert row["label_change"] == ["Pneumonia"]
        assert row["notes"] == []


def test_score_definition_says_it_is_lexical():
    evaluator = CounterfactualEvaluator(
        backend=ImageDependentBackend(),
        config=CounterfactualConfig(perturbations=(pert.BLANK_IMAGE,)),
    )
    result = evaluator.evaluate(make_samples(2))
    assert "Lexical only" in result["score_definition"]
    assert "not a clinical measure" in result["score_definition"]


# --------------------------------------------------------------------------
# privacy
# --------------------------------------------------------------------------


def test_audit_output_carries_no_identifiers():
    evaluator = CounterfactualEvaluator(
        backend=ImageDependentBackend(),
        config=CounterfactualConfig(perturbations=pert.SELF_CONTAINED),
    )
    result = evaluator.evaluate(make_samples(3))
    assert privacy_violations(result) == []
    assert_shareable(result)


def test_privacy_check_finds_a_deeply_nested_identifier():
    """Top-level-only checking would miss this."""
    payload = {"records": [{"meta": {"inner": [{"subject_id": 10000032}]}}]}
    assert privacy_violations(payload) == ["records[0].meta.inner[0].subject_id"]
    with pytest.raises(ValueError, match="subject_id"):
        assert_shareable(payload)


@pytest.mark.parametrize(
    "key", ["subject_id", "study_id", "dicom_id", "image_path", "ref", "report"]
)
def test_each_forbidden_key_is_caught_when_nested(key):
    payload = {"records": [{"a": {"b": {key: "value"}}}]}
    assert privacy_violations(payload) == [f"records[0].a.b.{key}"]
