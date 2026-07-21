"""medgemma_direct must never silently become a text-only run.

The removed bug: ``AutoModelForImageTextToText`` was loaded inside a bare
``except Exception`` that fell through to ``AutoModelForCausalLM``. Any failure
-- auth, version skew, a missing processor file, OOM -- produced a text-only
language model that kept the name "native MedGemma image tower". Loss fell,
generation worked, NLG metrics looked normal, and the figure-9 vision row would
have been a language-prior baseline compared against the Q-Former row as though
only the visual pathway differed.

These tests use fakes; no transformers, no weights, no GPU.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "training"))

from training.medgemma.capabilities import (  # noqa: E402
    MultimodalCapabilityValidator,
    MultimodalModelLoadError,
    validate_multimodal_capability,
)
from training.pipeline_modes import (  # noqa: E402
    ABLATION_MODE,
    DEFAULT_PIPELINE_MODE,
    PIPELINE_MODES,
    TEXT_ONLY_LANGUAGE_PRIOR_ABLATION,
    requires_multimodal,
    resolve_pipeline_modes,
)

MODEL_ID = "google/medgemma-1.5-4b-it"


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeImageProcessor:
    pass


class MultimodalProcessor:
    def __init__(self):
        self.image_processor = FakeImageProcessor()

    def __call__(self, images=None, text=None, **kwargs):
        return {}


class TextOnlyProcessor:
    """A tokenizer-shaped processor: no image component, no ``images=`` kwarg."""

    def __call__(self, text=None, **kwargs):
        return {}


class VisionConfig:
    hidden_size = 1152


class MultimodalConfig:
    def __init__(self):
        self.vision_config = VisionConfig()


class TextOnlyConfig:
    hidden_size = 2048


class MultimodalModel:
    def __init__(self):
        self.config = MultimodalConfig()
        self.vision_tower = object()
        self.multi_modal_projector = object()

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None):
        return None


class PureCausalLM:
    """What the old fallback produced."""

    def __init__(self):
        self.config = TextOnlyConfig()

    def forward(self, input_ids=None, attention_mask=None):
        return None


# --------------------------------------------------------------------------
# validator
# --------------------------------------------------------------------------


def test_genuine_multimodal_model_passes():
    report = validate_multimodal_capability(
        MultimodalModel(), MultimodalProcessor(), model_id=MODEL_ID
    )
    assert report.multimodal is True
    assert report.failures == []
    assert all(report.checks.values())


def test_pure_text_model_is_rejected():
    """The exact object the old fallback produced must not pass."""
    with pytest.raises(MultimodalModelLoadError) as excinfo:
        validate_multimodal_capability(
            PureCausalLM(), MultimodalProcessor(), model_id=MODEL_ID
        )
    message = str(excinfo.value)
    assert "model_has_vision_config" in message
    assert "not_pure_causal_lm" in message


def test_processor_without_an_image_processor_is_rejected():
    with pytest.raises(MultimodalModelLoadError) as excinfo:
        validate_multimodal_capability(
            MultimodalModel(), TextOnlyProcessor(), model_id=MODEL_ID
        )
    assert "processor_has_image_processor" in str(excinfo.value)


def test_missing_processor_entirely_is_rejected():
    with pytest.raises(MultimodalModelLoadError):
        validate_multimodal_capability(MultimodalModel(), None, model_id=MODEL_ID)


def test_forward_that_cannot_accept_pixels_is_rejected():
    class NoPixelForward(MultimodalModel):
        def forward(self, input_ids=None, attention_mask=None):
            return None

    with pytest.raises(MultimodalModelLoadError) as excinfo:
        validate_multimodal_capability(
            NoPixelForward(), MultimodalProcessor(), model_id=MODEL_ID
        )
    assert "forward_accepts_pixel_values" in str(excinfo.value)


def test_kwargs_forward_is_accepted_when_config_confirms_vision():
    """A **kwargs forward cannot be shown to reject pixels by signature alone."""

    class KwargsForward(MultimodalModel):
        def forward(self, input_ids=None, **kwargs):
            return None

    assert validate_multimodal_capability(
        KwargsForward(), MultimodalProcessor(), model_id=MODEL_ID
    ).multimodal

def test_class_name_alone_does_not_satisfy_the_check():
    """A wrapper can carry the right name with no vision tower behind it."""

    class AutoModelForImageTextToText(PureCausalLM):
        pass

    assert type(AutoModelForImageTextToText()).__name__ == "AutoModelForImageTextToText"
    with pytest.raises(MultimodalModelLoadError):
        validate_multimodal_capability(
            AutoModelForImageTextToText(), MultimodalProcessor(), model_id=MODEL_ID
        )


def test_inspect_reports_without_raising():
    report = MultimodalCapabilityValidator().inspect(PureCausalLM(), TextOnlyProcessor())
    assert report.multimodal is False
    assert set(report.failures) >= {
        "processor_has_image_processor",
        "model_has_vision_config",
        "not_pure_causal_lm",
    }


# --------------------------------------------------------------------------
# error content
# --------------------------------------------------------------------------


def test_error_carries_model_id_revision_and_transformers_version():
    original = OSError("gated repo: 401")
    error = MultimodalModelLoadError(
        model_id=MODEL_ID,
        revision="abc123",
        transformers_version="4.57.0",
        original=original,
    )
    message = str(error)
    assert MODEL_ID in message
    assert "abc123" in message
    assert "4.57.0" in message
    assert "gated repo: 401" in message
    assert "OSError" in message
    assert error.original is original


def test_error_states_that_native_medgemma_requires_multimodal():
    message = str(
        MultimodalModelLoadError(
            model_id=MODEL_ID, revision=None, transformers_version=None, original=None
        )
    )
    assert "REQUIRES multimodal" in message
    assert "language-prior baseline" in message
    assert "text_only_language_prior_ablation" in message


def test_exception_context_is_preservable():
    """``raise ... from exc`` must keep the original traceback reachable."""
    original = RuntimeError("CUDA out of memory")
    try:
        try:
            raise original
        except RuntimeError as exc:
            raise MultimodalModelLoadError(
                model_id=MODEL_ID,
                revision=None,
                transformers_version=None,
                original=exc,
            ) from exc
    except MultimodalModelLoadError as raised:
        assert raised.__cause__ is original
        assert raised.original is original


# --------------------------------------------------------------------------
# pipeline modes
# --------------------------------------------------------------------------


def test_text_only_ablation_is_not_the_default():
    assert DEFAULT_PIPELINE_MODE == "medgemma_direct"
    assert DEFAULT_PIPELINE_MODE != TEXT_ONLY_LANGUAGE_PRIOR_ABLATION.name


def test_text_only_ablation_requires_explicit_selection():
    """No other mode selection can reach it -- including the ablation bundle."""
    for selection in (DEFAULT_PIPELINE_MODE, ABLATION_MODE, "meta_cxr_qformer", "native", "both"):
        names = [mode.name for mode in resolve_pipeline_modes(selection)]
        assert TEXT_ONLY_LANGUAGE_PRIOR_ABLATION.name not in names

    chosen = resolve_pipeline_modes(TEXT_ONLY_LANGUAGE_PRIOR_ABLATION.name)
    assert [mode.name for mode in chosen] == [TEXT_ONLY_LANGUAGE_PRIOR_ABLATION.name]


def test_text_only_ablation_is_not_called_native_medgemma():
    description = TEXT_ONLY_LANGUAGE_PRIOR_ABLATION.description
    assert "native MedGemma" not in description.replace("'native ", "'NATIVE ")
    assert "ABLATION, NOT A VISION PIPELINE" in description
    assert TEXT_ONLY_LANGUAGE_PRIOR_ABLATION.image_mode == "text_only"


def test_only_the_text_only_mode_waives_multimodal_capability():
    for name, mode in PIPELINE_MODES.items():
        expected = name != TEXT_ONLY_LANGUAGE_PRIOR_ABLATION.name
        assert requires_multimodal(mode) is expected


def test_text_only_ablation_needs_no_stage1():
    assert TEXT_ONLY_LANGUAGE_PRIOR_ABLATION.requires_stage1 is False


# --------------------------------------------------------------------------
# the fallback is gone from the source
# --------------------------------------------------------------------------


FIG9 = REPO_ROOT / "training" / "train_eval_figure9_llm_variants_200.py"


def test_no_except_falls_through_to_causal_lm():
    """Structural check: no handler may construct AutoModelForCausalLM.

    A comment or a docstring can claim the fallback is gone. This reads the
    file and fails if any ``except`` block instantiates the text-only class.
    """
    tree = __import__("ast").parse(FIG9.read_text(encoding="utf-8"))
    offenders = []
    for node in __import__("ast").walk(tree):
        if not isinstance(node, __import__("ast").ExceptHandler):
            continue
        for inner in __import__("ast").walk(node):
            if (
                isinstance(inner, __import__("ast").Call)
                and isinstance(inner.func, __import__("ast").Attribute)
                and inner.func.attr == "from_pretrained"
                and isinstance(inner.func.value, __import__("ast").Name)
                and inner.func.value.id == "AutoModelForCausalLM"
            ):
                offenders.append(node.lineno)
    assert offenders == [], (
        f"AutoModelForCausalLM is constructed inside an except block at line(s) "
        f"{offenders}; that is the silent text-only fallback this test forbids"
    )


def test_multimodal_load_failure_raises_the_typed_error():
    """The multimodal load must be wrapped in MultimodalModelLoadError."""
    source = FIG9.read_text(encoding="utf-8")
    assert "raise MultimodalModelLoadError(" in source
    assert "validate_multimodal_capability(" in source


def test_variant_llm_accepts_the_text_only_image_mode():
    source = FIG9.read_text(encoding="utf-8")
    assert '"qformer", "native", "text_only"' in source


def test_saved_metadata_records_multimodal_status():
    """A checkpoint must not be mistakable later for a vision run."""
    source = FIG9.read_text(encoding="utf-8")
    assert "capability.as_metadata()" in source


def test_capabilities_module_imports_no_transformers():
    """Keeps the validator testable on a CPU box with no Stage-2 deps."""
    import training.medgemma.capabilities as capabilities

    assert "transformers" not in inspect.getsource(capabilities).split("\n")[0]
    for name in dir(capabilities):
        assert not name.startswith("AutoModel")
