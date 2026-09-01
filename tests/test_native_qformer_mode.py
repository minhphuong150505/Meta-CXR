"""The combined visual channel: MedGemma's own image PLUS Q-Former soft tokens.

``native_qformer_guided`` is the originally-designed Stage-2 architecture. The
two ``qformer_*`` modes SUBSTITUTE 32 soft tokens for the image; this one
SUPPLEMENTS it, which is a different experimental claim and needs its own
coverage. Everything here is torch-free apart from one stage2_utils call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2.prompts import PromptBuilder, load_prompt_config  # noqa: E402
from stage2.prompts.records import context_from_record  # noqa: E402
from stage2.prompts.schemas import PartKind, VisualMode  # noqa: E402
from training.pipeline_modes import resolve_pipeline_modes  # noqa: E402
from training.stage2_utils import SCHEMA_VERSION, adapter_is_complete  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "experiment_native_qformer_guided.yaml"
CONTROL = REPO_ROOT / "configs" / "experiment_native_anchor_only.yaml"

STAGE1_RECORD = {
    "study_id": "s",
    "ref": "target",
    "image_path": "/nowhere/anchor.jpg",
    "pred_groups": {
        "positive": ["Pleural Effusion"],
        "uncertain": ["Pneumonia"],
        "negative": ["Pneumothorax"],
    },
    "anchor_view": "PA",
    "auxiliary_views": [],
}
NATIVE_MANIFEST_RECORD = {
    "sample_key": "abc",
    "ref": "target",
    "image_path": "/nowhere/anchor.jpg",
    "anchor_view": "PA",
    "auxiliary_views": [],
}


def _parts(config_path):
    config = load_prompt_config(config_path)
    context = context_from_record(
        STAGE1_RECORD,
        visual_mode=config.visual_mode,
        qformer_token_count=32,
        prompt_version=config.version,
    )
    return PromptBuilder(config).build(context)


def test_the_mode_claims_both_visual_channels():
    mode = VisualMode.NATIVE_QFORMER_GUIDED
    assert mode.uses_native_image
    assert mode.uses_soft_tokens
    assert mode.includes_structured


def test_the_storage_key_is_its_own_third_value():
    """Reusing "native" or "qformer" would make every branch half-right."""
    assert VisualMode.NATIVE_QFORMER_GUIDED.image_mode == "native_qformer"
    assert VisualMode.NATIVE_ANCHOR_ONLY.image_mode == "native"
    assert VisualMode.QFORMER_GUIDED.image_mode == "qformer"


def test_the_prompt_carries_image_soft_tokens_and_cues_together():
    parts = _parts(CONFIG)
    kinds = [part.kind for part in parts.parts]
    assert PartKind.IMAGE in kinds, "MedGemma must still receive its own pixels"
    soft = [part for part in parts.parts if part.kind is PartKind.SOFT_TOKENS]
    assert [part.count for part in soft] == [32]
    text = parts.user_text()
    assert text.count("<qformer_soft_token>") == 32
    assert "Pleural Effusion" in text


def test_the_control_arm_has_neither_soft_tokens_nor_cues():
    """Otherwise the A/B measures nothing."""
    parts = _parts(CONTROL)
    assert PartKind.SOFT_TOKENS not in [part.kind for part in parts.parts]
    text = parts.user_text()
    assert "<qformer_soft_token>" not in text
    assert "Pleural Effusion" not in text


@pytest.mark.parametrize(
    "mode",
    [
        VisualMode.NATIVE_ANCHOR_GUIDED,
        VisualMode.NATIVE_QFORMER_GUIDED,
        VisualMode.QFORMER_GUIDED,
    ],
)
def test_a_guided_mode_refuses_a_record_with_no_predictions(mode):
    """The silent failure this guard exists for.

    Native manifest records carry no `pred_groups`; without this the builder
    emitted a cue-free prompt and the guided arm trained for days on prompts
    indistinguishable from the control.
    """
    with pytest.raises(ValueError, match="carries none of"):
        context_from_record(NATIVE_MANIFEST_RECORD, visual_mode=mode, qformer_token_count=32)


def test_a_study_predicted_entirely_normal_is_still_accepted():
    """Present-but-empty is a real prediction, not a missing one."""
    record = dict(
        NATIVE_MANIFEST_RECORD,
        pred_groups={"positive": [], "uncertain": [], "negative": []},
    )
    context = context_from_record(
        record, visual_mode=VisualMode.NATIVE_QFORMER_GUIDED, qformer_token_count=32
    )
    assert context.positive_findings == ()


def test_an_unguided_native_record_is_untouched():
    context = context_from_record(
        NATIVE_MANIFEST_RECORD, visual_mode=VisualMode.NATIVE_ANCHOR_ONLY
    )
    assert context.positive_findings == ()


def test_the_pipeline_mode_requires_stage_one_and_the_mhcac_prompt():
    mode = resolve_pipeline_modes("meta_cxr_native_qformer_guided")[0]
    assert mode.image_mode == "native_qformer"
    assert mode.requires_stage1, "soft tokens and cues both come from Stage 1"
    assert mode.uses_mhcac_prompt


def test_an_adapter_without_the_projector_is_incomplete(tmp_path):
    """The bridge is trained; resuming without it restarts a fresh nn.Linear."""
    import json

    def write_manifest(image_mode: str) -> None:
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "complete",
                    "image_mode": image_mode,
                }
            ),
            encoding="utf-8",
        )

    for name in ("adapter_config.json", "meta.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "trainer_state.pt").write_bytes(b"")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")

    write_manifest("native")
    assert adapter_is_complete(tmp_path, "native") is True

    write_manifest("native_qformer")
    assert adapter_is_complete(tmp_path, "native_qformer") is False, "no img_proj yet"
    (tmp_path / "img_proj.pt").write_bytes(b"")
    assert adapter_is_complete(tmp_path, "native_qformer") is True
