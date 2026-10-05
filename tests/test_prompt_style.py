"""``--prompt-style paper``: the META-CXR paper's Stage-2 prompt, no native image.

CPU-only. ``train_eval_figure9_llm_variants_200`` imports torch, transformers
and nltk at module scope, so the two prompt functions are lifted out of its
source with ``ast`` and executed on their own -- they are pure string code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from training.pipeline_modes import (
    MEDGEMMA_DIRECT,
    META_CXR_NATIVE_QFORMER_GUIDED,
    META_CXR_QFORMER,
    META_CXR_QFORMER_WITH_MHCAC_PROMPT,
    PROMPT_STYLE_FINE,
    PROMPT_STYLE_PAPER,
    PROMPT_STYLES,
    adapter_prompt_style,
    check_adapter_prompt_style,
    validate_prompt_style,
)

REPO = Path(__file__).resolve().parents[1]
FIG9 = REPO / "training" / "train_eval_figure9_llm_variants_200.py"
REFERENCE = REPO / "inference.py"

GROUPS = {
    "positive": ["Edema", "Pleural Effusion"],
    "negative": ["Pneumothorax"],
    "uncertain": ["Atelectasis"],
}


def _prompt_namespace() -> dict:
    tree = ast.parse(FIG9.read_text(encoding="utf-8"))
    wanted_funcs = {"format_findings", "build_instruction"}
    wanted_names = {"PROMPT_STYLE_FINE", "PROMPT_STYLE_PAPER", "LEGACY_PROMPT_STYLES"}
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted_funcs)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in wanted_names for t in node.targets)
        )
    ]
    assert len(nodes) == 5, "the prompt helpers moved; update this test"
    namespace: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(FIG9), "exec"), namespace)
    return namespace


def _reference_instruction() -> str:
    """The instruction body the reference ``inference.py`` shows the user."""
    tree = ast.parse(REFERENCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) and all(
            isinstance(v, ast.Constant) for v in node.values
        ):
            text = "".join(v.value for v in node.values)
            if text.startswith("Act as an expert radiologist"):
                return text
    raise AssertionError("reference instruction not found in inference.py")


def test_paper_style_is_byte_identical_to_the_reference_implementation():
    ns = _prompt_namespace()
    findings = ns["format_findings"](GROUPS)
    expected = f"Abnormality information: {findings}\n\n{_reference_instruction()}"
    assert ns["build_instruction"](GROUPS, "paper") == expected


def test_findings_are_listed_positive_negative_uncertain_like_the_reference():
    ns = _prompt_namespace()
    assert ns["format_findings"](GROUPS) == (
        "Positive findings: Edema, Pleural Effusion. Negative findings: "
        "Pneumothorax. Uncertain findings: Atelectasis"
    )
    assert ns["format_findings"]({}) == "no common findings"


def test_fine_style_is_unchanged():
    ns = _prompt_namespace()
    findings = ns["format_findings"](GROUPS)
    assert ns["build_instruction"](GROUPS, "fine") == (
        f"Abnormality information: {findings}\n\n"
        "Act as an expert radiologist. Write only the Findings section of a chest "
        "X-ray report as one concise clinical paragraph. Do not invent facts, add "
        "an Impression section, or repeat the structured findings."
    )


def test_unknown_style_raises_instead_of_falling_through_to_paper():
    ns = _prompt_namespace()
    with pytest.raises(ValueError):
        ns["build_instruction"](GROUPS, "instruction")


def test_styles_mirror_the_engine():
    assert tuple(_prompt_namespace()["LEGACY_PROMPT_STYLES"]) == PROMPT_STYLES


@pytest.mark.parametrize("mode", [
    MEDGEMMA_DIRECT, META_CXR_QFORMER, META_CXR_NATIVE_QFORMER_GUIDED,
])
def test_paper_style_refuses_every_mode_but_soft_tokens_plus_mhcac_text(mode):
    with pytest.raises(ValueError, match="meta_cxr_qformer_with_mhcac_prompt"):
        validate_prompt_style(
            PROMPT_STYLE_PAPER, [mode], has_prompt_config=False, cue_rule="argmax"
        )


@pytest.mark.parametrize("cue_rule", ["argmax", "paper_thresholds"])
def test_paper_style_accepts_the_paper_mode(cue_rule):
    validate_prompt_style(
        PROMPT_STYLE_PAPER, [META_CXR_QFORMER_WITH_MHCAC_PROMPT],
        has_prompt_config=False, cue_rule=cue_rule,
    )


def test_paper_style_refuses_a_prompt_config_and_an_empty_cue_list():
    with pytest.raises(ValueError, match="--prompt-config"):
        validate_prompt_style(
            PROMPT_STYLE_PAPER, [META_CXR_QFORMER_WITH_MHCAC_PROMPT],
            has_prompt_config=True, cue_rule="argmax",
        )
    with pytest.raises(ValueError, match="none"):
        validate_prompt_style(
            PROMPT_STYLE_PAPER, [META_CXR_QFORMER_WITH_MHCAC_PROMPT],
            has_prompt_config=False, cue_rule="none",
        )


def test_fine_style_is_accepted_everywhere():
    for mode in (MEDGEMMA_DIRECT, META_CXR_QFORMER, META_CXR_QFORMER_WITH_MHCAC_PROMPT):
        validate_prompt_style(PROMPT_STYLE_FINE, [mode], has_prompt_config=True, cue_rule="none")


def test_old_legacy_manifests_count_as_fine():
    old = {"prompt": {"builder": "legacy", "version": "legacy_build_instruction"}}
    assert adapter_prompt_style(old) == PROMPT_STYLE_FINE
    check_adapter_prompt_style(old, PROMPT_STYLE_FINE)
    with pytest.raises(ValueError, match="trained with --prompt-style fine"):
        check_adapter_prompt_style(old, PROMPT_STYLE_PAPER)


def test_a_paper_adapter_is_not_reused_under_fine():
    paper = {"prompt": {"builder": "legacy", "version": "paper_build_instruction",
                        "prompt_style": "paper"}}
    check_adapter_prompt_style(paper, PROMPT_STYLE_PAPER)
    with pytest.raises(ValueError, match="trained with --prompt-style paper"):
        check_adapter_prompt_style(paper, PROMPT_STYLE_FINE)


def test_a_v2_builder_adapter_refuses_the_paper_style():
    v2 = {"prompt": {"builder": "stage2.prompts.PromptBuilder", "version": "x"}}
    assert adapter_prompt_style(v2) is None
    check_adapter_prompt_style(v2, PROMPT_STYLE_FINE)
    with pytest.raises(ValueError, match="--prompt-config"):
        check_adapter_prompt_style(v2, PROMPT_STYLE_PAPER)


def test_paper_style_defaults_to_the_papers_lora_size():
    from training.pipeline_modes import resolve_lora_size

    assert resolve_lora_size(PROMPT_STYLE_PAPER, None, None) == (8, 16)
    assert resolve_lora_size(PROMPT_STYLE_FINE, None, None) == (16, 32)


def test_explicit_lora_flags_always_win():
    from training.pipeline_modes import resolve_lora_size

    assert resolve_lora_size(PROMPT_STYLE_PAPER, 16, None) == (16, 16)
    assert resolve_lora_size(PROMPT_STYLE_FINE, None, 64) == (16, 64)
    with pytest.raises(ValueError):
        resolve_lora_size(PROMPT_STYLE_FINE, 0, None)
