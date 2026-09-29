"""Three classes, never binary (D-023, 2026-09-29).

META-CXR classifies every finding as Negative / Positive / Uncertain and is
evaluated that way. Between 2026-08 and 2026-09 an AI assistant, unverified,
turned this repository's task into "present / not present": a binary mention
gate, a mention-conditioned objective, uncertain-folding policies and a
``study_presence`` evaluation framing. The user removed all of it and asked
that it never come back. This file is the tripwire:

1. the retired config keys are refused (``pretraining/retired_keys.py``);
2. old checkpoints still load, minus the retired heads;
3. no executable Python and no shipped YAML names a retired identifier.

If one of these fails because you are adding a binary framing back: stop and
ask the user. The two binary evaluations the paper itself uses (Table 3 CE,
Table 4 CheXpert cross-domain) live in ``training/evaluation/paper_protocol.py``
and need nothing on the list below.
"""

from __future__ import annotations

import ast
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pretraining.retired_keys import (  # noqa: E402
    RETIRED_MODEL_KEYS,
    RETIRED_RUN_KEYS,
    RetiredConfigKey,
    drop_retired_state,
    reject_retired_model_keys,
    reject_retired_run_keys,
)

#: Identifiers of the retired binary framing. Matched as substrings of names,
#: attribute names, keyword arguments and non-docstring string literals.
FORBIDDEN = (
    "study_presence",
    "masked_polarity",
    "marginal_presence",
    "conditional_positive",
    "marginal_positive",
    "mention_gated",
    "mention_logits",
    "mention_heads",
    "mention_targets",
    "mention_probabilities",
    "mention_gate",
    "mention_conditioned",
    "MentionGate",
    "MentionConditioned",
    "lambda_gate",
    "gate_class_weights",
    "uncertain_policy",
    "ignore_uncertain",
    "uncertain_as_positive",
    "uncertain_as_negative",
    "positive_macro",
    "f1_positive_macro",
    "label_framing",
    "include_meta_labels",
    "return_mention",
    "positive_enabled",
    "binarize_labels",
)

#: Files allowed to spell the retired names: the module that refuses them and
#: this test.
ALLOWED = {
    REPO / "pretraining/retired_keys.py",
    REPO / "tests/test_three_class_only.py",
}

SOURCE_DIRS = ("mhcac", "model", "pretraining", "training", "scripts", "stage2",
               "vision_encoders", "safety", "runtime", "preporcessing")


def _docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(
                getattr(body[0], "value", None), ast.Constant
            ) and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _code_tokens(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.lineno, node.id
        elif isinstance(node, ast.Attribute):
            yield node.lineno, node.attr
        elif isinstance(node, ast.arg):
            yield node.lineno, node.arg
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.value.lineno, node.arg
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            yield node.lineno, node.name
        elif isinstance(node, ast.alias):
            yield 0, node.asname or node.name
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            yield node.lineno, node.value


def _python_sources():
    for directory in SOURCE_DIRS:
        for path in sorted((REPO / directory).rglob("*.py")):
            if path in ALLOWED or "__pycache__" in path.parts:
                continue
            # Legacy MHCAC variants and the vendored upstream stay untouched.
            if path.parent.name == "mhcac" and path.name != "mhcac_12.py" and path.name.startswith("mhcac_"):
                continue
            yield path
    for path in sorted(REPO.glob("*.py")):
        yield path


def test_no_executable_python_names_a_retired_binary_identifier():
    hits = []
    for path in _python_sources():
        for line, token in _code_tokens(path):
            for word in FORBIDDEN:
                if word in token:
                    hits.append(f"{path.relative_to(REPO)}:{line}: {word!r} in {token[:60]!r}")
    assert not hits, "retired binary framing found:\n" + "\n".join(hits[:40])


def _yaml_items(node, trail=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield f"{trail}.{key}", str(key)
            yield from _yaml_items(value, f"{trail}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _yaml_items(value, f"{trail}[{index}]")
    elif isinstance(node, str):
        yield trail, node


def test_no_shipped_yaml_names_a_retired_binary_key_or_value():
    yaml = pytest.importorskip("yaml")
    hits = []
    for path in sorted(REPO.rglob("*.yaml")):
        if any(part in {"struct", "docs", ".git"} for part in path.parts):
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for trail, text in _yaml_items(payload):
            for word in FORBIDDEN:
                if word in text:
                    hits.append(f"{path.relative_to(REPO)} {trail}: {word}")
    assert not hits, "\n".join(hits)


def test_the_production_config_passes_the_guard_including_every_phase():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((REPO / "pretraining/configs/mimic_cxr_full.yaml").read_text())
    reject_retired_model_keys(cfg["model"])
    reject_retired_run_keys(cfg["run"])
    for block in cfg["run"].get("phases", {}).values():
        reject_retired_model_keys(block.get("model"))
        reject_retired_run_keys(block.get("run"))
    assert cfg["model"]["mhcac"]["blank_label_policy"] == "negative"
    assert cfg["model"]["mhcac"]["excluded_labels"] == []


@pytest.mark.parametrize(("section", "key"), RETIRED_MODEL_KEYS)
def test_every_retired_model_key_is_refused(section, key):
    with pytest.raises(RetiredConfigKey, match="D-023"):
        reject_retired_model_keys({section: {key: 0.0}})


@pytest.mark.parametrize("key", RETIRED_RUN_KEYS)
def test_every_retired_run_key_is_refused(key):
    with pytest.raises(RetiredConfigKey, match="three-class"):
        reject_retired_run_keys({key: False})


def test_old_checkpoints_drop_only_the_retired_heads():
    state = OrderedDict(
        [
            ("mhcac.classifiers.0.0.weight", 1),
            ("mhcac.mention_heads.3.weight", 2),
            ("gate_loss_fn.pos_weight", 3),
            ("mention_conditioned_loss_fn.pos_weight", 4),
            ("Qformer.bert.embeddings.word_embeddings.weight", 5),
        ]
    )
    state._metadata = {"": {"version": 1}}
    kept = drop_retired_state(state)
    assert list(kept) == [
        "mhcac.classifiers.0.0.weight",
        "Qformer.bert.embeddings.word_embeddings.weight",
    ]
    assert kept._metadata == {"": {"version": 1}}
