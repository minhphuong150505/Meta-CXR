"""Stage-2's Stage-1 record pass must forward every input forward_image reads.

CPU-only: the engine module imports torch/transformers/nltk, so the key set and
``forward_image``'s own-input list are read from source with ``ast``.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIG9 = REPO / "training" / "train_eval_figure9_llm_variants_200.py"
QFORMER = REPO / "model" / "lavis" / "models" / "blip2_models" / "blip2_qformer.py"


def _record_keys() -> set[str]:
    tree = ast.parse(FIG9.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "STAGE1_IMAGE_INPUT_KEYS" for t in node.targets
        ):
            call = node.value
            assert isinstance(call, ast.Call) and isinstance(call.args[0], ast.Set)
            return {elt.value for elt in call.args[0].elts}
    raise AssertionError("STAGE1_IMAGE_INPUT_KEYS not found")


def _forward_image_own_inputs() -> set[str]:
    tree = ast.parse(QFORMER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward_image":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Tuple) and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str) for e in sub.elts
                ):
                    names = {e.value for e in sub.elts}
                    if "pubmedclip_image" in names:
                        return names
    raise AssertionError("forward_image own-input tuple not found")


def test_own_preprocessing_inputs_reach_forward_image():
    keys = _record_keys()
    missing = _forward_image_own_inputs() - keys
    assert not missing, f"build_stage1_records drops {sorted(missing)}"


def test_the_record_pass_uses_the_shared_key_set():
    source = FIG9.read_text(encoding="utf-8")
    assert "image_input_keys = STAGE1_IMAGE_INPUT_KEYS" in source
