"""An evaluate-only run must never score randomly initialised weights.

This is a source-text guard rather than a behavioural test because importing
``runner_base`` drags in torchvision and the whole GPU stack, which the CPU
suite deliberately cannot do. It is still worth pinning: on 2026-08-20 an
evaluate-only pass ran to completion in 108 s, wrote both prediction files and
reported no error, while the model held its random initialisation --
``eval_epoch`` reloads ``checkpoint_best`` only when ``cur_epoch == "best"``,
and the evaluate-only path passes ``"provided"``. Nothing in the output said so;
the giveaway was that the mention gate came out near-constant.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_RUNNER = Path(__file__).resolve().parents[1] / "model" / "lavis" / "runners" / "runner_base.py"


@pytest.fixture(scope="module")
def source() -> str:
    return _RUNNER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(source: str) -> ast.Module:
    return ast.parse(source)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in runner_base.py")


def test_loader_exists(tree: ast.Module) -> None:
    _function(tree, "_load_eval_weights")


def test_evaluate_only_branch_loads_weights_before_validating(tree: ast.Module) -> None:
    """Inside train(), not setup_output_dir() -- both branch on evaluate_only."""
    train = _function(tree, "train")
    called: list[str] = []
    for node in ast.walk(train):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Attribute)
            and test.attr == "evaluate_only"
            and isinstance(test.value, ast.Name)
            and test.value.id == "self"
        ):
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                if call.func.attr in {"_load_eval_weights", "validate"}:
                    called.append(call.func.attr)
        if "validate" in called:
            break

    assert called[:2] == ["_load_eval_weights", "validate"], (
        "the evaluate_only branch of train() must load weights and then "
        f"validate; found calls {called!r}. Without the load it scores "
        "whatever the model was built with."
    )


def test_missing_weights_are_refused_not_warned(tree: ast.Module) -> None:
    fn = _function(tree, "_load_eval_weights")
    raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
    assert raises, (
        "_load_eval_weights must fail closed when no weight source exists -- a "
        "warning is what let a random-init evaluation through once already"
    )


def test_eval_load_does_not_touch_optimizer_state(tree: ast.Module) -> None:
    """Nothing is trained, so restoring optimizer/scaler/epoch is out of scope.

    It is also actively unhelpful: ``checkpoint_best.pth`` omits optimizer state
    on purpose, so a resume-shaped load logs a warning that reads like a fault.
    """
    fn = _function(tree, "_load_eval_weights")
    statements = fn.body[1:] if ast.get_docstring(fn) else fn.body
    body = "\n".join(ast.dump(node) for node in statements)
    for forbidden in ("scaler", "start_epoch"):
        assert forbidden not in body, f"_load_eval_weights should not restore {forbidden}"
