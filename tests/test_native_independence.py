"""``medgemma_direct`` must not import the Stage-1 stack.

The guarantee under test is architectural, not behavioural: importing the
Stage-2 entrypoint must not pull in LAVIS, the three META-CXR vision encoders,
the Q-Former or MHCAC. Before this was enforced, ``run_medgemma_qlora.py``
imported the Figure-9 module at module scope, which imported
``model.lavis.tasks`` and ``MIMIC_CXR_Dataset`` -- so a native run could not
start on a machine without the entire Stage-1 stack installed, contradicting
its own docstring.

This runs on CPU with no model weights and no MIMIC data. MedGemma-side
dependencies (transformers, peft, the NLG metric packages) are stubbed because
they are absent from the CPU test environment; the Stage-1 packages are
deliberately *not* stubbed, so any import of them raises.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Packages that native MedGemma legitimately needs. Absent from the CPU test
# environment, so stub them; the test is about Stage-1 coupling, not these.
STUBBED_ROOTS = (
    "transformers",
    "peft",
    "nltk",
    "bert_score",
    "pycocoevalcap",
)

# Packages that a native run must never import. Not stubbed -- importing any of
# them under the guard below is the failure this test exists to catch.
FORBIDDEN_ROOTS = (
    "model.lavis",
    "mhcac",
    "biovil_t",
    "vision_encoders",
)


class Stage1ImportGuard:
    """A meta-path finder that raises when a forbidden module is imported."""

    def __init__(self, forbidden: tuple[str, ...]):
        self.forbidden = forbidden
        self.violations: list[str] = []

    def find_module(self, fullname, path=None):  # pragma: no cover - legacy API
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path=None, target=None):
        for root in self.forbidden:
            if fullname == root or fullname.startswith(root + "."):
                self.violations.append(fullname)
                raise AssertionError(
                    f"medgemma_direct import path pulled in Stage-1 module {fullname!r}. "
                    "Move the import into training/stage1/lavis_loader.py and call it "
                    "lazily from the Stage-1 branch only."
                )
        return None


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # mark as a package so submodule imports resolve
    return mod


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal stand-ins for the MedGemma-side imports the module performs."""
    transformers = _module("transformers")
    for name in (
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoProcessor",
        "AutoTokenizer",
        "BitsAndBytesConfig",
    ):
        setattr(transformers, name, type(name, (), {}))
    transformers.get_cosine_schedule_with_warmup = lambda *a, **k: None

    peft = _module("peft")
    for name in ("LoraConfig", "PeftModel"):
        setattr(peft, name, type(name, (), {}))
    peft.get_peft_model = lambda *a, **k: None
    peft.prepare_model_for_kbit_training = lambda *a, **k: None

    nltk = _module("nltk")
    nltk.data = types.SimpleNamespace(find=lambda *a, **k: None)
    nltk.download = lambda *a, **k: None
    nltk_translate = _module("nltk.translate")
    bleu = _module("nltk.translate.bleu_score")
    bleu.corpus_bleu = lambda *a, **k: 0.0
    bleu.SmoothingFunction = lambda: types.SimpleNamespace(method1=lambda *a, **k: 0.0)
    meteor = _module("nltk.translate.meteor_score")
    meteor.meteor_score = lambda *a, **k: 0.0

    bert_score = _module("bert_score")
    bert_score.score = lambda *a, **k: (None, None, None)

    cider_mod = _module("pycocoevalcap.cider.cider")
    cider_mod.Cider = type("Cider", (), {})
    rouge_mod = _module("pycocoevalcap.rouge.rouge")
    rouge_mod.Rouge = type("Rouge", (), {})

    stubs = {
        "transformers": transformers,
        "peft": peft,
        "nltk": nltk,
        "nltk.translate": nltk_translate,
        "nltk.translate.bleu_score": bleu,
        "nltk.translate.meteor_score": meteor,
        "bert_score": bert_score,
        "pycocoevalcap": _module("pycocoevalcap"),
        "pycocoevalcap.cider": _module("pycocoevalcap.cider"),
        "pycocoevalcap.cider.cider": cider_mod,
        "pycocoevalcap.rouge": _module("pycocoevalcap.rouge"),
        "pycocoevalcap.rouge.rouge": rouge_mod,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def native_import_env(monkeypatch: pytest.MonkeyPatch):
    """Import environment with MedGemma deps stubbed and Stage-1 imports fatal.

    ``tests/conftest.py`` pre-registers ``model`` and ``model.lavis`` as
    path-only packages for other tests. Drop them here so the guard sees a real
    import attempt rather than a cache hit.
    """
    for name in list(sys.modules):
        if name.startswith(FORBIDDEN_ROOTS) or name in (
            "train_eval_figure9_llm_variants_200",
            "run_medgemma_qlora",
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)

    _install_stubs(monkeypatch)
    for path in (REPO_ROOT, REPO_ROOT / "training"):
        monkeypatch.syspath_prepend(str(path))

    guard = Stage1ImportGuard(FORBIDDEN_ROOTS)
    monkeypatch.setattr(sys, "meta_path", [guard, *sys.meta_path])
    return guard


def test_guard_actually_fires(native_import_env):
    """Positive control: the guard must reject a Stage-1 import.

    Without this, a guard that silently matched nothing would let every other
    assertion in this file pass vacuously.
    """
    with pytest.raises(AssertionError, match="Stage-1 module"):
        __import__("mhcac.mhcac_12")
    # The guard trips on the parent package, before the submodule is reached.
    assert native_import_env.violations == ["mhcac"]


def test_figure9_module_imports_without_stage1(native_import_env):
    import train_eval_figure9_llm_variants_200 as fig9

    assert native_import_env.violations == []
    assert fig9.MEDGEMMA_MODEL_ID == "google/medgemma-1.5-4b-it"


def test_native_entrypoint_imports_without_stage1(native_import_env):
    import run_medgemma_qlora

    assert native_import_env.violations == []
    assert run_medgemma_qlora.DEFAULT_PIPELINE_MODE == "medgemma_direct"


def test_native_record_builder_needs_no_stage1(native_import_env):
    """The medgemma_direct data path reads split CSVs, not ReportDataset."""
    import run_medgemma_qlora

    assert hasattr(run_medgemma_qlora, "build_native_records")
    assert native_import_env.violations == []


def test_stage1_records_no_longer_accepts_a_native_flag(native_import_env):
    """The dead ``include_stage1_features=False`` branch must stay removed.

    That branch still constructed a LAVIS ``Config`` and a ``MIMIC_CXR_Dataset``
    loader, so a caller who used it would have reintroduced the coupling while
    believing they had opted out of Stage-1.
    """
    import inspect

    import train_eval_figure9_llm_variants_200 as fig9

    params = inspect.signature(fig9.build_stage1_records).parameters
    assert "include_stage1_features" not in params


def test_lavis_loader_is_the_only_stage1_import_site():
    """No Stage-2 module may import LAVIS at module scope."""
    stage2_files = [
        REPO_ROOT / "training" / "run_medgemma_qlora.py",
        REPO_ROOT / "training" / "train_eval_figure9_llm_variants_200.py",
        REPO_ROOT / "training" / "stage2_utils.py",
        REPO_ROOT / "training" / "dataio" / "manifest.py",
    ]
    for path in stage2_files:
        source = path.read_text(encoding="utf-8")
        assert "model.lavis" not in source.replace(
            "training/stage1/lavis_loader.py", ""
        ), f"{path.name} references model.lavis outside the lazy Stage-1 loader"
