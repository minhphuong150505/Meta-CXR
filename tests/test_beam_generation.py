"""Beam search in Stage-2 generation (2026-10-11): opt-in, greedy stays the default."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import generate_stage2_reports as gen  # noqa: E402


def _mode(name):
    from training.pipeline_modes import resolve_pipeline_modes

    (mode,) = resolve_pipeline_modes(name)
    return mode


def test_cli_defaults_to_greedy(tmp_path):
    args = gen.parse_args(["--output-dir", str(tmp_path)])
    assert args.num_beams == 1 and args.length_penalty == 0.0


def test_cli_refuses_length_penalty_without_beams(tmp_path):
    args = gen.parse_args(["--output-dir", str(tmp_path), "--length-penalty", "2.0"])
    with pytest.raises(SystemExit, match="only applies to beam search"):
        gen.validate_invocation(args, _mode("medgemma_direct"))


def test_cli_refuses_zero_beams(tmp_path):
    args = gen.parse_args(["--output-dir", str(tmp_path), "--num-beams", "0"])
    with pytest.raises(SystemExit, match="must be >= 1"):
        gen.validate_invocation(args, _mode("medgemma_direct"))


def _llm(captured, image_mode="native"):
    torch = pytest.importorskip("torch")
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")

    class Decoder:
        generation_config = SimpleNamespace(eos_token_id=[1, 106])

        def eval(self):
            pass

        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.tensor([kwargs["input_ids"][0].tolist() + [20, 106]])

        def get_input_embeddings(self):
            return "embedding"

        def set_input_embeddings(self, value):
            captured.setdefault("set_embeddings", []).append(value)

    llm = object.__new__(fig9.VariantLLM)
    llm.model = Decoder()
    llm.img_proj = None
    llm.image_mode = image_mode
    llm.device = torch.device("cpu")
    llm.dtype = torch.float32
    llm.tokenizer = SimpleNamespace(
        pad_token_id=0, eos_token_id=1,
        decode=lambda ids, **kw: " ".join(str(x) for x in ids.tolist() if x not in (1, 106)))
    llm._native_chat_inputs = lambda *a, **k: {
        "input_ids": torch.tensor([[5, 6]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}
    return llm, fig9, torch


def test_greedy_is_unchanged_by_default():
    captured = {}
    llm, _, _ = _llm(captured)
    assert llm.generate({}, "fine", 20) == "20"
    assert captured["num_beams"] == 1 and "length_penalty" not in captured


def test_beam_passes_width_and_length_penalty():
    captured = {}
    llm, _, _ = _llm(captured)
    llm.generate({}, "fine", 20, num_beams=4, length_penalty=2.0)
    assert captured["num_beams"] == 4 and captured["length_penalty"] == 2.0


def test_beam_repeats_soft_tokens_per_beam(monkeypatch):
    captured = {}
    llm, fig9, torch = _llm(captured, image_mode="qformer")
    llm.img_token_id = 7
    llm.finding_encoder = None
    llm.finding_token_id = None
    llm.img_proj = torch.nn.Identity()
    llm._chat_texts = lambda record, style: ("prompt", "prompt target")

    class Tok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, *a, **k):
            return {"input_ids": torch.tensor([[5, 6]]),
                    "attention_mask": torch.ones((1, 2), dtype=torch.long)}

        def decode(self, ids, **k):
            return " ".join(str(x) for x in ids.tolist() if x not in (1, 106))

    llm.tokenizer = Tok()
    seen = {}

    class Wrapper:
        def __init__(self, base, token_id, projected, count):
            seen["rows"] = projected.shape[0]

    monkeypatch.setattr(fig9, "SoftTokenEmbeddingWrapper", Wrapper)
    llm.generate({"qformer_embs": torch.zeros(32, 8)}, "paper", 20, num_beams=4)
    assert seen["rows"] == 4
