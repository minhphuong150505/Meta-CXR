"""The report ends at a model's chat terminator as well as tokenizer EOS."""
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("configured, expected", [([1, 106], [1, 106]), (106, 106), (None, 1)])
def test_generate_preserves_model_stop_tokens(configured, expected):
    from training.train_eval_figure9_llm_variants_200 import VariantLLM

    class Decoder:
        generation_config = SimpleNamespace(eos_token_id=configured)

        def eval(self):
            pass

        def generate(self, **kwargs):
            assert kwargs["eos_token_id"] == expected
            # A deterministic synthetic stream ends its first answer at 106,
            # then starts unrelated text if the caller dropped that stop ID.
            stops = kwargs["eos_token_id"]
            stops = stops if isinstance(stops, list) else [stops]
            output = kwargs["input_ids"][0].tolist()
            for token in [20, 106, 30, 1]:
                output.append(token)
                if token in stops:
                    break
            return torch.tensor([output])

    llm = object.__new__(VariantLLM)
    llm.model = Decoder()
    llm.img_proj = None
    llm.image_mode = "native"
    llm.device = torch.device("cpu")
    llm.dtype = torch.float32
    llm.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1,
                                    decode=lambda ids, **kwargs: " ".join(str(x) for x in ids.tolist() if x not in (1, 106)))
    llm._native_chat_inputs = lambda *args, **kwargs: {"input_ids": torch.tensor([[5, 6]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}
    actual = llm.generate({}, "fine", 20)
    assert actual == ("20" if configured is not None else "20 30")


def test_eval_cache_invalidates_when_stop_ids_or_prompt_change(monkeypatch, tmp_path):
    from dataclasses import replace
    from pathlib import Path

    from stage2.prompts import load_prompt_config
    from training import train_eval_figure9_llm_variants_200 as fig9
    from training.run_context import Stage1Context

    llm = object.__new__(fig9.VariantLLM)
    llm.adapter = None
    llm.image_mode = "native"
    llm.model_id = "synthetic-model"
    llm.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=1))
    llm.tokenizer = SimpleNamespace(eos_token_id=1)
    llm.prompt_config = load_prompt_config(Path(__file__).resolve().parents[1] / "configs/experiment_native_anchor_only.yaml")
    calls = []

    def generate(*args):
        calls.append(1)
        return "Synthetic generated text."

    llm.generate = generate
    monkeypatch.setattr(fig9, "compute_sectioned_nlg", lambda *args: {})
    records = [{"sample_key": "synthetic-example", "ref": "Synthetic reference text."}]

    def evaluate():
        return fig9.evaluate_variant("medgemma", "fine", llm, records, tmp_path, 160, "fine", context=Stage1Context(run_name="synthetic"))

    first = evaluate()
    assert evaluate()["EvalId"] == first["EvalId"] and len(calls) == 1
    llm.model.generation_config.eos_token_id = [1, 106]
    second = evaluate()
    assert second["EvalId"] != first["EvalId"] and len(calls) == 2
    llm.prompt_config = replace(llm.prompt_config, max_sentences=llm.prompt_config.max_sentences + 1)
    third = evaluate()
    assert third["EvalId"] != second["EvalId"] and len(calls) == 3
    monkeypatch.setattr(fig9, "_prompt_template_hash", lambda *args: "synthetic-new-template")
    fourth = evaluate()
    assert fourth["EvalId"] != third["EvalId"] and len(calls) == 4
