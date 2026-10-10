"""``--cue-rule cutpoints``: the Stage-1 headline rule as Stage-2 prompt cues (2026-10-10).

Per finding, Negative below t1, Uncertain in [t1, t2), Positive at or above t2
on p_pos / (p_pos + p_neg). The fig9 tests skip on a CPU box without
transformers; the drift they guard is caught on the training host.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import generate_stage2_reports as gen  # noqa: E402
from training.evaluation.threshold_calibration import CutpointFile  # noqa: E402
from training.run_context import Stage1Context  # noqa: E402

REPORTABLE = (
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
)


def test_empty_cutpoints_leave_the_fingerprint_unchanged():
    # Every Stage-1 record cache is keyed on this payload; the new field must not
    # move it for any run that does not use the rule.
    payload = Stage1Context(run_name="r").fingerprint_payload()
    assert set(payload) == {"run_name", "config_path", "checkpoint_path", "thresholds"}
    with_cut = Stage1Context(run_name="r", cutpoints={"Edema": (0.3, 0.6)})
    assert with_cut.fingerprint_payload()["cutpoints"] == {"Edema": [0.3, 0.6]}
    with pytest.raises(TypeError):
        with_cut.cutpoints["Edema"] = (0.1, 0.2)


def _mode(name):
    from training.pipeline_modes import resolve_pipeline_modes

    (mode,) = resolve_pipeline_modes(name)
    return mode


def _args(tmp_path, *extra):
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "img_proj.pt").write_bytes(b"")
    return gen.parse_args([
        "--output-dir", str(tmp_path), "--checkpoint-root", str(tmp_path),
        "--pipeline-mode", "meta_cxr_qformer_with_mhcac_prompt", "--prompt-style", "paper",
        "--adapter", str(adapter), "--cue-rule", "cutpoints", *extra,
    ])


def test_generation_cli_needs_a_cutpoint_file(tmp_path):
    with pytest.raises(SystemExit, match="cutpoints needs --threshold-path"):
        gen.validate_invocation(_args(tmp_path), _mode("meta_cxr_qformer_with_mhcac_prompt"))


def test_generation_cli_accepts_cutpoints_with_the_paper_prompt(tmp_path):
    path = CutpointFile({n: (0.4, 0.6) for n in REPORTABLE}, {"split": "val"}).save(
        tmp_path / "c.json")
    gen.validate_invocation(_args(tmp_path, "--threshold-path", str(path)),
                            _mode("meta_cxr_qformer_with_mhcac_prompt"))


def _fig9():
    return pytest.importorskip("training.train_eval_figure9_llm_variants_200")


def test_cutpoints_place_every_finding_by_severity():
    torch = pytest.importorskip("torch")
    fig9 = _fig9()
    cut = {n: (0.4, 0.6) for n in REPORTABLE}
    probs = torch.tensor([[0.8, 0.1, 0.1]] * 14)          # s = 0.11 -> negative
    e, c = fig9.ABNORMALITIES_14.index("Edema"), fig9.ABNORMALITIES_14.index("Cardiomegaly")
    probs[e] = torch.tensor([0.25, 0.25, 0.5])             # s = 0.50 -> uncertain
    probs[c] = torch.tensor([0.1, 0.3, 0.6])               # s = 0.75 -> positive
    groups = fig9.classify_with_thresholds(
        Stage1Context(run_name="s", cutpoints=cut), torch.log(probs), cue_rule="cutpoints")
    assert groups["uncertain"] == ["Edema"] and groups["positive"] == ["Cardiomegaly"]
    assert len(groups["negative"]) == 11 and "No Finding" not in groups["negative"]
    record = fig9.with_cue_state({"pred_groups": groups}, "cutpoints")
    assert record["cue_state"] == "predicted"


def test_cutpoint_rule_matches_the_stage1_evaluator():
    torch = pytest.importorskip("torch")
    fig9 = _fig9()
    import numpy as np

    from training.evaluation.threshold_calibration import apply_cutpoints

    gen_ = torch.Generator().manual_seed(0)
    logits = torch.randn(14, 3, generator=gen_)
    cut = {n: (0.35, 0.55) for n in fig9.ABNORMALITIES_14}
    groups = fig9.classify_with_thresholds(
        Stage1Context(run_name="s", cutpoints=cut), logits, cue_rule="cutpoints")
    decided = apply_cutpoints(torch.softmax(logits, -1).numpy()[None], cut,
                              tuple(fig9.ABNORMALITIES_14))[0]
    names = {0: "negative", 1: "positive", 2: "uncertain"}
    for i, name in enumerate(fig9.ABNORMALITIES_14):
        if name != "No Finding":
            assert name in groups[names[int(decided[i])]]
    assert not math.isnan(float(np.sum(decided)))


def test_cutpoint_rule_refuses_missing_or_test_fitted_files(tmp_path):
    torch = pytest.importorskip("torch")
    fig9 = _fig9()
    with pytest.raises(ValueError, match="cutpoints needs"):
        fig9.classify_with_thresholds(Stage1Context(run_name="s"), torch.zeros(14, 3),
                                      cue_rule="cutpoints")
    path = CutpointFile({n: (0.4, 0.6) for n in REPORTABLE}, {"split": "test"}).save(
        tmp_path / "t.json")
    with pytest.raises(ValueError, match="validation"):
        fig9.load_cue_cutpoints(path)
