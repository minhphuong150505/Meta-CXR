"""Selective cues may abstain, but may never silently fall back to noisy labels."""

import json

import numpy as np
import pytest

from scripts.calibrate_cue_precision import fit_selective_thresholds, main


def test_infeasible_label_is_disabled_even_at_saturated_score():
    thresholds, details = fit_selective_thresholds(
        np.ones((30, 1)), np.zeros((30, 1)), ["Edema"]
    )
    assert thresholds["Edema"]["positive_enabled"] == 0
    assert details["Edema"]["precision"] is None
    assert details["Edema"]["reason"] == "no_feasible_threshold"


def test_one_lucky_prediction_cannot_meet_support_floor():
    scores = np.array([[1.0]] + [[0.1]] * 29)
    labels = np.array([[1]] + [[0]] * 29)
    thresholds, _ = fit_selective_thresholds(scores, labels, ["Edema"])
    assert thresholds["Edema"]["positive_enabled"] == 0


def test_maximum_recall_subject_to_precision_and_support():
    scores = np.array([[0.9]] * 20 + [[0.8]] * 10 + [[0.1]] * 30)
    labels = np.array([[1]] * 20 + [[1]] * 5 + [[0]] * 35)
    thresholds, details = fit_selective_thresholds(scores, labels, ["Edema"])
    assert thresholds["Edema"] == {"positive_enabled": 1, "marginal_positive": 0.8}
    assert details["Edema"]["n_predicted"] == 30
    assert details["Edema"]["recall"] == 1.0


def test_tied_scores_cannot_be_split_to_manufacture_precision():
    _, details = fit_selective_thresholds(
        np.ones((30, 1)), np.array([[1]] * 15 + [[0]] * 15), ["Edema"]
    )
    assert not details["Edema"]["enabled"]


@pytest.mark.parametrize("missing", [-1, -100])
def test_blanks_and_uncertainty_are_not_positive_report_labels(missing):
    _, details = fit_selective_thresholds(
        np.ones((30, 1)), np.array([[1]] * 10 + [[missing]] * 10 + [[2]] * 10), ["Edema"]
    )
    assert details["Edema"]["n_positive"] == 10
    assert not details["Edema"]["enabled"]


def test_no_finding_never_becomes_a_positive_cue():
    thresholds, _ = fit_selective_thresholds(np.ones((30, 1)), np.ones((30, 1)), ["No Finding"])
    assert thresholds == {}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_scores_raise(bad):
    with pytest.raises(ValueError, match="finite"):
        fit_selective_thresholds([[bad]], [[1]], ["Edema"])


def test_cli_refuses_test_fitting(tmp_path):
    with pytest.raises(SystemExit):
        main(["--predictions", "unused.npz", "--split", "test", "--output", str(tmp_path / "out.json")])


def test_cli_records_provenance_and_refuses_overwrite(tmp_path):
    source = tmp_path / "synthetic.npz"
    np.savez(source, probabilities=np.tile([0., 1., 0.], (30, 1, 1)),
             mention_probabilities=np.ones((30, 1)), labels=np.ones((30, 1)),
             pathology_names=np.array(["Edema"]), split=np.array("val"))
    output = tmp_path / "thresholds.json"
    argv = ["--predictions", str(source), "--split", "val", "--output", str(output)]
    main(argv)
    assert json.loads(output.read_text())["Edema"]["positive_enabled"] == 1
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert metadata["fit_split"] == "val" and metadata["min_predicted"] == 20
    with pytest.raises(SystemExit):
        main(argv)


def selective_context(fig9, enabled=()):
    from training.run_context import Stage1Context

    return Stage1Context(run_name="synthetic", thresholds={
        name: {"positive_enabled": int(name in enabled), "marginal_positive": 0.5}
        for name in fig9.ABNORMALITIES_14 if name != "No Finding"
    })


def test_emission_honours_disabled_label_even_when_its_score_is_one():
    torch = pytest.importorskip("torch")
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    logits = torch.full((14, 3), -100.0)
    logits[:, 1] = 100.0
    groups = fig9.classify_with_thresholds(
        selective_context(fig9, ("Edema",)), logits, torch.full((14,), 100.0),
        cue_rule="marginal_positive",
    )
    assert groups == {"positive": ["Edema"], "negative": [], "uncertain": []}


def test_incomplete_selective_artifact_raises_before_cache_access(monkeypatch, tmp_path):
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    from training.run_context import Stage1Context

    context = Stage1Context(run_name="synthetic", thresholds={"Edema": {"positive_enabled": 0}})
    monkeypatch.setattr(fig9, "stage1_cohort_fingerprint", lambda *a: pytest.fail("must validate before cache access"))
    with pytest.raises(ValueError, match="selective thresholds need"):
        fig9.build_stage1_records(context, tmp_path, tmp_path, "val", None, 0, cue_rule="marginal_positive")


def test_selective_artifact_cannot_silently_use_conditional_rule():
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    with pytest.raises(ValueError, match="require cue_rule=marginal_positive"):
        fig9.validate_selective_thresholds(selective_context(fig9), "conditional_positive")


def test_fractional_enable_flag_is_rejected(tmp_path):
    fig9 = pytest.importorskip("training.train_eval_figure9_llm_variants_200")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"Edema": {"positive_enabled": 0.5}}))
    with pytest.raises(ValueError, match="positive_enabled must"):
        fig9.load_thresholds(path)
