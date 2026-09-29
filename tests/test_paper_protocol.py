"""Paper Table 4 (CheXpert cross-domain, Eq. 21) and Table 3 (Clinical Efficacy)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.evaluation.classification_metrics import PAPER_FIVE_FINDINGS  # noqa: E402
from training.evaluation.paper_protocol import (  # noqa: E402
    CHEXPERT_14,
    chexpert_crossdomain,
    clinical_efficacy,
)
from training.evaluation.schemas import ClassificationPredictions  # noqa: E402


def test_eq21_drops_the_uncertain_mass():
    names = list(PAPER_FIVE_FINDINGS)
    # One study, Edema: p0 .2, p1 .2, p2 .6 -> p_final .5 exactly.
    probs = np.full((2, 5, 3), 1 / 3)
    probs[:, names.index("Edema")] = [[0.2, 0.2, 0.6], [0.6, 0.3, 0.1]]
    labels = np.zeros((2, 5), dtype=int)
    labels[0, names.index("Edema")] = 1
    out = chexpert_crossdomain(probs, labels, names, threshold=0.5)
    edema = out["per_finding"]["Edema"]
    # study 0: p_final 0.5 (>= 0.5 -> predicted 1, true 1); study 1: 1/3 (-> 0, true 0)
    assert edema["f1"] == pytest.approx(1.0)
    assert edema["auc"] == pytest.approx(1.0)
    assert out["threshold"] == 0.5


def test_missing_cells_are_skipped_and_one_class_auc_is_nan():
    names = list(PAPER_FIVE_FINDINGS)
    probs = np.full((3, 5, 3), 1 / 3)
    labels = np.zeros((3, 5), dtype=int)
    labels[2, :] = -1
    out = chexpert_crossdomain(probs, labels, names)
    assert out["per_finding"]["Atelectasis"]["n"] == 2
    assert math.isnan(out["per_finding"]["Atelectasis"]["auc"])


def test_clinical_efficacy_counts_only_labeler_positives():
    n = len(CHEXPERT_14)
    gen = np.full((3, n), np.nan)
    ref = np.full((3, n), np.nan)
    j = CHEXPERT_14.index("Edema")
    ref[:, j] = [1, 1, 0]
    gen[:, j] = [1, -1, 1]  # hit, miss (uncertain is not positive), false alarm
    out = clinical_efficacy(gen, ref)
    edema = out["per_finding"]["Edema"]
    assert (edema["tp"], edema["fp"], edema["fn"]) == (1, 1, 1)
    assert edema["f1"] == pytest.approx(0.5)
    assert out["micro_f1"] == pytest.approx(0.5)
    # Findings absent everywhere are undefined, not zero, and left out of the macro.
    assert out["macro_f1"] == pytest.approx(0.5)


def test_crossdomain_cli_refuses_three_class_labels(tmp_path):
    from scripts import evaluate_chexpert_crossdomain as cli

    names = tuple(PAPER_FIVE_FINDINGS)
    preds = ClassificationPredictions(
        labels=np.array([[2, 0, 0, 0, 0], [1, 0, 0, 0, 0]]),
        probabilities=np.full((2, 5, 3), 1 / 3),
        pathology_names=names,
        sample_keys=np.array(["a", "b"]),
    )
    path = preds.save(tmp_path / "p.npz")
    assert cli.main(["--predictions", str(path)]) == 2


def test_crossdomain_cli_writes_json_with_the_paper_row(tmp_path):
    from scripts import evaluate_chexpert_crossdomain as cli

    names = tuple(PAPER_FIVE_FINDINGS)
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=(40, 5))
    probs = np.zeros((40, 5, 3))
    probs[..., 1] = np.where(labels == 1, 0.7, 0.2)
    probs[..., 0] = 1 - probs[..., 1] - 0.05
    probs[..., 2] = 0.05
    preds = ClassificationPredictions(
        labels=labels, probabilities=probs, pathology_names=names,
        sample_keys=np.arange(40).astype(str),
    )
    path = preds.save(tmp_path / "p.npz")
    out = tmp_path / "t4.json"
    assert cli.main(["--predictions", str(path), "--output", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["mean_auc"] == pytest.approx(1.0)
    assert payload["paper_table4"]["mean"] == [0.824, 0.699]


def test_ce_cli_joins_on_sample_key(tmp_path):
    pd = pytest.importorskip("pandas")
    from scripts import evaluate_clinical_efficacy as cli

    rows = [{"sample_key": k, **{c: (1 if c == "Edema" else None) for c in CHEXPERT_14}}
            for k in ("a", "b")]
    pd.DataFrame(rows).to_csv(tmp_path / "gen.csv", index=False)
    pd.DataFrame(rows[::-1]).to_csv(tmp_path / "ref.csv", index=False)
    out = tmp_path / "ce.json"
    assert cli.main(["--generated-labels", str(tmp_path / "gen.csv"),
                     "--reference-labels", str(tmp_path / "ref.csv"),
                     "--output", str(out)]) == 0
    assert json.loads(out.read_text())["macro_f1"] == pytest.approx(1.0)


def test_paper_bertscore_flag_selects_rescaled_distilroberta(monkeypatch, tmp_path):
    from scripts import evaluate_stage2 as cli

    seen = {}

    def fake(generated, references, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(cli, "compute_generation_metrics", fake)
    from training.evaluation.schemas import GenerationRecord, save_generation_records

    path = save_generation_records(
        [GenerationRecord(sample_key="s", generated="a", reference="a")], tmp_path / "r.jsonl"
    )
    with pytest.raises(SystemExit):
        cli.main(["--predictions", str(path), "--metrics", "bertscore",
                  "--paper-bertscore", "--output-dir", str(tmp_path / "o")])
    assert seen["bertscore_model"] == "distilroberta-base"
    assert seen["bertscore_rescale"] is True
