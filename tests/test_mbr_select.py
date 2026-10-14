"""MBR selection over Stage-2 candidates (scripts/mbr_select_stage2.py), CPU only.

The utility here is a toy word-overlap score, so what is pinned is the
selection rule itself: the expectation runs over the samples only, a sample is
never scored against itself, ties go to the earlier hypothesis, and two
different cohorts are never pooled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import mbr_select_stage2 as mbr  # noqa: E402


def overlap(a: list[str], b: list[str]) -> np.ndarray:
    out = []
    for x, y in zip(a, b, strict=True):
        sx, sy = set(x.split()), set(y.split())
        out.append(len(sx & sy) / max(len(sx | sy), 1))
    return np.asarray(out)


def _study(texts, sources):
    return {"sample_key": "k", "reference": "r", "texts": list(texts),
            "sources": list(sources), "is_ref": [s == mbr.SAMPLE_SOURCE for s in sources]}


def _chosen_text(study):
    return study["texts"][study["chosen"]]


def test_the_outlier_sample_is_not_chosen():
    (study,) = mbr.select([_study(["a b c", "a b d", "x y z"], ["sample"] * 3)], overlap)
    assert _chosen_text(study) == "a b c"          # tie with "a b d" -> earlier one
    assert study["expected"][2] == 0.0


def test_a_sample_is_never_scored_against_itself():
    # With a self-term each sample would gain u(s, s) = 1 and beat the beam.
    (study,) = mbr.select([_study(["a b c", "a b c d", "x y"], ["beam", "sample", "sample"])],
                          overlap)
    assert _chosen_text(study) == "a b c"
    assert study["expected"][1] == 0.0 and study["expected"][2] == 0.0


def test_deterministic_decodes_do_not_vote():
    # If beam and greedy counted as pseudo-references, "a b" would win on their votes.
    texts = ["a b", "a b", "c d", "c d", "a b"]
    sources = ["beam", "greedy", "sample", "sample", "sample"]
    (study,) = mbr.select([_study(texts, sources)], overlap)
    assert _chosen_text(study) == "c d"
    np.testing.assert_allclose(study["expected"], [1 / 3, 1 / 3, 0.5, 0.5, 0.0])


def test_ties_go_to_the_first_hypothesis():
    (study,) = mbr.select([_study(["a", "a", "a"], ["beam", "sample", "sample"])], overlap)
    assert study["chosen"] == 0


def test_only_pairs_with_a_sample_are_scored():
    pairs = mbr.needed_pairs(["h1", "h2", "s1", "s2"], [False, False, True, True])
    assert (0, 1) not in pairs
    assert pairs == [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_identical_texts_need_no_model_call():
    calls = []

    def recording(a, b):
        calls.append(list(zip(a, b, strict=True)))
        return overlap(a, b)

    mbr.select([_study(["same", "same", "same"], ["beam", "sample", "sample"])], recording)
    assert calls == []


def test_chunking_does_not_change_the_choice():
    studies = [_study(t, ["beam", "sample", "sample", "sample"]) for t in (
        ["a b", "a b c", "a c", "x"], ["p q r", "p", "q r", "p q"], ["m", "n", "m n", "m"])]
    whole = [s["chosen"] for s in mbr.select([dict(s) for s in studies], overlap, chunk=250)]
    single = [s["chosen"] for s in mbr.select([dict(s) for s in studies], overlap, chunk=1)]
    assert whole == single


def test_a_utility_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="shape"):
        mbr.select([_study(["a", "b", "c"], ["sample"] * 3)], lambda a, b: np.zeros(1))


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _files(tmp_path, *, ref_b="ref B", sample_keys=("k1", "k2"), n_samples=2):
    beam = _write(tmp_path / "beam.jsonl", [
        {"sample_key": "k1", "generated": "beam one", "reference": "ref A"},
        {"sample_key": "k2", "generated": "beam two", "reference": ref_b}])
    samples = _write(tmp_path / "candidates_val.jsonl", [
        {"sample_key": k, "candidates": [f"s{i} {k}" for i in range(n_samples)],
         "reference": "ref A" if k == "k1" else "ref B"} for k in sample_keys])
    return beam, samples


def test_pool_orders_hypotheses_before_samples(tmp_path):
    beam, samples = _files(tmp_path)
    studies = mbr.pool_candidates({"beam": beam}, samples)
    assert [s["sample_key"] for s in studies] == ["k1", "k2"]
    assert studies[0]["texts"] == ["beam one", "s0 k1", "s1 k1"]
    assert studies[0]["sources"] == ["beam", "sample", "sample"]
    assert studies[0]["is_ref"] == [False, True, True]


def test_pool_refuses_a_different_cohort(tmp_path):
    beam, samples = _files(tmp_path, sample_keys=("k1", "k3"))
    with pytest.raises(ValueError, match="not the same cohort"):
        mbr.pool_candidates({"beam": beam}, samples)


def test_pool_refuses_a_different_reference(tmp_path):
    beam, samples = _files(tmp_path, ref_b="another report")
    with pytest.raises(ValueError, match="disagree on a reference"):
        mbr.pool_candidates({"beam": beam}, samples)


def test_pool_needs_two_samples_per_study(tmp_path):
    beam, samples = _files(tmp_path, n_samples=1)
    with pytest.raises(ValueError, match="at least 2 samples"):
        mbr.pool_candidates({"beam": beam}, samples)


def test_pool_refuses_a_deterministic_file_as_samples(tmp_path):
    beam, _ = _files(tmp_path)
    with pytest.raises(ValueError, match="candidates"):
        mbr.pool_candidates({}, beam)


def test_pool_reserves_the_sample_name(tmp_path):
    beam, samples = _files(tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        mbr.pool_candidates({"sample": beam}, samples)


def test_main_writes_the_choice_and_a_text_free_summary(tmp_path, monkeypatch):
    beam, samples = _files(tmp_path)
    monkeypatch.setattr(mbr, "bertscore_utility", lambda device, batch_size: overlap)
    out = tmp_path / "mbr"
    assert mbr.main(["--split", "val", "--hyp", f"beam={beam}", "--samples", str(samples),
                     "--output-dir", str(out), "--device", "cpu"]) == 0
    rows = [json.loads(line) for line in (out / "generated_val.jsonl").read_text().splitlines()]
    assert [r["sample_key"] for r in rows] == ["k1", "k2"]
    assert {r["mbr_source"] for r in rows} <= {"beam", "sample"}
    assert rows[0]["reference"] == "ref A"
    summary_text = (out / "mbr_summary.json").read_text()
    summary = json.loads(summary_text)
    assert summary["n_studies"] == 2 and summary["samples_per_study"] == [2]
    for text in ("beam one", "s0 k1", "ref A"):
        assert text not in summary_text
