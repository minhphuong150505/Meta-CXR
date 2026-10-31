"""Pins the BioViL-T previous-image probe helpers (docs/handoff/PLAN-2026-10-31-biovil-prior-probe.md)."""

import importlib.util
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

_ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


feat = _module("biovil_prior_features")
fit = _module("biovil_prior_fit")


def _toy():
    # patient 1: s10 (day 0, AP + lateral), s11 (day 5, lateral only), s12 (day 9, PA + AP),
    #            s13 (day 9, same instant as s12), s14 (day 20, frontal)
    # patient 2: s20 (day 3, PA) -- must never be patient 1's prior
    # patient 3: s30 (day 0, AP + PA), s31 (day 1, PA)
    rows = [
        (3, 30, "d30a", "AP"), (3, 30, "d30b", "PA"), (3, 31, "d31", "PA"),
        (1, 10, "d10a", "AP"), (1, 10, "d10b", "LATERAL"),
        (1, 11, "d11", "LATERAL"),
        (1, 12, "d12b", "AP"), (1, 12, "d12a", "PA"),
        (1, 13, "d13", "PA"),
        (1, 14, "d14", "LATERAL"), (1, 14, "d14f", "AP"),
        (2, 20, "d20", "PA"),
    ]
    split = pd.DataFrame(rows, columns=["subject_id", "study_id", "dicom_id", "ViewPosition"])
    split["image_path"] = "files/" + split["dicom_id"] + ".jpg"
    day = {10: 0, 11: 5, 12: 9, 13: 9, 14: 20, 20: 3, 30: 0, 31: 1}
    times = pd.Series({s: pd.Timestamp("2150-01-01") + pd.Timedelta(days=d) for s, d in day.items()})
    return split, times


def test_prior_is_latest_strictly_earlier_frontal_study_of_same_patient():
    split, times = _toy()
    t = feat.prior_frontal_table(split, times)
    assert pd.isna(t.at[10, "prior_image_path"])          # first study: no prior
    assert t.at[11, "prior_image_path"] == "files/d10a.jpg" and t.at[11, "prior_days"] == 5
    assert t.at[12, "prior_image_path"] == "files/d10a.jpg"  # s11 has no frontal image
    assert t.at[13, "prior_image_path"] == "files/d10a.jpg"  # s12 is the same instant, not earlier
    assert t.at[14, "prior_image_path"] == "files/d13.jpg"   # s12/s13 tie: the larger study_id
    assert t.at[31, "prior_image_path"] == "files/d30b.jpg"  # PA chosen over AP
    assert pd.isna(t.at[20, "prior_image_path"])          # other patient's studies never leak in


def test_lookup_offers_a_prior_only_to_frontal_anchors():
    split, times = _toy()
    lookup = feat.prior_lookup(split, times, "/root")
    assert lookup("d14") == (None, -1)                     # lateral anchor
    path, days = lookup("d14f")
    assert days == 11 and path.endswith("d13.jpg") and path.startswith("/root")
    assert lookup("d10a") == (None, -1)


def test_reading_rule_and_gate1():
    assert fit.verdict(0.021, 0.001) == "GO_ON"
    assert fit.verdict(0.021, -0.001) == "NOT_ENOUGH" and fit.verdict(0.005, 0.001) == "NOT_ENOUGH"
    assert fit.verdict(0.0, -0.01) == "STOP"
    ok = fit.gate1({"attn_swin": 0.4263, "attn_biovil": 0.4060, "attn_swin+biovil": 0.4300})
    assert all(v["passed"] for v in ok.values())
    bad = fit.gate1({"attn_swin": 0.4263, "attn_biovil": 0.4100, "attn_swin+biovil": 0.4315})
    assert not bad["attn_biovil"]["passed"]
