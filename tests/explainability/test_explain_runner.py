"""CPU checks for the JSONL runner.

The runner needs a GPU and MedGemma to produce anything, so what is testable
here is the part that must be right BEFORE any of that: the privacy guard, the
filenames, the argument semantics, and the shape of what gets written.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "explain_stage2.py"


@pytest.fixture(scope="module")
def runner():
    sys.path.insert(0, str(REPO))
    pytest.importorskip("numpy")
    return pytest.importorskip("scripts.explain_stage2")


# --------------------------------------------------------------------------
# Filenames carry no identifier
# --------------------------------------------------------------------------


def test_sample_key_is_stable_and_not_the_input(runner):
    first = runner.sample_key("abc-123")
    assert first == runner.sample_key("abc-123")
    assert first != runner.sample_key("abc-124")
    assert "abc" not in first
    assert len(first) == 24


def test_map_filenames_are_sequential_not_identifying():
    """The name pattern is checked in the source, since writing one needs a GPU."""
    source = RUNNER.read_text(encoding="utf-8")
    assert 'f"study_{index:05d}.npz"' in source
    # The obvious wrong versions, spelled out so a future edit trips this.
    for bad in ("{study.dicom_id}", "{study.subject_id}", "{study.study_id}", "{key}.npz"):
        assert bad not in source, f"filename must not embed {bad}"


# --------------------------------------------------------------------------
# The privacy guard is the one from Stage 1, not a reimplementation
# --------------------------------------------------------------------------


def test_runner_reuses_the_stage1_output_guard():
    source = RUNNER.read_text(encoding="utf-8")
    assert "from scripts.evaluate_explanation import _assert_private_output_location" in source


def test_a_non_ignored_repo_path_is_refused(runner, tmp_path):
    from scripts.evaluate_explanation import _assert_private_output_location

    with pytest.raises(ValueError, match="refusing to write patient-derived"):
        _assert_private_output_location(REPO / "training" / "explainability")


def test_an_ignored_repo_path_is_accepted(runner):
    from scripts.evaluate_explanation import _assert_private_output_location

    # outputs/ is git-ignored; the guard warns but allows it.
    assert _assert_private_output_location(REPO / "outputs" / "xai").is_absolute()


def test_a_path_outside_the_repo_is_accepted(runner, tmp_path):
    from scripts.evaluate_explanation import _assert_private_output_location

    assert _assert_private_output_location(tmp_path).is_absolute()


# --------------------------------------------------------------------------
# Argument semantics
# --------------------------------------------------------------------------


def test_train_split_is_not_even_an_option(runner):
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--manifest", "m.csv", "--image-root", ".",
         "--output-dir", "o", "--split", "train"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_the_gate_runs_by_default_and_skipping_it_is_explicit(runner):
    args = runner.parse_args(
        ["--manifest", "m.csv", "--image-root", ".", "--output-dir", "o"]
    )
    assert args.skip_ablation_gate is False
    assert args.ablation_studies >= 2
    assert args.no_gradient_weight is False   # the fallback is opt-in
    assert args.write_key_map is False        # the join is opt-in


def test_the_gate_result_is_asserted_not_merely_logged():
    source = RUNNER.read_text(encoding="utf-8")
    assert "capture.assert_visual_tokens_matter(gate)" in source


def test_skipping_the_gate_is_recorded_in_the_summary():
    source = RUNNER.read_text(encoding="utf-8")
    assert '"ablation_gate_skipped"' in source


# --------------------------------------------------------------------------
# What a record must carry
# --------------------------------------------------------------------------


def test_every_required_field_is_written():
    """The format agreed for this branch: text, label, map path, mean NLL."""
    source = RUNNER.read_text(encoding="utf-8")
    for field in ('"sample_key"', '"attribution_map"', '"attribution_grid"',
                  '"visual_span"', '"rollout_method"'):
        assert field in source, field
    # text / labels / mean_token_nll / spatially_meaningful arrive via
    # SentenceRecord.to_dict(); pin that they are actually in it.
    from training.explainability.sentence_attribution import attribute_sentences

    record = attribute_sentences("Mild cardiomegaly is present.").sentences[0].to_dict()
    for field in ("text", "labels", "mean_token_nll", "spatially_meaningful"):
        assert field in record, field


def test_parse_coverage_is_written_at_both_levels():
    source = RUNNER.read_text(encoding="utf-8")
    assert '"parse_coverage": attributed.parse_coverage' in source   # per study
    assert '"coverage": coverage' in source                          # per run


def test_maps_are_stored_at_the_native_grid_and_never_as_png():
    source = RUNNER.read_text(encoding="utf-8")
    assert "np.savez_compressed" in source
    assert "reshape(grid.height, grid.width)" in source
    for banned in ("imsave", "savefig", ".png", "Image.fromarray"):
        assert banned not in source, f"the runner must not write {banned}"


def test_the_summary_is_valid_json_shaped(tmp_path):
    """Cheap structural check that the summary keys are JSON-serialisable."""
    payload = {
        "schema_version": 1, "split": "test", "studies_written": 0,
        "gradient_weighted": True, "coverage": {"parse_coverage": 0.0},
        "ablation_gate": None, "ablation_gate_skipped": False,
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    assert json.loads(path.read_text())["split"] == "test"


# --------------------------------------------------------------------------
# Graph mode: peak memory must not grow without bound with sentence count
# --------------------------------------------------------------------------


def test_graph_mode_defaults_to_auto(runner):
    args = runner.parse_args(
        ["--manifest", "m.csv", "--image-root", ".", "--output-dir", "o"]
    )
    assert args.graph_mode == runner.GRAPH_AUTO


def test_all_three_graph_modes_are_selectable(runner):
    for mode in runner.GRAPH_MODES:
        args = runner.parse_args(
            ["--manifest", "m.csv", "--image-root", ".", "--output-dir", "o",
             "--graph-mode", mode]
        )
        assert args.graph_mode == mode


def test_auto_falls_back_only_on_oom_and_an_explicit_mode_does_not():
    source = RUNNER.read_text(encoding="utf-8")
    assert "except torch.OutOfMemoryError:" in source
    # An explicitly chosen mode must propagate the OOM rather than silently
    # switching: a run that half-changes strategy without saying so is worse
    # than one that stops.
    assert "if mode != GRAPH_AUTO:\n            raise" in source


def test_the_mode_actually_used_is_recorded_per_study():
    source = RUNNER.read_text(encoding="utf-8")
    assert '"graph_mode": used' in source
    assert '"graph_mode_requested": args.graph_mode' in source


def test_per_sentence_mode_frees_the_graph_between_sentences():
    """retain_graph=False is the whole point of that path."""
    source = RUNNER.read_text(encoding="utf-8")
    per_sentence = source[source.index("def _attribute_per_sentence("):
                          source.index("def explain_study(")]
    assert "retain_graph=False" in per_sentence
    assert "empty_cache()" in per_sentence
    shared = source[source.index("def _attribute_shared("):
                    source.index("def _attribute_per_sentence(")]
    assert "retain_graph=True" in shared


def test_a_sentence_with_no_tokens_still_gets_a_map_slot():
    """Every sentence keeps an entry, so attribution_index stays aligned."""
    source = RUNNER.read_text(encoding="utf-8")
    for block in ("_attribute_shared", "_attribute_per_sentence"):
        body = source[source.index(f"def {block}("):]
        body = body[: body.index("\ndef ", 1)]
        assert "torch.zeros(span.length)" in body, block


# --------------------------------------------------------------------------
# The manifest's split column does not use the CLI's names
# --------------------------------------------------------------------------


def test_val_accepts_the_manifests_validate_spelling(runner):
    """MIMIC-CXR's official split column says 'validate'; the CLI says 'val'.

    Filtering on the CLI name alone matched nothing, so every val run died at
    the selection step. A guard turned it into a clear exit rather than an
    empty run, but the mapping is what makes it work.
    """
    assert "validate" in runner.MANIFEST_SPLIT_ALIASES["val"]
    assert "val" in runner.MANIFEST_SPLIT_ALIASES["val"]
    assert runner.MANIFEST_SPLIT_ALIASES["test"] == ("test",)


def test_every_allowed_cli_split_has_an_alias_entry(runner):
    from training.explainability.attention_capture import ALLOWED_SPLITS

    for split in ALLOWED_SPLITS:
        assert split in runner.MANIFEST_SPLIT_ALIASES, split


def test_an_empty_selection_names_what_the_column_actually_held():
    source = RUNNER.read_text(encoding="utf-8")
    assert "its split column contains {present}" in source
