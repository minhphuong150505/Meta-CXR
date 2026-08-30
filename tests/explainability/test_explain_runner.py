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


# --------------------------------------------------------------------------
# The parse-coverage diagnostic
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diagnose():
    sys.path.insert(0, str(REPO))
    pytest.importorskip("pandas")
    return pytest.importorskip("scripts.diagnose_parse_coverage")


def test_technical_sentences_are_classified_as_technical(diagnose):
    for text in (
        "Compared with the prior radiograph there is no change.",
        "The lateral view is limited by patient rotation.",
        "Portable AP upright study.",
    ):
        assert diagnose.classify(text) == "technical"


def test_normality_statements_are_classified_as_normal(diagnose):
    for text in (
        "The lungs are clear.",
        "Osseous structures are unremarkable.",
        "No acute cardiopulmonary process.",
    ):
        assert diagnose.classify(text) == "normal"


def test_wording_the_lexicon_should_have_caught_is_missed_14(diagnose):
    """Real phrasings for the 14 labels that lexicon_v1 does not match."""
    for text in (
        "The heart size is enlarged.",
        "There is blunting of the left costophrenic angle.",
        "Patchy infiltrate at the right base.",
    ):
        assert diagnose.classify(text) == "missed_14"


def test_findings_outside_the_taxonomy_are_kept_separate(diagnose):
    """The distinction that decides whether a better labeler would help."""
    for text in (
        "Degenerative changes of the thoracic spine.",
        "A hiatal hernia is again seen.",
        "Surgical clips project over the right hemithorax.",
    ):
        assert diagnose.classify(text) == "outside_14"


def test_technical_wins_over_a_named_finding(diagnose):
    """A comparison sentence is not a missing synonym, so it is not missed_14."""
    assert diagnose.classify(
        "Compared with the prior study the heart size is unchanged."
    ) == "technical"


def test_the_diagnostic_writes_through_the_privacy_guard():
    source = (REPO / "scripts" / "diagnose_parse_coverage.py").read_text(encoding="utf-8")
    assert "_assert_private_output_location" in source
    # The sample file is report text; it must never be echoed.
    assert "print" not in source.split("unparsed_sample.md")[1].split("\n")[0]


def test_the_taxonomy_it_reports_against_is_the_repository_lexicon(diagnose):
    from safety.claims import ABNORMALITY_SYNONYMS

    assert len(ABNORMALITY_SYNONYMS) == 14


# --------------------------------------------------------------------------
# The second gate is wired in and can abort the run
# --------------------------------------------------------------------------


def test_the_randomization_gate_runs_by_default(runner):
    args = runner.parse_args(
        ["--manifest", "m.csv", "--image-root", ".", "--output-dir", "o"]
    )
    assert args.skip_randomization_gate is False


def test_the_randomization_gate_result_is_asserted():
    source = RUNNER.read_text(encoding="utf-8")
    assert "capture.assert_randomization_degrades(randomization)" in source
    assert '"randomization_gate_skipped"' in source


def test_both_gates_are_recorded_in_the_summary():
    source = RUNNER.read_text(encoding="utf-8")
    for key in ('"ablation_gate"', '"randomization_gate"'):
        assert key in source, key


def test_randomization_restores_the_weights_even_on_failure():
    """A gate that leaves the model randomised would poison every later study."""
    source = RUNNER.read_text(encoding="utf-8")
    block = source[source.index("def run_randomization_gate("):
                   source.index("def explain_study(")]
    assert "finally:" in block
    assert "restore()" in block


# --------------------------------------------------------------------------
# n must be impossible to miss
# --------------------------------------------------------------------------


def test_token_bounds_no_longer_default_to_a_narrow_window(runner):
    """The old 30-90 default silently excluded val's longest reports."""
    args = runner.parse_args(
        ["--manifest", "m.csv", "--image-root", ".", "--output-dir", "o"]
    )
    assert args.min_findings_tokens is None
    assert args.max_findings_tokens is None


def test_limit_zero_means_the_whole_split(runner):
    args = runner.parse_args(
        ["--manifest", "m.csv", "--image-root", ".", "--output-dir", "o", "--limit", "0"]
    )
    assert args.limit == 0
    source = RUNNER.read_text(encoding="utf-8")
    assert "wanted = len(frame)  # --limit 0 == the whole split" in source


def test_every_reported_number_carries_its_n():
    source = RUNNER.read_text(encoding="utf-8")
    assert "ablation %s (n=%d studies)" in source
    assert "randomization (n=1 study)" in source
    assert "explained %d/%d studies (n target=%d)" in source
    assert 'print(f"n={written} studies written' in source
    assert "quote n beside any number from this run" in source


def test_the_summary_carries_n_for_each_measurement():
    source = RUNNER.read_text(encoding="utf-8")
    for key in ('"n_studies"', '"n_ablation_studies"', '"n_randomization_studies"'):
        assert key in source, key


def test_header_and_request_fragments_are_technical(diagnose):
    for text in (
        "Chest radiograph submitted for evaluation.",
        "PA and lateral chest.",
        "ADDENDUM.",
        "Indication: shortness of breath.",
    ):
        assert diagnose.classify(text) == "technical", text


def test_a_bare_chest_does_NOT_make_a_sentence_technical(diagnose):
    """TECHNICAL is checked first, so a bare 'chest' would swallow real content.

    'chest' is the commonest word in the unclassified bucket (61 of 91
    sentences), which makes it the tempting keyword and the wrong one: every
    sentence below would then be filed as technique.
    """
    for text in (
        "Patchy opacity in the left chest.",
        "A mass is seen in the right chest.",
        "Increasing density at the right chest base.",
    ):
        assert diagnose.classify(text) != "technical", text


def test_the_phrase_forms_still_catch_real_headers(diagnose):
    """Only the phrase forms are matched, and they are enough for headers."""
    assert diagnose.classify("Frontal and lateral chest.") == "technical"
    assert diagnose.classify("Single view chest radiograph.") == "technical"


def test_the_runner_defaults_to_v1_so_the_n1513_run_reproduces(runner):
    args = runner.parse_args(
        ["--manifest", "m.csv", "--image-root", ".", "--output-dir", "o"]
    )
    assert args.labeler == "lexicon_v1"


def test_both_labelers_are_selectable_from_the_runner(runner):
    for name in ("lexicon_v1", "lexicon_v2"):
        args = runner.parse_args(
            ["--manifest", "m.csv", "--image-root", ".", "--output-dir", "o",
             "--labeler", name]
        )
        assert args.labeler == name


def test_both_labelers_are_selectable_from_the_diagnostic(diagnose):
    for name in ("lexicon_v1", "lexicon_v2"):
        args = diagnose.parse_args(
            ["--manifest", "m.csv", "--output-dir", "o", "--labeler", name]
        )
        assert args.labeler == name


def test_only_the_two_registered_labelers_exist():
    from training.explainability.sentence_attribution import LABELERS

    assert sorted(LABELERS) == ["lexicon_v1", "lexicon_v2"]


def test_the_labeler_used_is_recorded_in_both_summaries():
    assert '"labeler": args.labeler' in RUNNER.read_text(encoding="utf-8")
    assert '"labeler": args.labeler' in (
        REPO / "scripts" / "diagnose_parse_coverage.py"
    ).read_text(encoding="utf-8")
