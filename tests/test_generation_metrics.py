"""Stage-2 generation metric and error-analysis tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.error_analysis import (  # noqa: E402
    EMPTY_OUTPUT,
    EXCESSIVE_REPETITION,
    POSSIBLE_LATERALITY_ERROR,
    POSSIBLE_NEGATION_ERROR,
    POSSIBLE_OMISSION,
    POSSIBLE_TEMPORAL_HALLUCINATION,
    analyse_sample,
    detect_temporal_hallucination,
    repetition_ratio,
    summarise_errors,
)
from training.evaluation.generation_metrics import (  # noqa: E402
    BERTSCORE,
    BLEU,
    CIDER,
    METEOR,
    ROUGE,
    MissingMetricDependency,
    compute_generation_metrics,
    corpus_bleu,
    normalize,
    rouge_l,
    tokenize,
)
from training.evaluation.schemas import (  # noqa: E402
    SchemaError,
    load_generation_records,
)


# --------------------------------------------------------------------------
# Normalisation must not destroy clinical meaning
# --------------------------------------------------------------------------


def test_normalisation_collapses_whitespace_and_newlines():
    assert normalize("  There  is\n\n no   effusion. ") == "there is no effusion."


def test_clinically_load_bearing_words_survive_tokenisation():
    text = "No new left pleural effusion; right base unchanged, without consolidation."
    tokens = tokenize(text)
    for word in ("no", "new", "left", "right", "unchanged", "without"):
        assert word in tokens, f"{word!r} was dropped by tokenisation"


def test_punctuation_is_kept_as_separate_tokens():
    assert tokenize("no effusion.") == ["no", "effusion", "."]


def test_unicode_is_handled():
    tokens = tokenize("Opacité in the naso—gastric région")
    assert "opacité" in tokens
    assert "région" in tokens


# --------------------------------------------------------------------------
# BLEU / ROUGE values
# --------------------------------------------------------------------------


def test_identical_text_scores_perfectly():
    text = ["the lungs are clear without focal consolidation"]
    scores = corpus_bleu(text, text)
    assert scores["bleu_1"] == pytest.approx(1.0)
    assert scores["bleu_4"] == pytest.approx(1.0)
    assert rouge_l(text[0], text[0]) == pytest.approx(1.0)


def test_completely_disjoint_text_scores_zero():
    assert corpus_bleu(["alpha beta gamma"], ["delta epsilon zeta"])["bleu_1"] == 0.0
    assert rouge_l("alpha beta gamma", "delta epsilon zeta") == 0.0


def test_bleu_1_is_hand_computable():
    # prediction 4 tokens, 3 of them appear in the reference -> precision 3/4.
    # Lengths are equal, so the brevity penalty is 1.
    scores = corpus_bleu(["a b c d"], ["a b c e"])
    assert scores["bleu_1"] == pytest.approx(0.75)


def test_brevity_penalty_punishes_short_output():
    long_reference = "a b c d e f g h"
    short = corpus_bleu(["a b"], [long_reference])["bleu_1"]
    full = corpus_bleu([long_reference], [long_reference])["bleu_1"]
    assert short < full


def test_rouge_l_rewards_subsequence_order():
    # Same tokens, reversed order: LCS is much shorter than the token overlap,
    # so a bag-of-words metric would score this far higher.
    assert rouge_l("a b c d", "d c b a") < rouge_l("a b c d", "a b c d")


def test_rouge_l_is_hand_computable():
    # LCS("the lungs are clear", "the lungs are hyperinflated") = 3 tokens.
    # precision = recall = 3/4; with beta=1.2 the F-measure is still 0.75.
    assert rouge_l("the lungs are clear", "the lungs are hyperinflated") == pytest.approx(
        0.75
    )


# --------------------------------------------------------------------------
# Empty and malformed input
# --------------------------------------------------------------------------


def test_empty_prediction_scores_zero_and_does_not_crash():
    suite = compute_generation_metrics([""], ["there is no effusion"])
    assert suite.corpus["bleu_4"] == 0.0
    assert suite.corpus["rouge_l"] == 0.0
    assert suite.provenance["empty_predictions"] == 1


def test_empty_reference_does_not_crash():
    suite = compute_generation_metrics(["there is no effusion"], [""])
    assert suite.corpus["rouge_l"] == 0.0


def test_mismatched_counts_raise():
    with pytest.raises(ValueError, match="one reference per prediction"):
        compute_generation_metrics(["a", "b"], ["a"])


def test_no_predictions_raises():
    with pytest.raises(ValueError, match="no predictions"):
        compute_generation_metrics([], [])


def test_unknown_metric_name_raises():
    with pytest.raises(ValueError, match="unknown metric"):
        compute_generation_metrics(["a"], ["a"], metrics=("blue",))


def test_duplicate_sample_keys_are_rejected(tmp_path):
    path = tmp_path / "reports.jsonl"
    path.write_text(
        '{"sample_key": "s1", "generated": "a", "reference": "a"}\n'
        '{"sample_key": "s1", "generated": "b", "reference": "b"}\n'
    )
    with pytest.raises(SchemaError, match="repeats sample_key"):
        load_generation_records(path)


def test_missing_required_field_is_rejected(tmp_path):
    path = tmp_path / "reports.jsonl"
    path.write_text('{"sample_key": "s1", "generated": "a"}\n')
    with pytest.raises(SchemaError, match="missing required field"):
        load_generation_records(path)


# --------------------------------------------------------------------------
# A missing dependency must never become a score of 0.0
# --------------------------------------------------------------------------


def test_missing_dependency_is_reported_not_scored_zero():
    """The single most important Stage-2 guarantee.

    The legacy `compute_nlg` wrapped these in `except Exception: x = 0.0`.
    Here an unavailable metric is absent from `corpus` and present in
    `unavailable` with an install command.
    """
    suite = compute_generation_metrics(
        ["the lungs are clear"],
        ["the lungs are clear"],
        metrics=(BLEU, METEOR, CIDER, BERTSCORE),
    )
    for metric in (METEOR, CIDER, BERTSCORE):
        if metric in suite.unavailable:
            assert metric not in suite.corpus
            assert "pip install" in suite.unavailable[metric]
    # BLEU is native and always available.
    assert suite.corpus["bleu_1"] == pytest.approx(1.0)


def test_strict_mode_raises_on_missing_dependency():
    try:
        compute_generation_metrics(["a"], ["a"], metrics=(METEOR,), strict=True)
    except MissingMetricDependency as exc:
        assert "pip install" in str(exc)
    else:
        pytest.skip("nltk is installed in this environment")


def test_provenance_records_implementation_and_normalisation():
    suite = compute_generation_metrics(["a b"], ["a b"], metrics=(BLEU, ROUGE))
    assert "corpus_bleu" in suite.provenance[BLEU]["implementation"]
    assert suite.provenance["normalization"]["remove_stopwords"] is False
    assert suite.provenance["normalization"]["strip_punctuation"] is False


def test_per_sample_scores_are_returned():
    suite = compute_generation_metrics(
        ["a b c", "x y z"], ["a b c", "a b c"], metrics=(ROUGE,)
    )
    assert len(suite.per_sample["rouge_l"]) == 2
    assert suite.per_sample["rouge_l"][0] > suite.per_sample["rouge_l"][1]


# --------------------------------------------------------------------------
# Temporal hallucination
# --------------------------------------------------------------------------


def test_temporal_phrase_without_prior_context_is_flagged():
    flagged, matched = detect_temporal_hallucination(
        "The effusion is unchanged from the prior study."
    )
    assert flagged
    assert "unchanged" in matched


def test_temporal_phrase_is_suppressed_when_context_has_a_prior():
    flagged, _ = detect_temporal_hallucination(
        "The effusion is unchanged.", context="Comparison: prior radiograph from 2 days ago."
    )
    assert not flagged


def test_report_without_comparison_is_not_flagged():
    flagged, matched = detect_temporal_hallucination(
        "The lungs are clear. No focal consolidation."
    )
    assert not flagged
    assert matched == []


# --------------------------------------------------------------------------
# Error analysis
# --------------------------------------------------------------------------


def test_negation_difference_is_caught_despite_high_lexical_overlap():
    """The case NLG metrics cannot see.

    These two reports differ by one word and are clinically opposite. ROUGE-L
    stays high; the negation flag is what catches it.
    """
    reference = "There is no pneumothorax."
    generated = "There is a pneumothorax."

    assert rouge_l(generated, reference) > 0.6  # lexically similar

    report = analyse_sample("s1", generated, reference)
    assert POSSIBLE_NEGATION_ERROR in report.flags
    assert "Pneumothorax" in report.negation_mismatches


def test_omitted_finding_is_flagged():
    report = analyse_sample(
        "s1",
        generated="The lungs are clear.",
        reference="There is a large pleural effusion.",
    )
    assert POSSIBLE_OMISSION in report.flags
    assert "Pleural Effusion" in report.false_negative_findings


def test_empty_report_is_flagged_and_does_not_crash():
    report = analyse_sample("s1", generated="", reference="No acute process.")
    assert EMPTY_OUTPUT in report.flags
    assert report.empty
    assert report.generated_length == 0


def test_repeated_sentences_are_flagged():
    text = "The lungs are clear. The lungs are clear. The lungs are clear."
    assert repetition_ratio(text) == pytest.approx(2 / 3)
    report = analyse_sample("s1", generated=text, reference="The lungs are clear.")
    assert EXCESSIVE_REPETITION in report.flags


def test_laterality_mismatch_is_flagged():
    report = analyse_sample(
        "s1",
        generated="Left pleural effusion.",
        reference="Right pleural effusion.",
    )
    assert POSSIBLE_LATERALITY_ERROR in report.flags


def test_temporal_hallucination_reaches_the_sample_flags():
    report = analyse_sample(
        "s1",
        generated="Unchanged since prior study.",
        reference="The lungs are clear.",
    )
    assert POSSIBLE_TEMPORAL_HALLUCINATION in report.flags


def test_summary_reports_rates_and_carries_a_caveat():
    reports = [
        analyse_sample("s1", "The lungs are clear.", "The lungs are clear."),
        analyse_sample("s2", "", "No acute process."),
    ]
    summary = summarise_errors(reports)
    assert summary["num_samples"] == 2
    assert summary["empty_output_rate"] == pytest.approx(0.5)
    assert "heuristic" in summary["caveat"]
    assert summary["generated_length"]["min"] == 0.0
