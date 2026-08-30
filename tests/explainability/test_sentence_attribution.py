"""Sentence splitting, token alignment, labels and parse coverage.

The alignment tests matter most. Mapping generated tokens onto sentences is the
step where an off-by-one is silent: every downstream number still computes, and
one sentence simply carries another sentence's uncertainty and another
sentence's heatmap.
"""

from __future__ import annotations

import pytest

from safety.claims import split_sentences
from training.explainability.sentence_attribution import (
    LEXICON_LABELER_NAME,
    LexiconSentenceLabeler,
    SentenceLabel,
    align_tokens_to_sentences,
    attribute_sentences,
    dataset_parse_coverage,
    locate_sentences,
)

REPORT = "There is no pneumothorax. Mild cardiomegaly is present. Patient tolerated well."


# --------------------------------------------------------------------------
# locate_sentences -- offsets that agree with the existing splitter
# --------------------------------------------------------------------------


def test_locate_sentences_agrees_with_the_existing_splitter():
    """Do not let a second splitter appear.

    ``safety.claims.split_sentences`` is the repository's splitter. This module
    only recovers offsets, so the sentence TEXTS must be identical to what that
    function returns, on every input.
    """
    for text in (
        REPORT,
        "One finding; another finding. A third.",
        "No punctuation at all",
        "  leading and trailing whitespace.  ",
        "",
    ):
        assert [item[0] for item in locate_sentences(text)] == split_sentences(text)


def test_locate_sentences_returns_spans_that_index_back_into_the_source():
    for sentence, start, end in locate_sentences(REPORT):
        assert REPORT[start:end] == sentence


def test_locate_sentences_spans_are_ordered_and_non_overlapping():
    spans = locate_sentences(REPORT)
    for (_, _, previous_end), (_, start, _) in zip(spans, spans[1:], strict=False):
        assert previous_end <= start


def test_locate_sentences_handles_a_repeated_sentence():
    # A naive ``str.find`` without a cursor would return the first occurrence
    # twice and collapse the two spans onto each other.
    text = "No acute finding. No acute finding."
    spans = locate_sentences(text)
    assert len(spans) == 2
    assert spans[0][1] != spans[1][1]
    assert text[spans[1][1] : spans[1][2]] == "No acute finding."


def test_locate_sentences_on_empty_text():
    assert locate_sentences("") == []


# --------------------------------------------------------------------------
# align_tokens_to_sentences
# --------------------------------------------------------------------------


def test_tokens_are_assigned_to_the_sentence_they_fall_inside():
    text = "Aa bb. Cc dd."
    spans = locate_sentences(text)
    tokens = ["Aa", " bb", ".", " ", "Cc", " dd", "."]
    buckets = align_tokens_to_sentences(tokens, spans)
    assert buckets[0] == (0, 1, 2)
    # Index 3 is the inter-sentence space and belongs to neither sentence.
    assert buckets[1] == (4, 5, 6)


def test_a_token_straddling_a_boundary_goes_to_its_larger_overlap():
    # "b. Ccc" spans 1 char of sentence 0 (the '.') and 3 of sentence 1.
    text = "Ab. Ccc dd."
    spans = locate_sentences(text)
    tokens = ["A", "b. Ccc", " dd", "."]
    buckets = align_tokens_to_sentences(tokens, spans)
    assert buckets[0] == (0,)
    assert 1 in buckets[1]


def test_every_token_lands_in_at_most_one_sentence():
    spans = locate_sentences(REPORT)
    tokens = list(REPORT)  # one character per token: the finest possible split
    buckets = align_tokens_to_sentences(tokens, spans)
    seen: list[int] = []
    for bucket in buckets:
        seen.extend(bucket)
    assert len(seen) == len(set(seen))


def test_character_tokens_reconstruct_each_sentence_exactly():
    spans = locate_sentences(REPORT)
    tokens = list(REPORT)
    buckets = align_tokens_to_sentences(tokens, spans)
    for (sentence, _, _), bucket in zip(spans, buckets, strict=True):
        assert "".join(tokens[index] for index in bucket) == sentence


def test_zero_width_tokens_are_dropped_rather_than_assigned():
    text = "Aa bb."
    spans = locate_sentences(text)
    buckets = align_tokens_to_sentences(["Aa", "", " bb."], spans)
    assert buckets[0] == (0, 2)


def test_align_rejects_a_bare_string():
    with pytest.raises(TypeError, match="sequence of strings"):
        align_tokens_to_sentences("not a token list", locate_sentences(REPORT))


# --------------------------------------------------------------------------
# The labeler -- lexicon_v1, and never presented as anything else
# --------------------------------------------------------------------------


def test_the_labeler_reports_itself_as_lexicon_v1():
    assert LexiconSentenceLabeler().name == LEXICON_LABELER_NAME == "lexicon_v1"
    assert attribute_sentences(REPORT).labeler == "lexicon_v1"
    assert attribute_sentences(REPORT).to_dict()["labeler"] == "lexicon_v1"


def test_the_labeler_detects_negated_and_positive_findings():
    labels = LexiconSentenceLabeler().label("There is no pneumothorax.")
    assert labels == (SentenceLabel(finding="Pneumothorax", polarity="negative"),)
    labels = LexiconSentenceLabeler().label("Mild cardiomegaly is present.")
    assert labels == (SentenceLabel(finding="Cardiomegaly", polarity="positive"),)


def test_the_labeler_returns_nothing_for_an_unmatched_sentence():
    assert LexiconSentenceLabeler().label("Patient tolerated well.") == ()
    assert LexiconSentenceLabeler().label("   ") == ()


def test_a_custom_labeler_can_be_substituted():
    class StubLabeler:
        name = "stub_v0"

        def label(self, sentence: str) -> tuple[SentenceLabel, ...]:
            return (SentenceLabel(finding="Edema", polarity="uncertain"),)

    study = attribute_sentences(REPORT, labeler=StubLabeler())
    assert study.labeler == "stub_v0"
    assert study.parse_coverage == 1.0


def test_an_object_that_is_not_a_labeler_is_refused():
    with pytest.raises(TypeError, match="SentenceLabeler protocol"):
        attribute_sentences(REPORT, labeler=object())


# --------------------------------------------------------------------------
# Unlabelled sentences are kept, and flagged
# --------------------------------------------------------------------------


def test_unlabelled_sentences_are_kept_and_marked_not_spatially_meaningful():
    study = attribute_sentences(REPORT)
    assert study.num_sentences == 3  # nothing dropped
    assert [record.spatially_meaningful for record in study.sentences] == [True, True, False]
    assert study.unparsed == ("Patient tolerated well.",)


def test_parse_coverage_is_labelled_over_total():
    study = attribute_sentences(REPORT)
    assert study.parse_coverage == pytest.approx(2 / 3)
    assert study.num_labelled_sentences == 2


def test_parse_coverage_of_an_empty_report_is_zero_not_one():
    study = attribute_sentences("")
    assert study.num_sentences == 0
    assert study.parse_coverage == 0.0


def test_parse_coverage_is_one_when_every_sentence_is_labelled():
    study = attribute_sentences("There is no pneumothorax. Mild cardiomegaly is present.")
    assert study.parse_coverage == 1.0
    assert study.unparsed == ()


# --------------------------------------------------------------------------
# Dataset-level coverage
# --------------------------------------------------------------------------


def test_dataset_coverage_pools_by_sentence_not_by_study():
    # Study A: 1 sentence, labelled.        study coverage 1.0
    # Study B: 3 sentences, 1 labelled.     study coverage 1/3
    # Pooled: 2 of 4 = 0.5. Mean of study fractions: 2/3. They differ, which is
    # the whole reason the pooled figure is the one to quote.
    a = attribute_sentences("Mild cardiomegaly is present.")
    b = attribute_sentences("Mild cardiomegaly is present. Patient sat up. He walked out.")
    summary = dataset_parse_coverage([a, b])
    assert summary["num_studies"] == 2
    assert summary["num_sentences"] == 4
    assert summary["num_labelled_sentences"] == 2
    assert summary["parse_coverage"] == pytest.approx(0.5)
    assert summary["mean_study_parse_coverage"] == pytest.approx(2 / 3)


def test_dataset_coverage_of_no_studies_is_zero():
    summary = dataset_parse_coverage([])
    assert summary["parse_coverage"] == 0.0
    assert summary["num_sentences"] == 0


# --------------------------------------------------------------------------
# mean_token_nll
# --------------------------------------------------------------------------


def test_mean_token_nll_is_averaged_within_each_sentence():
    text = "Aa bb. Cc dd."
    tokens = ["Aa", " bb", ".", " ", "Cc", " dd", "."]
    nll = [1.0, 2.0, 3.0, 99.0, 10.0, 20.0, 30.0]
    study = attribute_sentences(text, token_texts=tokens, token_nll=nll)
    assert study.sentences[0].mean_token_nll == pytest.approx(2.0)  # (1+2+3)/3
    assert study.sentences[1].mean_token_nll == pytest.approx(20.0)  # (10+20+30)/3


def test_mean_token_nll_is_none_when_no_nll_is_supplied():
    # None means "not measured". 0.0 would mean "the model was certain".
    study = attribute_sentences(REPORT)
    assert all(record.mean_token_nll is None for record in study.sentences)


def test_mean_token_nll_is_none_for_a_sentence_with_no_tokens():
    study = attribute_sentences("Aa bb.", token_texts=[], token_nll=[])
    assert study.sentences[0].mean_token_nll is None


def test_a_token_nll_length_mismatch_is_refused():
    with pytest.raises(ValueError, match="misalignment"):
        attribute_sentences("Aa bb.", token_texts=["Aa", " bb."], token_nll=[1.0])


# --------------------------------------------------------------------------
# The serialised record
# --------------------------------------------------------------------------


def test_to_dict_is_json_serialisable_and_carries_coverage():
    import json

    text = "Aa bb. Mild cardiomegaly is present."
    tokens = ["Aa", " bb", ".", " Mild cardiomegaly is present."]
    study = attribute_sentences(text, token_texts=tokens, token_nll=[1.0, 2.0, 3.0, 4.0])
    payload = study.to_dict()
    encoded = json.dumps(payload, allow_nan=False)
    assert '"labeler": "lexicon_v1"' in encoded
    assert payload["parse_coverage"] == pytest.approx(0.5)
    assert payload["sentences"][1]["labels"] == [
        {"finding": "Cardiomegaly", "polarity": "positive"}
    ]
    assert payload["sentences"][0]["spatially_meaningful"] is False


def test_the_only_labeler_name_ever_emitted_is_on_the_allowlist():
    """The output must not imply a labeler this repository does not have.

    Written as an allowlist rather than as a search for a forbidden string, so
    that the name of the trained labeler this project does NOT implement never
    appears in the source either. Any new labeler must be added here
    deliberately, which is the point.
    """
    allowed = {LEXICON_LABELER_NAME}
    for text in (REPORT, "", "Mild cardiomegaly is present."):
        assert attribute_sentences(text).to_dict()["labeler"] in allowed
