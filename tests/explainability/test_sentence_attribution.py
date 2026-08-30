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


def test_each_labeler_reports_its_own_name():
    assert LexiconSentenceLabeler().name == LEXICON_LABELER_NAME == "lexicon_v1"
    assert (
        attribute_sentences(REPORT, labeler=LexiconSentenceLabeler()).labeler
        == "lexicon_v1"
    )


def test_the_default_labeler_is_v2_and_says_so_in_the_record():
    """Changed 2026-08-30. Every artifact records which labeler produced it, so
    an older output never becomes ambiguous -- it says lexicon_v1."""
    from training.explainability.sentence_attribution import DEFAULT_LABELER_NAME

    assert DEFAULT_LABELER_NAME == "lexicon_v2"
    assert attribute_sentences(REPORT).labeler == "lexicon_v2"
    assert attribute_sentences(REPORT).to_dict()["labeler"] == "lexicon_v2"


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
    assert '"labeler": "lexicon_v2"' in encoded
    assert payload["parse_coverage"] == pytest.approx(0.5)
    assert payload["sentences"][1]["labels"] == [
        {"finding": "Cardiomegaly", "polarity": "positive", "tier": "chexpert_14"}
    ]
    assert payload["sentences"][0]["spatially_meaningful"] is False


def test_the_only_labeler_name_ever_emitted_is_on_the_allowlist():
    """The output must not imply a labeler this repository does not have.

    Written as an allowlist rather than as a search for a forbidden string, so
    that the name of the trained labeler this project does NOT implement never
    appears in the source either. Any new labeler must be added here
    deliberately, which is the point.
    """
    allowed = {LEXICON_LABELER_NAME, EXTENDED_LABELER_NAME}
    for text in (REPORT, "", "Mild cardiomegaly is present."):
        assert attribute_sentences(text).to_dict()["labeler"] in allowed
        assert (
            attribute_sentences(text, labeler=LexiconSentenceLabeler())
            .to_dict()["labeler"]
            in allowed
        )


# --------------------------------------------------------------------------
# lexicon_v2: the 14 with better synonyms, plus findings outside that taxonomy
# --------------------------------------------------------------------------

from training.explainability.sentence_attribution import (  # noqa: E402
    EXTENDED_FINDINGS,
    EXTENDED_LABELER_NAME,
    LABELERS,
    TIER_CHEXPERT_14,
    TIER_EXTENDED,
    ExtendedLexiconSentenceLabeler,
    build_extended_synonyms,
    label_tier,
)


def test_safety_claims_is_left_alone():
    """The 14 map one-to-one onto Stage 1's head and safety/ reconciles against it.

    Extending that lexicon in place would create claims with no prediction to
    check them against, turning a verification pipeline into a description
    pipeline. The extension lives in this layer instead.
    """
    from safety.claims import ABNORMALITY_SYNONYMS

    assert len(ABNORMALITY_SYNONYMS) == 14
    assert not set(EXTENDED_FINDINGS) & set(ABNORMALITY_SYNONYMS)


def test_the_extended_lexicon_keeps_all_14_and_adds_more():
    merged = build_extended_synonyms()
    from safety.claims import ABNORMALITY_SYNONYMS

    assert set(ABNORMALITY_SYNONYMS) <= set(merged)
    assert len(merged) == 14 + len(EXTENDED_FINDINGS)


def test_the_base_synonyms_are_taken_by_reference_not_copied():
    """A change to the repository's lexicon must propagate, not diverge."""
    from safety.claims import ABNORMALITY_SYNONYMS

    merged = build_extended_synonyms()
    for finding, terms in ABNORMALITY_SYNONYMS.items():
        assert set(terms) <= set(merged[finding]), finding


def test_every_label_reports_its_tier():
    assert label_tier("Cardiomegaly") == TIER_CHEXPERT_14
    assert label_tier("Degenerative Change") == TIER_EXTENDED
    for finding in EXTENDED_FINDINGS:
        assert label_tier(finding) == TIER_EXTENDED


def test_v1_labels_also_carry_a_tier_and_it_is_always_chexpert_14():
    for label in LexiconSentenceLabeler().label("There is no pneumothorax."):
        assert label.tier == TIER_CHEXPERT_14


def test_the_tier_survives_into_the_serialised_record():
    """Whoever reads the JSONL must be able to tell the two apart."""
    study = attribute_sentences(
        "Degenerative changes of the spine.", labeler=ExtendedLexiconSentenceLabeler()
    )
    payload = study.to_dict()["sentences"][0]["labels"][0]
    assert payload["tier"] == TIER_EXTENDED
    assert payload["finding"] == "Degenerative Change"


@pytest.mark.parametrize(
    ("text", "finding"),
    [
        ("Degenerative changes of the thoracic spine.", "Degenerative Change"),
        ("Mild thoracic scoliosis.", "Spinal Deformity"),
        ("The aorta is tortuous.", "Aortic Abnormality"),
        ("A hiatal hernia is again seen.", "Hernia"),
        ("Median sternotomy wires are intact.", "Postsurgical Change"),
        ("Emphysematous changes are present.", "Hyperinflation"),
        ("Biapical scarring is unchanged.", "Scarring"),
        ("Pectus excavatum deformity.", "Chest Wall Deformity"),
        ("Bowel gas is seen below the diaphragm.", "Upper Abdomen"),
    ],
)
def test_extended_findings_are_recognised(text, finding):
    labels = ExtendedLexiconSentenceLabeler().label(text)
    assert finding in {label.finding for label in labels}, text


@pytest.mark.parametrize(
    ("text", "finding"),
    [
        ("The heart size is enlarged.", "Cardiomegaly"),
        ("Low lung volumes are noted.", "Atelectasis"),
        ("Patchy infiltrate at the right base.", "Lung Opacity"),
        ("Blunting of the left costophrenic angle.", "Pleural Effusion"),
        ("The tube tip is in good position.", "Support Devices"),
    ],
)
def test_wording_v1_missed_is_now_caught_and_stays_in_the_14(text, finding):
    labels = ExtendedLexiconSentenceLabeler().label(text)
    matched = [label for label in labels if label.finding == finding]
    assert matched, text
    assert matched[0].tier == TIER_CHEXPERT_14


def test_v1_did_not_catch_those_which_is_why_v2_exists():
    v1 = LexiconSentenceLabeler()
    for text in ("The heart size is enlarged.", "Low lung volumes are noted."):
        assert not v1.label(text), text


def test_polarity_still_works_on_extended_findings():
    labels = ExtendedLexiconSentenceLabeler().label("No hiatal hernia is seen.")
    assert labels[0].finding == "Hernia"
    assert labels[0].polarity == "negative"


def test_word_boundaries_stop_line_matching_midline():
    labels = ExtendedLexiconSentenceLabeler().label("The trachea is midline.")
    assert "Support Devices" not in {label.finding for label in labels}


def test_both_labelers_are_registered_and_named():
    assert LABELERS[LEXICON_LABELER_NAME] is LexiconSentenceLabeler
    assert LABELERS[EXTENDED_LABELER_NAME] is ExtendedLexiconSentenceLabeler
    assert ExtendedLexiconSentenceLabeler().name == "lexicon_v2"


def test_v1_remains_available_so_earlier_runs_stay_reproducible():
    """v2 is the default; v1 must still be reachable, or the recorded n=1,513
    val run could never be reproduced."""
    from training.explainability.sentence_attribution import LABELERS

    assert LABELERS["lexicon_v1"] is LexiconSentenceLabeler
    study = attribute_sentences(REPORT, labeler=LexiconSentenceLabeler())
    assert study.labeler == "lexicon_v1"
    # and v1 must still behave as v1: it does NOT know the extended findings
    assert not LexiconSentenceLabeler().label("Degenerative changes of the spine.")
