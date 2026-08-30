"""Split a generated report into sentences and attach labels, NLL and coverage.

Standard library plus this repository's existing ``safety.claims`` lexicon.
No torch, no model, no tokenizer -- token identities arrive as decoded strings
and per-token NLL arrives as a plain sequence of floats, so the whole module
runs anywhere the test suite does.

The labeler, stated plainly
===========================

This repository implements no trained clinical labeler. ``training/evaluation/
clinical.py`` deliberately raises rather than returning a fabricated score, and
that policy applies here too. Sentence labels therefore come from
:class:`LexiconSentenceLabeler`, a deterministic synonym-and-cue matcher over
the repository's 14 abnormality labels, and it is reported under the name
``lexicon_v1``. It is NOT a trained labeler and must never be presented as one.

Its limits are load-bearing, so they are measured rather than assumed:
``parse_coverage`` is the fraction of sentences that produced at least one
label, and it is carried on every study record and aggregated across the
dataset. A run whose coverage is 0.3 has labelled three sentences in ten, and
every sentence-level conclusion drawn from it is bounded by that number. Quote
it beside any result.

Sentences that produce no label are KEPT, never dropped. They still receive an
attribution map -- the model did generate them, and where it looked is still a
fact -- but they carry ``spatially_meaningful=False`` so a reader cannot mistake
an unlabelled sentence for a grounded finding.

:class:`SentenceLabeler` is a Protocol so a trained labeler can be dropped in
later. No such adapter is implemented here, on purpose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

try:  # ``python script.py`` from inside training/
    from safety.claims import (
        ABNORMALITY_SYNONYMS,
        LexiconClaimParser,
        split_sentences,
        unparsed_sentences,
    )
except ImportError:  # pragma: no cover - exercised only outside the repo root
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from safety.claims import (  # noqa: F401
        ABNORMALITY_SYNONYMS,
        LexiconClaimParser,
        split_sentences,
        unparsed_sentences,
    )

LEXICON_LABELER_NAME = "lexicon_v1"
EXTENDED_LABELER_NAME = "lexicon_v2"

#: Which taxonomy a label belongs to. This distinction is load-bearing and must
#: survive into every output: Stage 1 predicts the 14 and nothing predicts the
#: rest, so a sentence labelled ``Aortic Abnormality`` has no classifier to be
#: checked against, while one labelled ``Cardiomegaly`` does. Collapsing the two
#: would let an unverifiable finding be read as a verified one -- the same
#: reason ``explanation_metrics`` refuses to average its lung and bbox
#: populations together.
TIER_CHEXPERT_14 = "chexpert_14"
TIER_EXTENDED = "extended"

#: Wording for the existing 14 that ``lexicon_v1`` does not match. Measured, not
#: guessed: these are the terms that dominate the unlabelled sentences on the
#: val split -- lung(91), volumes(75), enlarged(40), heart(36), tube(30).
#: They stay inside the 14, so nothing about the Stage-1 correspondence changes.
ADDITIONAL_CHEXPERT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Cardiomegaly": (
        "heart size", "cardiac silhouette", "heart is enlarged",
        "cardiac contour", "heart borders",
    ),
    "Atelectasis": (
        "low lung volumes", "lung volumes are low", "low volumes",
        "decreased lung volumes", "collapse", "collapsed",
    ),
    "Edema": ("pulmonary vascular congestion", "vascular prominence", "vascularity"),
    "Lung Opacity": ("infiltrate", "infiltrates", "opacification", "density", "densities"),
    "Pleural Effusion": ("blunting", "costophrenic angle", "pleural fluid"),
    "Support Devices": ("tube", "line", "leads", "device", "devices", "port", "stent"),
}

#: Findings a radiologist genuinely reports that the CheXpert 14 has no slot
#: for. Derived from the `outside_14` bucket, which is 10.8% of unlabelled
#: sentences on the val split.
#:
#: ⚠ NOTHING PREDICTS THESE. They are labels for what a sentence says, not
#: predictions to check it against, and they carry ``TIER_EXTENDED`` for that
#: reason. Adding them raises coverage; it does not add verification.
#:
#: ⚠ This list is a proposal and needs a clinician's sign-off before any
#: published claim, exactly like the kappa table in `CLAUDE.md`.
EXTENDED_FINDINGS: dict[str, tuple[str, ...]] = {
    "Degenerative Change": (
        "degenerative", "degenerative changes", "spondylosis", "osteophyte",
        "osteophytes", "disc space narrowing", "osteopenia", "osteoporosis",
    ),
    "Spinal Deformity": ("scoliosis", "scoliotic", "kyphosis", "kyphotic"),
    "Aortic Abnormality": (
        "tortuous aorta", "aortic tortuosity", "tortuosity", "tortuous",
        "unfolded aorta",
        "ectatic aorta", "aortic calcification", "atherosclerotic",
        "atherosclerosis", "aortic knob",
    ),
    "Hernia": ("hiatal hernia", "hernia", "herniation"),
    "Postsurgical Change": (
        "sternotomy", "sternotomy wires", "surgical clips", "clips",
        "post-surgical", "postsurgical", "cabg", "resection", "lobectomy",
    ),
    "Hyperinflation": (
        "hyperinflation", "hyperinflated", "hyperexpanded", "emphysema",
        "emphysematous", "copd",
    ),
    "Scarring": ("scarring", "scar", "fibrosis", "fibrotic", "granuloma", "granulomas"),
    "Chest Wall Deformity": ("pectus excavatum", "pectus", "chest wall deformity"),
    "Upper Abdomen": ("bowel gas", "gastric distension", "free air", "pneumoperitoneum"),
}


@dataclass(frozen=True)
class SentenceLabel:
    """One finding asserted by one sentence, with its polarity and its tier."""

    finding: str
    polarity: str
    tier: str = TIER_CHEXPERT_14

    def to_dict(self) -> dict[str, str]:
        return {"finding": self.finding, "polarity": self.polarity, "tier": self.tier}


@runtime_checkable
class SentenceLabeler(Protocol):
    """Assigns findings to a single sentence.

    ``name`` is written verbatim into every output record, so a result can
    always be traced to the labeler that produced it. Implement this to swap in
    a trained labeler; nothing else in this module needs to change.
    """

    name: str

    def label(self, sentence: str) -> tuple[SentenceLabel, ...]:
        ...


def build_extended_synonyms() -> dict[str, tuple[str, ...]]:
    """The 14 (augmented) plus the extra findings, as one lexicon.

    ``ABNORMALITY_SYNONYMS`` is taken by reference rather than copied, so a
    change to the repository's own lexicon propagates here instead of quietly
    diverging.
    """
    merged: dict[str, tuple[str, ...]] = {}
    for finding, terms in ABNORMALITY_SYNONYMS.items():
        merged[finding] = tuple(terms) + ADDITIONAL_CHEXPERT_SYNONYMS.get(finding, ())
    for finding, terms in EXTENDED_FINDINGS.items():
        if finding in merged:
            raise ValueError(f"{finding!r} is already one of the 14 labels")
        merged[finding] = tuple(terms)
    return merged


def label_tier(finding: str) -> str:
    """Which taxonomy a finding belongs to. Never guess this downstream."""
    return TIER_CHEXPERT_14 if finding in ABNORMALITY_SYNONYMS else TIER_EXTENDED


class LexiconSentenceLabeler:
    """Deterministic synonym + polarity-cue labeler over 14 abnormality labels.

    A thin adapter over :class:`safety.claims.LexiconClaimParser`, which already
    implements the matching and the clause-aware polarity detection. Nothing is
    reimplemented here -- this class exists to expose one sentence at a time and
    to carry the ``lexicon_v1`` name.
    """

    name = LEXICON_LABELER_NAME

    def __init__(self, parser: LexiconClaimParser | None = None):
        self._parser = parser if parser is not None else LexiconClaimParser()

    def label(self, sentence: str) -> tuple[SentenceLabel, ...]:
        text = str(sentence or "").strip()
        if not text:
            return ()
        # Parsing one sentence at a time keeps ``sentence_index`` irrelevant and
        # makes the adapter independent of how the caller split the report.
        return tuple(
            SentenceLabel(
                finding=claim.finding,
                polarity=claim.polarity,
                tier=label_tier(claim.finding),
            )
            for claim in self._parser.parse(text)
        )


class ExtendedLexiconSentenceLabeler(LexiconSentenceLabeler):
    """The 14 with better synonyms, plus findings outside that taxonomy.

    ⚠ **`safety/claims.py` is deliberately NOT modified.** Its 14 labels
    correspond one-to-one with Stage 1's classification head, and
    `safety/pipeline.py` reconciles each claim against that classifier. Adding
    labels there would produce claims with no prediction to check them against,
    which is a quiet way to turn a verification pipeline into a description
    pipeline. The extension lives here, where nothing is being verified.

    Everything this labeler emits carries a ``tier``. Sentences labelled
    ``TIER_EXTENDED`` gained coverage, not verification -- keep the two apart
    in any report.
    """

    name = EXTENDED_LABELER_NAME

    def __init__(self, parser: LexiconClaimParser | None = None):
        super().__init__(parser or LexiconClaimParser(build_extended_synonyms()))


LABELERS = {
    LEXICON_LABELER_NAME: LexiconSentenceLabeler,
    EXTENDED_LABELER_NAME: ExtendedLexiconSentenceLabeler,
}

#: The default since 2026-08-30. Measured on the full val split, v2 raises
#: coverage 0.483 -> 0.648, cutting `missed_14` from 217 sentences to 68 and
#: `outside_14` from 437 to 105.
#:
#: ⚠ Only 61% of that gain is verifiable: of the 1,285 newly labelled
#: sentences, 782 are CheXpert-14 wording that Stage 1 predicts and 473 are
#: findings nothing predicts. Read the `tier` field before treating a label as
#: a checked claim.
#:
#: ⚠ Changing this changed what a default invocation produces. The n=1,513 val
#: run recorded before this date used v1 and is reproduced with
#: ``--labeler lexicon_v1``; every artifact records the labeler it used, so no
#: existing output becomes ambiguous.
DEFAULT_LABELER_NAME = EXTENDED_LABELER_NAME


@dataclass(frozen=True)
class SentenceRecord:
    """One sentence of a generated report and everything known about it."""

    index: int
    text: str
    char_start: int
    char_end: int
    token_indices: tuple[int, ...]
    labels: tuple[SentenceLabel, ...]
    spatially_meaningful: bool
    mean_token_nll: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_indices": list(self.token_indices),
            "labels": [label.to_dict() for label in self.labels],
            "spatially_meaningful": self.spatially_meaningful,
            "mean_token_nll": self.mean_token_nll,
        }


@dataclass(frozen=True)
class StudyAttribution:
    """Every sentence of one study, plus this study's parse coverage."""

    sentences: tuple[SentenceRecord, ...]
    labeler: str
    parse_coverage: float
    unparsed: tuple[str, ...] = field(default_factory=tuple)

    @property
    def num_sentences(self) -> int:
        return len(self.sentences)

    @property
    def num_labelled_sentences(self) -> int:
        return sum(1 for sentence in self.sentences if sentence.labels)

    def to_dict(self) -> dict[str, object]:
        return {
            "labeler": self.labeler,
            "parse_coverage": self.parse_coverage,
            "num_sentences": self.num_sentences,
            "num_labelled_sentences": self.num_labelled_sentences,
            "unparsed_sentences": list(self.unparsed),
            "sentences": [sentence.to_dict() for sentence in self.sentences],
        }


def dataset_parse_coverage(studies: Sequence[StudyAttribution]) -> dict[str, object]:
    """Pool coverage over studies by SENTENCE, not by study.

    A mean of per-study fractions would let a one-sentence study weigh as much
    as a twelve-sentence one. The pooled figure is the one to quote; the mean of
    study fractions is reported beside it only because the two diverging is
    itself informative.
    """
    total_sentences = sum(study.num_sentences for study in studies)
    total_labelled = sum(study.num_labelled_sentences for study in studies)
    per_study = [study.parse_coverage for study in studies if study.num_sentences]
    return {
        "num_studies": len(studies),
        "num_sentences": total_sentences,
        "num_labelled_sentences": total_labelled,
        "parse_coverage": (total_labelled / total_sentences) if total_sentences else 0.0,
        "mean_study_parse_coverage": (sum(per_study) / len(per_study)) if per_study else 0.0,
    }


def locate_sentences(text: str) -> list[tuple[str, int, int]]:
    """Sentences with their character spans in ``text``.

    :func:`safety.claims.split_sentences` strips each fragment, which loses the
    offsets needed to map tokens onto sentences. Rather than reimplement the
    split -- which would let two splitters drift -- this calls it and then walks
    the original string to recover each span. Pinned to it by
    ``tests/explainability/test_sentence_attribution.py``.
    """
    source = str(text or "")
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for sentence in split_sentences(source):
        start = source.find(sentence, cursor)
        if start < 0:
            raise ValueError(
                "a split sentence was not found in the source text; the splitter "
                "and this offset walk have diverged"
            )
        spans.append((sentence, start, start + len(sentence)))
        cursor = start + len(sentence)
    return spans


def align_tokens_to_sentences(
    token_texts: Sequence[str],
    sentence_spans: Sequence[tuple[str, int, int]],
) -> list[tuple[int, ...]]:
    """Assign each decoded token to at most one sentence, by character overlap.

    Tokens are assumed to concatenate to the text the spans were computed from,
    which is true of any subword tokenizer's decoded pieces. A token that
    straddles a boundary goes to the sentence it overlaps most; ties go to the
    earlier sentence, so the assignment is deterministic. A token overlapping no
    sentence (inter-sentence whitespace) is assigned to none.
    """
    if not isinstance(token_texts, Sequence) or isinstance(token_texts, (str, bytes)):
        raise TypeError("token_texts must be a sequence of strings")

    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in token_texts:
        length = len(str(piece))
        offsets.append((cursor, cursor + length))
        cursor += length

    buckets: list[list[int]] = [[] for _ in sentence_spans]
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue  # a zero-width piece belongs nowhere
        best_sentence = None
        best_overlap = 0
        for sentence_index, (_, start, end) in enumerate(sentence_spans):
            overlap = min(token_end, end) - max(token_start, start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence_index
        if best_sentence is not None:
            buckets[best_sentence].append(token_index)
    return [tuple(bucket) for bucket in buckets]


def _mean_token_nll(
    token_indices: Sequence[int],
    token_nll: Sequence[float] | None,
) -> float | None:
    """Mean NLL over a sentence's tokens, or ``None`` when it is unknown.

    ``None`` means "not measured" and is deliberately distinct from ``0.0``,
    which would mean "the model was certain" -- the same distinction
    ``safety.claims.Claim`` makes for its scores.
    """
    if token_nll is None or not token_indices:
        return None
    values = [float(token_nll[index]) for index in token_indices]
    return sum(values) / len(values)


def attribute_sentences(
    text: str,
    *,
    token_texts: Sequence[str] | None = None,
    token_nll: Sequence[float] | None = None,
    labeler: SentenceLabeler | None = None,
) -> StudyAttribution:
    """Build the per-sentence record for one generated report.

    ``token_texts`` are the decoded pieces of the generated tokens, in order,
    and ``token_nll`` their per-token negative log-likelihoods from the
    teacher-forced pass. Both are optional: without them the sentences, labels
    and coverage are still produced, and ``token_indices`` is empty with
    ``mean_token_nll`` ``None``.
    """
    active = labeler if labeler is not None else LABELERS[DEFAULT_LABELER_NAME]()
    if not isinstance(active, SentenceLabeler):
        raise TypeError("labeler must implement the SentenceLabeler protocol")

    source = str(text or "")
    spans = locate_sentences(source)

    if token_texts is None:
        buckets: list[tuple[int, ...]] = [() for _ in spans]
    else:
        if token_nll is not None and len(token_nll) != len(token_texts):
            raise ValueError(
                f"token_nll has {len(token_nll)} values but there are "
                f"{len(token_texts)} tokens; a misalignment here silently "
                "attributes one sentence's uncertainty to another"
            )
        buckets = align_tokens_to_sentences(token_texts, spans)

    records: list[SentenceRecord] = []
    for index, ((sentence, start, end), token_indices) in enumerate(
        # strict: one bucket per sentence by construction. If that ever stops
        # holding, it is the silent misalignment this module exists to avoid.
        zip(spans, buckets, strict=True)
    ):
        labels = active.label(sentence)
        records.append(
            SentenceRecord(
                index=index,
                text=sentence,
                char_start=start,
                char_end=end,
                token_indices=token_indices,
                labels=labels,
                # An unlabelled sentence keeps its map but is flagged: the model
                # did look somewhere, and that is a fact worth recording, but it
                # is not a grounded finding.
                spatially_meaningful=bool(labels),
                mean_token_nll=_mean_token_nll(token_indices, token_nll),
            )
        )

    labelled = sum(1 for record in records if record.labels)
    coverage = (labelled / len(records)) if records else 0.0
    return StudyAttribution(
        sentences=tuple(records),
        labeler=active.name,
        parse_coverage=coverage,
        unparsed=tuple(record.text for record in records if not record.labels),
    )
