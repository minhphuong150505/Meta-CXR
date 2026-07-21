"""Stage-2 report generation metrics.

Two defects in ``training/train_eval_figure9_llm_variants_200.py::compute_nlg``
drive the design here:

1. ``except Exception: rouge_l = 0.0``. A dependency failure became a *score of
   zero on the results table*, indistinguishable from a model that genuinely
   scored zero. Nothing in this module converts an exception into a number. A
   metric either produces a value or is reported as unavailable with the reason.
2. Corpus-level only. Without per-sample scores there is no error analysis, no
   bootstrap and no way to find the worst outputs.

Implementation provenance
-------------------------
BLEU-1..4 and ROUGE-L are implemented natively here: both are unambiguous
algorithms, and a native implementation keeps the metric core runnable in the
CPU environment, which has no nltk. The exact variant is documented on each
function and unit-tested against hand-computed values.

METEOR, CIDEr and BERTScore are **not** reimplemented -- they have genuine
implementation variance, so they are delegated to their reference packages and
reported as unavailable when those are missing. Every result records which
implementation and version produced it, in ``MetricSuite.provenance``.

Normalisation
-------------
Lowercasing and whitespace collapsing only, with punctuation split off as
separate tokens. Clinically load-bearing words -- ``no``, ``not``, ``without``,
``new``, ``increased``, ``decreased``, ``left``, ``right``, ``stable`` -- are
**never** removed. There is no stopword list, because in radiology "no" is the
difference between two opposite reports.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

BLEU = "bleu"
ROUGE = "rouge"
METEOR = "meteor"
CIDER = "cider"
BERTSCORE = "bertscore"

LEXICAL_METRICS = (BLEU, ROUGE, METEOR, CIDER, BERTSCORE)
DEFAULT_METRICS = (BLEU, ROUGE)

#: Words that must survive normalisation. Documented as a guard: a future
#: "cleanup" that drops them inverts clinical meaning.
PROTECTED_TOKENS = frozenset(
    {
        "no", "not", "without", "never", "none", "negative",
        "new", "increased", "decreased", "improved", "worsened",
        "left", "right", "bilateral", "upper", "lower",
        "stable", "unchanged", "resolved", "persistent",
        "mild", "moderate", "severe", "small", "large",
    }
)

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


class MissingMetricDependency(RuntimeError):
    """A metric was requested but its reference package is not installed."""

    def __init__(self, metric: str, package: str, install: str):
        self.metric = metric
        self.package = package
        self.install = install
        super().__init__(
            f"metric {metric!r} needs the {package!r} package, which is not "
            f"installed. Install it with: {install}. The evaluator does not "
            "substitute a different implementation and does not report 0.0."
        )


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace. Nothing is deleted."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokenize(text: str) -> list[str]:
    """Word and punctuation tokens, lowercased.

    Punctuation is kept as separate tokens rather than stripped, so sentence
    boundaries survive into ROUGE-L's longest-common-subsequence.
    """
    return _TOKEN_PATTERN.findall(normalize(text))


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    predictions: list[str], references: list[str], max_n: int = 4
) -> dict[str, float]:
    """Corpus BLEU-1..``max_n``, single reference per prediction.

    Standard corpus BLEU: modified n-gram precisions are pooled over the whole
    corpus before the geometric mean, with the standard brevity penalty against
    the summed reference length.

    Smoothing: when a corpus-level n-gram numerator is 0 the geometric mean would
    be 0 for every higher order, which hides all signal. Add-1 smoothing is
    applied to orders ``n >= 2`` only (NLTK's ``method1``); BLEU-1 is left
    unsmoothed because a zero there is genuinely meaningful.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"got {len(predictions)} predictions for {len(references)} references"
        )

    tokenized_predictions = [tokenize(p) for p in predictions]
    tokenized_references = [tokenize(r) for r in references]

    prediction_length = sum(len(t) for t in tokenized_predictions)
    reference_length = sum(len(t) for t in tokenized_references)

    if prediction_length == 0:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}

    brevity = (
        1.0
        if prediction_length > reference_length
        else math.exp(1 - reference_length / prediction_length)
    )

    numerators = [0] * (max_n + 1)
    denominators = [0] * (max_n + 1)
    for prediction, reference in zip(tokenized_predictions, tokenized_references):
        for n in range(1, max_n + 1):
            prediction_ngrams = _ngrams(prediction, n)
            reference_ngrams = _ngrams(reference, n)
            overlap = prediction_ngrams & reference_ngrams
            numerators[n] += sum(overlap.values())
            denominators[n] += max(sum(prediction_ngrams.values()), 0)

    scores: dict[str, float] = {}
    log_precisions: list[float] = []
    for n in range(1, max_n + 1):
        numerator = numerators[n]
        denominator = denominators[n]
        if denominator == 0:
            precision = 0.0
        elif numerator == 0 and n > 1:
            precision = 1.0 / (2 * denominator)  # add-1 style smoothing
        else:
            precision = numerator / denominator

        log_precisions.append(math.log(precision) if precision > 0 else -math.inf)
        if any(math.isinf(value) for value in log_precisions):
            scores[f"bleu_{n}"] = 0.0
        else:
            scores[f"bleu_{n}"] = brevity * math.exp(
                sum(log_precisions) / len(log_precisions)
            )
    return scores


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence, O(len(a) * len(b)) time."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[index] + 1)
            else:
                current.append(max(current[index], previous[index + 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str, beta: float = 1.2) -> float:
    """Sentence-level ROUGE-L F-measure.

    ``beta=1.2`` matches the original ROUGE package and pycocoevalcap, which
    weight recall slightly above precision. Returns 0.0 when either side is
    empty -- that is a real score for an empty report, not an error.
    """
    prediction_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0

    lcs = _lcs_length(prediction_tokens, reference_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    beta_squared = beta**2
    return ((1 + beta_squared) * precision * recall) / (recall + beta_squared * precision)


def rouge_n(prediction: str, reference: str, n: int) -> float:
    """Sentence-level ROUGE-N F1."""
    prediction_ngrams = _ngrams(tokenize(prediction), n)
    reference_ngrams = _ngrams(tokenize(reference), n)
    if not prediction_ngrams or not reference_ngrams:
        return 0.0
    overlap = sum((prediction_ngrams & reference_ngrams).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(prediction_ngrams.values())
    recall = overlap / sum(reference_ngrams.values())
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------
# Optional reference-package backends
# --------------------------------------------------------------------------


def _meteor(predictions: list[str], references: list[str]) -> list[float]:
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError as exc:
        raise MissingMetricDependency(
            METEOR, "nltk", "pip install nltk && python -m nltk.downloader wordnet"
        ) from exc
    return [
        float(meteor_score([tokenize(reference)], tokenize(prediction)))
        for prediction, reference in zip(predictions, references)
    ]


def _cider(predictions: list[str], references: list[str]) -> tuple[float, list[float]]:
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError as exc:
        raise MissingMetricDependency(
            CIDER, "pycocoevalcap", "pip install pycocoevalcap"
        ) from exc
    gts = {i: [normalize(r)] for i, r in enumerate(references)}
    res = {i: [normalize(p)] for i, p in enumerate(predictions)}
    score, per_sample = Cider().compute_score(gts, res)
    return float(score), [float(v) for v in per_sample]


def _bertscore(
    predictions: list[str], references: list[str], model_type: str, device: str
) -> dict[str, list[float]]:
    try:
        from bert_score import score as bert_score_fn
    except ImportError as exc:
        raise MissingMetricDependency(
            BERTSCORE, "bert_score", "pip install bert-score"
        ) from exc
    precision, recall, f1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        model_type=model_type,
        rescale_with_baseline=False,
        verbose=False,
        device=device,
    )
    return {
        "precision": [float(v) for v in precision],
        "recall": [float(v) for v in recall],
        "f1": [float(v) for v in f1],
    }


# --------------------------------------------------------------------------
# Suite
# --------------------------------------------------------------------------


@dataclass
class MetricSuite:
    """Corpus scores, per-sample scores, and what produced them."""

    corpus: dict[str, float] = field(default_factory=dict)
    per_sample: dict[str, list[float]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "provenance": self.provenance,
            "unavailable": self.unavailable,
        }


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - absence is the expected case
        return "not installed"


def compute_generation_metrics(
    predictions: list[str],
    references: list[str],
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    bertscore_model: str = "distilbert-base-uncased",
    bertscore_device: str = "cpu",
    strict: bool = False,
) -> MetricSuite:
    """Compute the requested lexical metrics.

    Parameters
    ----------
    strict:
        When True a missing dependency raises. When False (default) the metric
        is recorded in ``unavailable`` with the install command and every other
        metric still runs -- but it is **never** recorded as a score of 0.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"got {len(predictions)} predictions for {len(references)} references; "
            "generation metrics require one reference per prediction"
        )
    if not predictions:
        raise ValueError("no predictions supplied")

    unknown = set(metrics) - set(LEXICAL_METRICS)
    if unknown:
        raise ValueError(
            f"unknown metric(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(LEXICAL_METRICS)}"
        )

    suite = MetricSuite()
    suite.provenance["normalization"] = {
        "lowercase": True,
        "collapse_whitespace": True,
        "strip_punctuation": False,
        "remove_stopwords": False,
        "protected_tokens_preserved": True,
    }
    suite.provenance["num_samples"] = len(predictions)

    empty_predictions = sum(1 for p in predictions if not tokenize(p))
    suite.provenance["empty_predictions"] = empty_predictions
    if empty_predictions:
        logger.warning(
            "%d/%d predictions are empty after tokenisation; they score 0 on "
            "lexical metrics",
            empty_predictions,
            len(predictions),
        )

    def record_unavailable(exc: MissingMetricDependency) -> None:
        if strict:
            raise exc
        logger.warning("%s", exc)
        suite.unavailable[exc.metric] = str(exc)

    if BLEU in metrics:
        suite.corpus.update(corpus_bleu(predictions, references))
        suite.provenance[BLEU] = {
            "implementation": "training.evaluation.generation_metrics.corpus_bleu",
            "variant": "corpus BLEU, single reference, add-1 smoothing for n>=2",
        }

    if ROUGE in metrics:
        per_sample_l = [rouge_l(p, r) for p, r in zip(predictions, references)]
        per_sample_1 = [rouge_n(p, r, 1) for p, r in zip(predictions, references)]
        per_sample_2 = [rouge_n(p, r, 2) for p, r in zip(predictions, references)]
        suite.per_sample["rouge_l"] = per_sample_l
        suite.per_sample["rouge_1"] = per_sample_1
        suite.per_sample["rouge_2"] = per_sample_2
        suite.corpus["rouge_l"] = sum(per_sample_l) / len(per_sample_l)
        suite.corpus["rouge_1"] = sum(per_sample_1) / len(per_sample_1)
        suite.corpus["rouge_2"] = sum(per_sample_2) / len(per_sample_2)
        suite.provenance[ROUGE] = {
            "implementation": "training.evaluation.generation_metrics.rouge_l/rouge_n",
            "variant": "sentence-level F-measure, beta=1.2, averaged over samples",
        }

    if METEOR in metrics:
        try:
            values = _meteor(predictions, references)
        except MissingMetricDependency as exc:
            record_unavailable(exc)
        else:
            suite.per_sample["meteor"] = values
            suite.corpus["meteor"] = sum(values) / len(values)
            suite.provenance[METEOR] = {
                "implementation": "nltk.translate.meteor_score",
                "version": _package_version("nltk"),
            }

    if CIDER in metrics:
        try:
            corpus_score, values = _cider(predictions, references)
        except MissingMetricDependency as exc:
            record_unavailable(exc)
        else:
            suite.per_sample["cider"] = values
            suite.corpus["cider"] = corpus_score
            suite.provenance[CIDER] = {
                "implementation": "pycocoevalcap.cider.Cider",
                "version": _package_version("pycocoevalcap"),
            }

    if BERTSCORE in metrics:
        try:
            values = _bertscore(
                predictions, references, bertscore_model, bertscore_device
            )
        except MissingMetricDependency as exc:
            record_unavailable(exc)
        else:
            for key, series in values.items():
                suite.per_sample[f"bertscore_{key}"] = series
                suite.corpus[f"bertscore_{key}"] = sum(series) / len(series)
            suite.provenance[BERTSCORE] = {
                "implementation": "bert_score.score",
                "version": _package_version("bert-score"),
                "model_type": bertscore_model,
                "rescale_with_baseline": False,
                "note": (
                    "absolute BERTScore values sit in a narrow high band even for "
                    "unrelated text; compare against a baseline, not against 0"
                ),
            }

    return suite
