"""Generic metric mechanics: epoch observations, rates and stop reasons.

Epoch layout: every metric a benchmark reports is declared
``@metric(scores="unreduced")``. Inspect's epoch reducers keep only the first
epoch's score metadata, and the metrics here (and most benchmark metrics) read
metadata such as ``skipped``, ``parse_error`` and ``stop_reason``. Unreduced
metrics receive one ``SampleScore`` per sample per epoch, each with its own
metadata, so ``--epochs N`` aggregates every epoch. When all of a scorer's
metrics are unreduced, Inspect writes exactly one ``EvalScore`` for that scorer
(``reducer`` is ``None``), keyed by the same metric names as a single-epoch run.
A scorer that mixes reduced and unreduced metrics is split into two
``EvalScore`` entries, so never mix them; the tests' ``reduced_metrics``
(``tests/inspect_helpers.py``) finds offenders.

Rates and means pool every (sample, epoch) observation; with the same number of
epochs per sample and no exclusions that equals the mean of per-sample means.
Counts count observations. Statistics that assume independent observations,
such as a standard error, must first group a sample's epochs with
``scores_by_sample``.

Empty denominators yield NaN.
"""

from typing import Any

from inspect_ai.scorer import Metric, SampleScore, metric

EPOCH_SCORES = "unreduced"
"""The ``scores`` mode every benchmark metric declares: ``@metric(scores=...)``."""

INCOMPLETE_STOP_REASONS = frozenset({"max_tokens", "model_length"})
"""Stop reasons of an answer cut short by the token budget or the context window."""

UNKNOWN_STOP_REASON = "unknown"
"""Inspect's default ``stop_reason``, kept by providers that do not report why
generation ended. Inspect's ``hf/`` provider reports it for every answer."""


@metric(scores="unreduced")
def field_rate(
    field: str | None = None,
    *,
    metadata_key: str | None = None,
    metadata_value: Any = None,
) -> Metric:
    """Mean scalar or dict-field score over the eligible epoch observations.

    Denominator contract (pinned by ``test_field_rate_denominator_contract``).
    Every (sample, epoch) score is one observation; one is left out of both
    numerator and denominator when, checked in this order:

    1. its metadata has a truthy ``skipped`` (``judge_scorer(only_stop=True)``
       skipping an unfinished answer);
    2. its metadata has a truthy ``parse_error`` (an unclosed grader trace, an
       unmatched, unparsable or non-finite verdict);
    3. ``metadata_key`` is set and its metadata does not map that key to
       ``metadata_value`` (a missing key counts as ``None``);
    4. the value read is not an ``int`` or ``float``: a dict score without
       ``field``, a dict read with ``field=None``, a string, ``None`` or a
       list. With ``field`` set, a scalar score value is read as is.

    Everything else counts, including ``bool`` values (as 1 or 0) and value 0.
    Values are not checked for finiteness: a ``nan`` or ``inf`` that reaches
    this metric propagates into the mean (``judge_scorer`` already turns
    non-finite verdicts into parse errors). The placeholder value 0 of skipped
    and parse-error scores is never averaged. An empty denominator yields NaN.

    Args:
        field: Key to read when a score value is a dict. ``None`` reads scalar
            values.
        metadata_key: When set, only scores whose metadata maps this key to
            ``metadata_value`` count.
        metadata_value: The subgroup value ``metadata_key`` must have.

    Returns:
        A metric that averages the eligible (sample, epoch) values, or NaN
        when none remain.
    """

    def compute(scores: list[SampleScore]) -> float:
        values = []
        for sample in scores:
            score = sample.score
            meta = score.metadata or {}
            if meta.get("skipped") or meta.get("parse_error"):
                continue
            if metadata_key is not None and meta.get(metadata_key) != metadata_value:
                continue
            value = (
                score.value.get(field)
                if isinstance(score.value, dict) and field
                else score.value
            )
            if isinstance(value, (int, float)):
                values.append(float(value))
        return sum(values) / len(values) if values else float("nan")

    return compute


@metric(scores="unreduced")
def incomplete_response_rate() -> Metric:
    """Share of answers that ended at the token budget or the context window.

    Returns:
        A metric over the (sample, epoch) scores whose ``stop_reason`` metadata
        is known. An answer stopped by ``max_tokens`` or ``model_length`` is
        incomplete. Answers whose provider reported ``unknown``, or whose score
        records no stop reason, cannot be classified and are left out of the
        denominator, so a run in which no
        stop reason is known (any run on Inspect's ``hf/`` provider) reports
        NaN rather than 0.
    """

    def compute(scores: list[SampleScore]) -> float:
        reasons = [
            reason
            for reason in ((s.score.metadata or {}).get("stop_reason") for s in scores)
            if reason is not None and reason != UNKNOWN_STOP_REASON
        ]
        if not reasons:
            return float("nan")
        incomplete = sum(reason in INCOMPLETE_STOP_REASONS for reason in reasons)
        return incomplete / len(reasons)

    return compute


def scores_by_sample(scores: list[SampleScore]) -> list[list[SampleScore]]:
    """Group a metric's epoch observations by sample.

    Args:
        scores: The ``scores`` an unreduced metric receives: one per sample per
            epoch.

    Returns:
        One list per sample, in first-seen order, holding that sample's epoch
        scores in the order received. Scores without a ``sample_id`` are each
        their own group. With one epoch every group has one score.
    """
    groups: dict[Any, list[SampleScore]] = {}
    for index, sample in enumerate(scores):
        key = ("id", sample.sample_id) if sample.sample_id is not None else index
        groups.setdefault(key, []).append(sample)
    return list(groups.values())
