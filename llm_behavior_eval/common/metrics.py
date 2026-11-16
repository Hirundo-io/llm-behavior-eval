"""Generic metric mechanics: epoch observations, rates and stop reasons.

Epoch layout: every metric a benchmark reports is declared
``@metric(scores="unreduced")``. Inspect's epoch reducers keep only the first
epoch's score metadata, and benchmark metrics read score metadata such as
subgroups (``field_rate(metadata_key=...)``). Unreduced metrics receive one
``SampleScore`` per sample per epoch, each with its own metadata, so
``--epochs N`` aggregates every epoch. When all of a scorer's metrics are
unreduced, Inspect writes exactly one ``EvalScore`` for that scorer
(``reducer`` is ``None``), keyed by the same metric names as a single-epoch run.
A scorer that mixes reduced and unreduced metrics is split into two
``EvalScore`` entries, so never mix them; the tests' ``reduced_metrics``
(``tests/inspect_helpers.py``) finds offenders. Inspect's own metrics
(``mean()``, ``aggregate()``, ``stderr()``, ...) default to the reduced view, so
benchmarks call them inside an unreduced adapter, as ``field_rate`` does.

Unscored observations: a scorer that cannot score an observation (a skipped
answer, an unparsable grader reply, an unknown stop reason) returns
``Score.unscored(reason=...)``, whose value is NaN. Inspect drops root-NaN
scores before it calls any metric of that scorer and records the counts as the
``EvalScore``'s ``scored_samples`` and ``unscored_samples``. Each scorer's
denominator is therefore independent: completion diagnostics live in their own
scorer (``completion.completion_scorer``), so an ungradable answer still
counts towards ``incomplete_response_rate``.

Rates and means pool every (sample, epoch) observation; with the same number of
epochs per sample and no exclusions that equals the mean of per-sample means.
Counts count observations. A sample's epochs are not independent, so the
standard error of a pooled sample-epoch mean clusters them: use Inspect's
``stderr(cluster=<key>)`` inside an unreduced adapter, where ``<key>`` is a
**sample metadata** field holding a stable sample id (native clustering reads
``SampleScore.sample_metadata``, not ``sample_id``). An unweighted standard
error over per-sample means is a different estimator, and not equivalent when
exclusions leave samples with unequal epoch counts. ``scores_by_sample`` stays
for benchmark-specific statistics that need the groups themselves.

Empty denominators yield NaN.
"""

from typing import Any, cast

from inspect_ai.scorer import (
    Metric,
    MetricProtocol,
    SampleScore,
    aggregate,
    mean,
    metric,
)

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
    """Mean scalar or dict-field score over the scored epoch observations.

    A thin unreduced adapter over Inspect's native metrics. Denominator
    contract (pinned by ``test_field_rate_denominator_contract``):

    1. Unscored observations never count. A scorer that cannot grade an
       observation returns ``Score.unscored(...)`` (value NaN), and Inspect
       drops every root-NaN score before it calls any metric of that scorer.
    2. With ``metadata_key`` set, only observations whose **score** metadata
       maps it to ``metadata_value`` count (a missing key is ``None``).
    3. The remaining observations go to native ``mean()``, or with ``field``
       set to ``aggregate(field, mean(), on_missing="skip")``. So ``mean``'s
       ``value_to_float`` converts each value: numbers and ``bool`` (1/0) as
       is, grade letters ``"C"``/``"I"``/``"P"``/``"N"`` to 1/0/0.5/0,
       ``"yes"``/``"true"`` and ``"no"``/``"false"`` to 1 and 0, numeric
       strings to their number; anything else (a dict read without
       ``field``, a list, other text) counts as 0 with a warning. With
       ``field`` set, a missing or ``None`` key and a per-key NaN are
       skipped, and a non-dict score value raises ``ValueError``.

    An empty denominator yields NaN (native ``mean()`` would report 0).
    Values are not checked for finiteness, so an ``inf`` that reaches the
    metric propagates. Every (sample, epoch) observation counts once.

    The subgroup filter stays here rather than in native ``grouped()``:
    ``grouped`` reads ``SampleScore.sample_metadata`` and reports every
    group, while this metric reads one subgroup from the score's own
    metadata, which a scorer may set per observation.

    This replaces an earlier contract in which skipped and parse-error
    observations were scored 0 with ``skipped``/``parse_error`` metadata that
    this metric excluded, and non-numeric values were dropped rather than
    converted. Scorers now return ``Score.unscored`` for them instead.

    Args:
        field: Key to read when score values are dicts. ``None`` reads the
            scalar value.
        metadata_key: When set, only scores whose metadata maps this key to
            ``metadata_value`` count.
        metadata_value: The subgroup value ``metadata_key`` must have.

    Returns:
        A metric that averages the counted (sample, epoch) values, or NaN when
        none remain.
    """
    rate = cast(
        "MetricProtocol",
        mean() if field is None else aggregate(field, mean(), on_missing="skip"),
    )

    def compute(scores: list[SampleScore]) -> float:
        if metadata_key is not None:
            scores = [
                sample
                for sample in scores
                if (sample.score.metadata or {}).get(metadata_key) == metadata_value
            ]
        if not scores:
            return float("nan")
        return cast("float", rate(scores))

    return compute


@metric(scores="unreduced")
def incomplete_response_rate() -> Metric:
    """Share of answers that ended at the token budget or the context window.

    The metric of ``completion.completion_scorer``, whose value is 1 for an
    answer stopped by ``max_tokens`` or ``model_length``, 0 for another known
    stop reason, and unscored when the stop reason is unknown. It is native
    ``mean()`` over those values, unreduced, with NaN for an empty
    denominator: a run in which no stop reason is known (any run on
    Inspect's ``hf/`` provider) reports NaN rather than 0.

    Attach it only to ``completion_scorer``: it averages score values and
    no longer reads ``stop_reason`` metadata.

    Returns:
        A metric over the (sample, epoch) observations with a known stop
        reason.
    """
    rate = cast("MetricProtocol", mean())

    def compute(scores: list[SampleScore]) -> float:
        return cast("float", rate(scores)) if scores else float("nan")

    return compute


def scores_by_sample(scores: list[SampleScore]) -> list[list[SampleScore]]:
    """Group a metric's epoch observations by sample.

    For benchmark-specific statistics that need each sample's epochs as a
    group. A standard error of the pooled mean uses native
    ``stderr(cluster=...)`` instead (see the module docstring).

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
