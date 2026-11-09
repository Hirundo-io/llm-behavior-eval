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
``EvalScore`` entries, so never mix them; ``reduced_metrics`` finds offenders.

Rates and means pool every (sample, epoch) observation; with the same number of
epochs per sample and no exclusions that equals the mean of per-sample means.
Counts count observations. Statistics that assume independent observations,
such as a standard error, must first group a sample's epochs with
``scores_by_sample``.

Empty denominators yield NaN.
"""

from collections.abc import Callable, Iterator
from typing import Any

from inspect_ai import Task
from inspect_ai.scorer import Metric, SampleScore, metric
from inspect_ai.util import registry_info

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

    Args:
        field: Key to read when a score value is a dict. ``None`` reads scalar
            values.
        metadata_key: When set, only scores whose metadata maps this key to
            ``metadata_value`` count.
        metadata_value: The subgroup value ``metadata_key`` must have.

    Returns:
        A metric that averages the numeric values of every (sample, epoch)
        score, excluding skipped and unparsed scores and other subgroups, or
        NaN when none remain.
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


def reduced_metrics(target: Task | Callable[..., Any]) -> list[str]:
    """Metrics that would receive epoch-reduced scores.

    For a task, checks the metrics Inspect computes for each of its scorers
    (``Task(metrics=...)`` when given, otherwise each scorer's own). For a
    scorer, or a ``@scorer`` factory, checks its declared metrics. Inspect
    built-ins such as ``accuracy()`` and ``stderr()`` are reduced. Benchmark
    tests assert that this is empty for their tasks.

    Args:
        target: A constructed task, a scorer or a ``@scorer`` factory.

    Returns:
        Sorted registry names of the metrics not declared
        ``@metric(scores="unreduced")``.
    """
    scorers = (target.scorer or []) if isinstance(target, Task) else [target]
    names: set[str] = set()
    for scorer in scorers:
        # Inspect stores a scorer's metrics, or the task's metrics= override,
        # under this registry metadata key (scorer_metrics() reads it).
        declared = registry_info(scorer).metadata.get("metrics", [])
        for item in _flatten_metrics(declared):
            info = registry_info(item)
            if info.metadata.get("scores") != EPOCH_SCORES:
                names.add(info.name)
    return sorted(names)


def _flatten_metrics(
    metrics: list[Metric | dict[str, list[Metric]]] | dict[str, list[Metric]],
) -> Iterator[Metric]:
    """Every metric in Inspect's ``metrics`` shapes (a list, a dict, or both)."""
    groups = [metrics] if isinstance(metrics, dict) else metrics
    for group in groups:
        if isinstance(group, dict):
            for members in group.values():
                yield from members
        else:
            yield group
