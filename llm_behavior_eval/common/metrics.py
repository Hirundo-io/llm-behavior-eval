"""Rates with explicit eligible denominators; empty groups yield NaN."""

from typing import Any

from inspect_ai.scorer import Metric, SampleScore, metric


@metric
def field_rate(
    field: str | None = None,
    *,
    metadata_key: str | None = None,
    metadata_value: Any = None,
) -> Metric:
    """Mean scalar/dict score, excluding skipped/unparsed scores and other subgroups.

    Args:
        field: Key to read when a score value is a dict. ``None`` reads scalar
            values.
        metadata_key: When set, only scores whose metadata maps this key to
            ``metadata_value`` count.
        metadata_value: The subgroup value ``metadata_key`` must have.

    Returns:
        A metric that averages the eligible numeric values, or NaN when none
        remain.
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


@metric
def incomplete_response_rate() -> Metric:
    """Share of all scored samples whose answer ended at max_tokens.

    Returns:
        A metric over the ``stop_reason`` score metadata, NaN when there are no
        scores.
    """

    def compute(scores: list[SampleScore]) -> float:
        return (
            sum(
                (s.score.metadata or {}).get("stop_reason") == "max_tokens"
                for s in scores
            )
            / len(scores)
            if scores
            else float("nan")
        )

    return compute
