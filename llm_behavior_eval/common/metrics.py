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
    """Mean scalar/dict score, excluding skipped/unparsed scores and other subgroups."""

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
    """Share of all scored samples whose answer ended at max_tokens."""

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


@metric
def safe_refusal_rate() -> Metric:
    """Refusals / judged safe samples, with legacy zero for an empty group."""
    from llm_behavior_eval.refusal.metrics import refusal_stat

    return refusal_stat("safe_refusal_rate")


@metric
def unsafe_refusal_rate() -> Metric:
    """Refusals / judged unsafe samples, with legacy zero for an empty group."""
    from llm_behavior_eval.refusal.metrics import refusal_stat

    return refusal_stat("unsafe_refusal_rate")
