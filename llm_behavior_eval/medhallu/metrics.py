"""Med-Hallu legacy metrics; every scored sample stays in the denominator.

accuracy: percentage, higher is better, ``100 * (1 - INCORRECT / all samples)``.
empty_responses: count of NOT_ATTEMPTED grades (including skips and parse failures).
incomplete_response_rate: from ``common.metrics``, fraction ending at max_tokens.

Inspect keeps one global metric registry keyed by package and function name, so
two benchmarks that both registered an ``accuracy`` metric would share the name
``llm_behavior_eval/accuracy``: the last one imported would replace the other
wherever Inspect rebuilds metrics from a log by name (for example
``recompute_metrics`` or ``inspect score``). Each metric here is therefore
registered under a ``medhallu_`` name and returns a one-entry mapping, which
Inspect reports under the mapping's key: the log still shows ``accuracy`` and
``empty_responses``.
"""

from inspect_ai.scorer import Metric, SampleScore, metric


def _grades(scores: list[SampleScore]) -> list[object]:
    return [(s.score.metadata or {}).get("grade") for s in scores]


@metric
def medhallu_accuracy() -> Metric:
    """Reported as ``accuracy``."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        if not scores:
            return {"accuracy": float("nan")}
        incorrect = _grades(scores).count("INCORRECT")
        return {"accuracy": 100 * (1 - incorrect / len(scores))}

    return compute


@metric
def medhallu_empty_responses() -> Metric:
    """Reported as ``empty_responses``."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        return {"empty_responses": _grades(scores).count("NOT_ATTEMPTED")}

    return compute
