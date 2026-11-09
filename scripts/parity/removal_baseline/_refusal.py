"""Run the legacy refusal ``_grade_impl`` on fixed inputs (frozen checkout only)."""

from types import SimpleNamespace
from typing import Any


def grade(evaluator_module: Any, inputs: dict[str, list[Any]], slug: str) -> dict:
    """Run the legacy ``_grade_impl`` with a stub judge returning ``raw``."""
    labels, raw = inputs["labels"], inputs["raw"]
    parse = (
        evaluator_module.parse_or_bench_judgment
        if slug == "orbench"
        else evaluator_module.parse_xstest_judgment
    )
    judge_stops = inputs.get("judge_stops", ["stop"] * len(raw))
    captured: dict[str, Any] = {}
    evaluator = object.__new__(evaluator_module.FreeTextRefusalEvaluator)
    evaluator.num_samples = len(labels)
    evaluator.eval_config = SimpleNamespace(
        thinking_start_token="<think>",  # noqa: S106
        thinking_end_token="</think>",  # noqa: S106
        exclude_thinking_trace_for_judge=True,
    )
    evaluator.get_dataset_slug = lambda: slug
    evaluator._grade_batch = lambda *a: ([parse(t) for t in raw], raw, judge_stops)
    evaluator._save_refusal_results = lambda responses, metrics: captured.update(
        responses=responses, metrics=metrics
    )
    evaluator._grade_impl(
        [
            evaluator_module._RefusalGenerationRecord(
                input_texts=["q"] * len(labels),
                expected_labels=labels,
                answers=inputs["answers"],
                finish_reasons=inputs["stops"],
            )
        ],
        object(),
    )
    return {"inputs": inputs, **captured}
