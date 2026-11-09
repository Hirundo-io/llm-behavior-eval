from inspect_ai import Task, task
from inspect_ai.model import ModelOutput, get_model

from llm_behavior_eval.common.judge import THINKING_END, THINKING_START
from llm_behavior_eval.inspect_template.dataset import dataset
from llm_behavior_eval.inspect_template.scorer import scorer
from llm_behavior_eval.inspect_template.solver import solver


@task
def inspect_template(
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Task:
    """Reference benchmark wiring.

    Args:
        exclude_thinking_trace_for_judge: Strip the answer's inline thinking
            trace before the grader sees it (``common.judge.THINKING_TASK_ARGS``).
        thinking_start_token: Marker that opens the trace.
        thinking_end_token: Marker that closes the trace.
    """
    # Wiring demo only: a mock grader that always answers SCORE: 1. Real
    # benchmarks bind no grader and require --model-role grader=...
    # No `config=`: inference settings come from the user, never the task.
    return Task(
        model_roles={
            "grader": get_model(
                "mockllm/model",
                custom_outputs=lambda *args: ModelOutput.from_content(
                    "mockllm/model", "SCORE: 1"
                ),
            )
        },
        dataset=dataset(),
        solver=solver(),
        scorer=scorer(
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        ),
    )
