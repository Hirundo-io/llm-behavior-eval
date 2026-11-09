from inspect_ai import Task, task
from inspect_ai.model import ModelOutput, get_model

from llm_behavior_eval.inspect_template.dataset import dataset
from llm_behavior_eval.inspect_template.scorer import scorer
from llm_behavior_eval.inspect_template.solver import solver


@task
def inspect_template() -> Task:
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
        scorer=scorer(),
    )
