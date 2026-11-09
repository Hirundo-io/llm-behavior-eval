from inspect_ai import Task, task
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import generate, system_message

from llm_behavior_eval.common.generation import generation_config
from llm_behavior_eval.common.templates import FREE_TEXT_SYSTEM_MESSAGE
from llm_behavior_eval.inspect_template.dataset import dataset
from llm_behavior_eval.inspect_template.scorer import scorer


@task
def inspect_template(max_samples: int | None = 500) -> Task:
    return Task(
        model_roles={
            "grader": get_model(
                "mockllm/model",
                custom_outputs=lambda *args: ModelOutput.from_content(
                    "mockllm/model", "SCORE: 1"
                ),
            )
        },
        dataset=dataset(max_samples),
        solver=[system_message(FREE_TEXT_SYSTEM_MESSAGE["content"]), generate()],
        scorer=scorer(),
        config=generation_config("bias"),
    )
