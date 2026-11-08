"""Native Inspect Med-Hallu; no default grader and no legacy engine calls."""

from inspect_ai import Task, task
from inspect_ai.solver import system_message

from llm_behavior_eval.common.generation import generation_config
from llm_behavior_eval.hallucination.scorer import hallucination_scorer
from llm_behavior_eval.hallucination.solver import hallucination_generate
from llm_behavior_eval.medhallu.dataset import dataset
from llm_behavior_eval.medhallu.prompts import SYSTEM_MESSAGE


@task
def medhallu(
    max_samples: int | None = 500,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    thinking_disabled: bool = False,
    grader_thinking_disabled: bool = False,
) -> Task:
    """API flags attest to externally configured thinking-off, independently per role."""
    return Task(
        dataset=dataset(
            max_samples, dataset_path=dataset_path, dataset_token=dataset_token
        ),
        solver=[
            system_message(SYSTEM_MESSAGE),
            hallucination_generate(
                thinking_disabled=thinking_disabled,
                grader_thinking_disabled=grader_thinking_disabled,
            ),
        ],
        scorer=hallucination_scorer(thinking_disabled=grader_thinking_disabled),
        config=generation_config("hallucination"),
    )
