from inspect_ai import Task, task

from llm_behavior_eval.common.generation import generation_config
from llm_behavior_eval.common.judge import THINKING_END, THINKING_START
from llm_behavior_eval.orbench.dataset import dataset
from llm_behavior_eval.orbench.scorer import scorer
from llm_behavior_eval.refusal.generation import refusal_generate


@task
def orbench(
    max_samples: int | None = 500,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    seed: int = 42,
    thinking: bool = True,
    answer_thinking_configured: bool = False,
    judge_thinking_configured: bool = False,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Task:
    return Task(
        dataset=dataset(
            max_samples,
            dataset_path=dataset_path,
            dataset_token=dataset_token,
            seed=seed,
        ),
        solver=refusal_generate(
            thinking=thinking,
            answer_thinking_configured=answer_thinking_configured,
            judge_thinking_configured=judge_thinking_configured,
        ),
        scorer=scorer(
            judge_thinking_configured=judge_thinking_configured,
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        ),
        config=generation_config("refusal"),
    )
