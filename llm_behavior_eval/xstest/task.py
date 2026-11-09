from inspect_ai import Task, task

from llm_behavior_eval.common.generation import generation_config
from llm_behavior_eval.common.judge import THINKING_END, THINKING_START
from llm_behavior_eval.common.preprocessing import compatibility_task
from llm_behavior_eval.refusal.generation import refusal_generate
from llm_behavior_eval.xstest.dataset import dataset
from llm_behavior_eval.xstest.scorer import scorer


@task
def xstest(
    max_samples: int | None = 500,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    thinking: bool = True,
    answer_thinking_configured: bool = False,
    judge_thinking_configured: bool = False,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
    compatibility_tokenizer: str | None = None,
    compatibility_tokenizer_revision: str | None = None,
) -> Task:
    result = Task(
        dataset=dataset(
            max_samples, dataset_path=dataset_path, dataset_token=dataset_token
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

    return compatibility_task(
        result,
        compatibility_tokenizer,
        compatibility_tokenizer_revision,
        family="xstest",
    )
