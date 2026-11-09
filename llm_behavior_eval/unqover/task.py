"""Native UNQOVER; select each of the four ambiguous splits explicitly."""

from inspect_ai import Task, task
from inspect_ai.solver import system_message

from llm_behavior_eval.bias.scorer import bias_metrics, bias_scorer
from llm_behavior_eval.bias.solver import bias_generate
from llm_behavior_eval.common.generation import generation_config
from llm_behavior_eval.common.templates import FREE_TEXT_SYSTEM_MESSAGE
from llm_behavior_eval.unqover.dataset import dataset


@task
def unqover(
    bias_type: str = "gender",
    kind: str = "bias",
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    max_samples: int | None = 500,
    seed: int = 42,
    externally_configured_thinking: bool = False,
) -> Task:
    """Primary error=1-accuracy (fraction, displayed as percent; lower is better).

    Other metrics: accuracy/incomplete_response_rate (fractions),
    empty_responses (count), stereotype_bias (signed index, optional column).
    One split per task; explicit grader required, recommended gemma-4-12B-it.

    externally_configured_thinking asserts BOTH deployments disable thinking;
    use it for independently configured OpenAI-compatible endpoints.
    """
    data = dataset(
        bias_type,
        kind,
        dataset_path=dataset_path,
        dataset_token=dataset_token,
        max_samples=max_samples,
        seed=seed,
    )
    has_stereotype = (
        bool((data[0].metadata or {})["has_stereotype"]) if len(data) else False
    )
    return Task(
        dataset=data,
        solver=[
            system_message(FREE_TEXT_SYSTEM_MESSAGE["content"]),
            bias_generate(externally_configured_thinking),
        ],
        scorer=bias_scorer(),
        metrics=[bias_metrics(kind, has_stereotype)],
        config=generation_config("bias"),
    )
