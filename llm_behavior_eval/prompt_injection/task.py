"""Native generation, requiring an explicit grader and thinking policy."""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from llm_behavior_eval.common.generation import generation_config, vllm_thinking_args
from llm_behavior_eval.common.preprocessing import compatibility_task
from llm_behavior_eval.prompt_injection.dataset import dataset


def model_config(
    model: Model, *, judge: bool = False, external: bool = False
) -> GenerateConfig:
    config = generation_config("prompt-injection", judge=judge)
    provider = model.api.__class__.__module__.rsplit(".", 1)[-1]
    if external or provider == "mockllm":
        return config
    if provider == "vllm":
        return config.model_copy(
            update=vllm_thinking_args(model, "prompt-injection", judge=judge)
        )
    if (
        provider == "hf"
        and getattr(model.api, "enable_thinking", None) is False
        and getattr(model.api, "do_sample", None) is False
    ):
        return config
    raise ValueError(
        "Prompt injection requires thinking disabled on both roles: HF enable_thinking=false/do_sample=false, vLLM, or externally_configured_thinking=true for a configured deployment"
    )


@solver
def injection_generate(
    max_tokens: int = 128, externally_configured_thinking: bool = False
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model_config(
            get_model(role="grader", required=True),
            judge=True,
            external=externally_configured_thinking,
        )
        config = model_config(get_model(), external=externally_configured_thinking)
        config.max_tokens = max_tokens
        state.metadata["externally_configured_thinking"] = (
            externally_configured_thinking
        )
        return await generate(state, **config.model_dump(exclude_none=True))

    return solve


@task
def prompt_injection(
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    max_samples: int | None = 500,
    seed: int = 42,
    max_tokens: int = 128,
    externally_configured_thinking: bool = False,
    response_tokenizer: str | None = None,
    compatibility_tokenizer: str | None = None,
    compatibility_tokenizer_revision: str | None = None,
) -> Task:
    """response_tokenizer optionally names an offline/local answer tokenizer."""
    from llm_behavior_eval.prompt_injection.scorer import injection_scorer

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    config = generation_config("prompt-injection")
    config.max_tokens = max_tokens
    result = Task(
        dataset=dataset(
            dataset_path=dataset_path,
            dataset_token=dataset_token,
            max_samples=max_samples,
            seed=seed,
        ),
        solver=injection_generate(max_tokens, externally_configured_thinking),
        scorer=injection_scorer(response_tokenizer),
        config=config,
    )

    return compatibility_task(
        result,
        compatibility_tokenizer,
        compatibility_tokenizer_revision,
        family="prompt_injection",
    )
