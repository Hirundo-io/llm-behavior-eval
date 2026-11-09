"""Fail closed when provider thinking defaults cannot be verified."""

from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from llm_behavior_eval.common.generation import generation_config, thinking_args


def refusal_config(
    model: Model,
    *,
    judge: bool = False,
    thinking: bool = True,
    externally_configured: bool = False,
) -> GenerateConfig:
    config = generation_config("refusal", judge=judge)
    enabled = thinking and not judge
    provider = str(model).split("/")[0]
    if provider == "vllm":
        config.extra_body = thinking_args(
            "refusal", provider="vllm", judge=not enabled
        )["extra_body"]
    elif provider == "hf":
        if (
            getattr(model.api, "enable_thinking", None) is not enabled
            or getattr(model.api, "do_sample", None) is not False
        ):
            raise ValueError(
                f"HF refusal requires model args enable_thinking={enabled}, do_sample=False on each role"
            )
    elif provider != "mockllm" and not externally_configured:
        raise ValueError(
            "Thinking control is unsupported for this provider. Configure the endpoint explicitly and set answer_thinking_configured/judge_thinking_configured=True."
        )
    return config


@solver
def refusal_generate(
    *,
    thinking: bool = True,
    answer_thinking_configured: bool = False,
    judge_thinking_configured: bool = False,
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        grader = get_model(role="grader", required=True)
        refusal_config(
            grader, judge=True, externally_configured=judge_thinking_configured
        )
        config = refusal_config(
            get_model(),
            thinking=thinking,
            externally_configured=answer_thinking_configured,
        )
        state.user_prompt.text = state.input_text + "\n"
        state.metadata.update(
            thinking=thinking,
            judge_thinking=False,
            prompt_parity="native chat; legacy decoded template and 1024-token truncation unresolved",
        )
        return await generate(state, **config.model_dump(exclude_none=True))

    return solve
