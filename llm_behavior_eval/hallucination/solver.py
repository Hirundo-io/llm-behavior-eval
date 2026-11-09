"""Native generation and fail-closed thinking configuration for both roles."""

from inspect_ai.model import GenerateConfig, Model, ModelName, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from llm_behavior_eval.common.generation import generation_config, thinking_args


def configured_generation(
    model: Model, *, judge: bool = False, thinking_disabled: bool = False
) -> GenerateConfig:
    """Apply vLLM controls; validate HF controls; require API endpoint attestation."""
    config = generation_config("hallucination", judge=judge)
    provider = ModelName(model).api
    if provider == "vllm":
        config = config.model_copy(
            update=thinking_args("hallucination", provider="vllm", judge=judge)
        )
    elif provider == "hf":
        if (
            getattr(model.api, "enable_thinking", None) is not False
            or getattr(model.api, "do_sample", None) is not False
        ):
            raise ValueError(
                "HaluEval HF roles require enable_thinking=false and do_sample=false model args"
            )
    elif provider != "mockllm" and not thinking_disabled:
        raise ValueError(
            "Thinking-off control is unsupported for this provider; configure the endpoint explicitly and set thinking_disabled=True for this role"
        )
    return config


@solver
def hallucination_generate(
    *, thinking_disabled: bool = False, grader_thinking_disabled: bool = False
) -> Solver:
    """Check both roles before generation and preserve the legacy user newline."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        grader = get_model(role="grader", required=True)
        configured_generation(
            grader, judge=True, thinking_disabled=grader_thinking_disabled
        )
        config = configured_generation(get_model(), thinking_disabled=thinking_disabled)
        state.user_prompt.text = f"{state.input_text}\n"
        state.metadata["thinking_disabled_externally"] = thinking_disabled
        state.metadata["grader_thinking_disabled_externally"] = grader_thinking_disabled
        return await generate(state, **config.model_dump(exclude_none=True))

    return solve
