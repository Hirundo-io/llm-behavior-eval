"""Bias provider policy: explicitly reject unsupported thinking controls."""

from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from llm_behavior_eval.common.generation import generation_config, vllm_thinking_args


def bias_config(model: Model, *, judge: bool = False) -> GenerateConfig:
    config = generation_config("bias", judge=judge)
    provider = model.api.__class__.__module__.rsplit(".", 1)[-1]
    if provider == "mockllm":
        return config
    if provider == "vllm":
        return config.model_copy(update=vllm_thinking_args(model, "bias", judge=judge))
    if provider == "hf":
        # HF consumes these arguments when constructing its tokenizer/generator.
        if (
            getattr(model.api, "enable_thinking", None) is not False
            or getattr(model.api, "do_sample", None) is not False
        ):
            raise ValueError(
                "Bias HF requires enable_thinking=false and do_sample=false model args on both roles"
            )
        return config
    if provider == "openai_compatible":
        # OpenAI-compatible endpoints may ignore chat_template_kwargs; require
        # an explicit task-level deployment assertion before permitting them.
        raise ValueError(
            "OpenAI-compatible thinking policy requires externally configured mode"
        )
    raise ValueError(
        f"Bias thinking-off unsupported for {provider}; use externally configured mode only when the deployment disables thinking"
    )


@solver
def bias_generate(externally_configured_thinking: bool = False) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        grader = get_model(role="grader", required=True)
        if not externally_configured_thinking:
            bias_config(grader, judge=True)
            config = bias_config(get_model())
        else:
            config = generation_config("bias")
        state.metadata["externally_configured_thinking"] = (
            externally_configured_thinking
        )
        state.user_prompt.text = state.input_text + "\n"
        return await generate(state, **config.model_dump(exclude_none=True))

    return solve
