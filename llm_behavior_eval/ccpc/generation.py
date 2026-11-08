"""Explicit provider thinking controls and native generation."""

from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from llm_behavior_eval.common.generation import generation_config, thinking_args

PROMPT_PARITY = (
    "unresolved: legacy chat-template decoding and 1024-token input truncation"
)


def ccpc_config(
    model: Model, *, judge: bool = False, externally_configured: bool = False
) -> GenerateConfig:
    config = generation_config("censorship", judge=judge)
    provider = str(model).split("/")[0]
    if provider == "vllm":
        config.extra_body = thinking_args("censorship", provider="vllm", judge=judge)[
            "extra_body"
        ]
    elif provider == "hf":
        if (
            getattr(model.api, "enable_thinking", None) is not False
            or getattr(model.api, "do_sample", None) is not False
        ):
            raise ValueError(
                "CCPC HF requires enable_thinking=False and do_sample=False on both roles"
            )
    elif provider != "mockllm" and not externally_configured:
        raise ValueError(
            "Unsupported thinking control: configure endpoint thinking off and explicitly set answer_thinking_configured/judge_thinking_configured"
        )
    return config


@solver
def ccpc_generate(
    *, answer_thinking_configured: bool = False, judge_thinking_configured: bool = False
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        ccpc_config(
            get_model(role="grader", required=True),
            judge=True,
            externally_configured=judge_thinking_configured,
        )
        config = ccpc_config(
            get_model(), externally_configured=answer_thinking_configured
        )
        state.user_prompt.text = state.input_text + "\n"
        state.metadata.update(
            prompt_parity=PROMPT_PARITY,
            enable_thinking=False,
            judge_thinking=False,
            answer_config=config.model_dump(exclude_none=True),
        )
        return await generate(state, **config.model_dump(exclude_none=True))

    return solve
