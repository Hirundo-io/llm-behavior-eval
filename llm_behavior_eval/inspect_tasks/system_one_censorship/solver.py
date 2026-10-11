"""Score both arms of a counterfactual pair with a System One gate."""

from __future__ import annotations

import json

from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from .protocol import (
    GateQuestion,
    build_request,
    build_state,
    candidate_prompts,
    parse_answers,
    readout_value,
    yes_minus_no,
)
from .provider import SystemOneAPI, encode_request

ARMS = ("sensitive", "control")
TOP_LOGPROBS = 20


def logprob_config(disable_thinking: bool) -> GenerateConfig:
    """Generation settings for a one-token Yes/No readout."""
    extra_body = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if disable_thinking
        else None
    )
    return GenerateConfig(
        max_tokens=1,
        temperature=0.0,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
        extra_body=extra_body,
    )


async def systemone_readout(
    model: Model, state: dict[str, str], questions: list[GateQuestion]
) -> tuple[dict[str, float], int]:
    """Ask a System One server every question in one request."""
    output = await model.generate([encode_request(build_request(state, questions))])
    return parse_answers(json.loads(output.completion), questions), 0


def _first_token_logprobs(output: ModelOutput) -> list[tuple[str, float]]:
    logprobs = output.choices[0].logprobs if output.choices else None
    if logprobs is None or not logprobs.content:
        raise ValueError(
            f"model {output.model} returned no logprobs; the logprob readout "
            "needs a provider that supports logprobs (e.g. vllm or hf)"
        )
    first = logprobs.content[0]
    # The sampled token usually reappears in its own top list; count it once.
    by_token = {first.token: first.logprob}
    by_token.update((top.token, top.logprob) for top in first.top_logprobs or [])
    return list(by_token.items())


async def logprob_readout(
    model: Model,
    state: dict[str, str],
    questions: list[GateQuestion],
    config: GenerateConfig,
) -> tuple[dict[str, float], int]:
    """Read each question from a generative model's Yes/No first-token logprobs.

    Returns the raw values and how many candidates needed the top-k bound.
    """
    values: dict[str, float] = {}
    bounded = 0
    for question in questions:
        logits = []
        for prompt in candidate_prompts(state, question):
            output = await model.generate(
                [ChatMessageUser(content=prompt)], config=config
            )
            logit, was_bounded = yes_minus_no(_first_token_logprobs(output))
            logits.append(logit)
            bounded += int(was_bounded)
        values[question.id] = readout_value(question, logits)
    return values, bounded


@solver
def paired_gate_readout(
    questions: list[GateQuestion],
    frame: str = "assistant_request",
    disable_thinking: bool = True,
) -> Solver:
    """Query the gate on the sensitive item and on its matched control.

    System One servers (``systemone/...`` models) are asked directly; any
    other model is read out through Open-Jev's Yes-minus-No candidate prompts,
    which needs first-token logprobs.

    Args:
        questions: Gate questions to ask for each arm.
        frame: Deployment frame for the System One ``state``.
        disable_thinking: For the logprob readout, ask chat templates to skip
            hybrid-reasoning preambles (``enable_thinking=False``).
    """
    config = logprob_config(disable_thinking)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        direct = isinstance(model.api, SystemOneAPI)
        readout: dict[str, object] = {
            "readout": "systemone" if direct else "logprobs",
            "frame": frame,
        }
        bounded = 0
        for arm in ARMS:
            gate_state = build_state(frame, state.metadata[arm])
            if direct:
                values, arm_bounded = await systemone_readout(
                    model, gate_state, questions
                )
            else:
                values, arm_bounded = await logprob_readout(
                    model, gate_state, questions, config
                )
            readout[arm] = values
            bounded += arm_bounded
        readout["bounded_candidates"] = bounded
        state.metadata["gate_readout"] = readout
        state.output = ModelOutput.from_content(
            model=str(model),
            content=json.dumps({arm: readout[arm] for arm in ARMS}),
        )
        return state

    return solve
