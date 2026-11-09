"""CCPC answer prompt. Sets no GenerateConfig for the model under test."""

from inspect_ai.model import get_model
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

from llm_behavior_eval.common.judge import GRADER_ROLE

PROMPT_PARITY = (
    "unresolved: legacy chat-template decoding and 1024-token input truncation"
)


@solver
def ccpc_prompt() -> Solver:
    """Legacy answer prompt: the question plus a trailing newline, no system prompt.

    Resolves the required grader role first, so a run without
    ``--model-role grader=...`` fails before any answer is generated.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        get_model(role=GRADER_ROLE, required=True)
        state.user_prompt.text = state.input_text + "\n"
        state.metadata["prompt_parity"] = PROMPT_PARITY
        return state

    return solve


def solver() -> list[Solver]:
    return [ccpc_prompt(), generate()]
