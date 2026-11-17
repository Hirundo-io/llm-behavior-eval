"""Med-Hallu solver. It sets no GenerateConfig for the model under test."""

from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    generate,
    solver,
    system_message,
)

from .prompts import SYSTEM_MESSAGE


@solver
def medhallu_user_prompt() -> Solver:
    """Append the trailing newline the legacy engine added to the user turn."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = f"{state.input_text}\n"
        return state

    return solve


def medhallu_solver() -> list[Solver]:
    return [system_message(SYSTEM_MESSAGE), medhallu_user_prompt(), generate()]
