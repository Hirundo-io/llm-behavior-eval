"""BBQ solver chain. It sets no GenerateConfig for the model under test."""

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
def bbq_user_message() -> Solver:
    """Send the question with the legacy trailing newline (``f"{question}\\n"``)."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = state.input_text + "\n"
        return state

    return solve


def bbq_solver() -> list[Solver]:
    return [system_message(SYSTEM_MESSAGE), bbq_user_message(), generate()]
