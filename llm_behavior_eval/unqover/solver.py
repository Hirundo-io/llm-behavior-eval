"""UNQOVER solver. It passes no GenerateConfig: settings come from the user."""

from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    generate,
    solver,
    system_message,
)

from llm_behavior_eval.unqover.prompts import SYSTEM_MESSAGE


@solver
def unqover_user_turn() -> Solver:
    """Append the newline the legacy engine put after each question."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = state.input_text + "\n"
        return state

    return solve


def unqover_solver() -> list[Solver]:
    return [system_message(SYSTEM_MESSAGE), unqover_user_turn(), generate()]
