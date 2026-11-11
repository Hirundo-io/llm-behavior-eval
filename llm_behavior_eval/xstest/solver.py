"""XSTest solver. It sets no GenerateConfig for the model under test.

Like the legacy engine, there is no system prompt (the default free-text
system prompt hurts refusal evaluation) and the user message is the question
followed by a newline.
"""

from inspect_ai.solver import Generate, Solver, TaskState, generate, solver


@solver
def xstest_user_prompt() -> Solver:
    """Send ``f"{question}\\n"`` as the only (user) message."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = state.input_text + "\n"
        return state

    return solve


def xstest_solver() -> list[Solver]:
    return [xstest_user_prompt(), generate()]
