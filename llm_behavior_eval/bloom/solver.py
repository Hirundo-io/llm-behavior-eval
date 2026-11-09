"""BLOOM solver. It sets no GenerateConfig for the model under test."""

from inspect_ai.solver import Generate, Solver, TaskState, solver, system_message

from .prompts import SYSTEM_MESSAGE


@solver
def bloom_generate() -> Solver:
    """Generate with the legacy user message: the question plus a newline."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = state.input_text + "\n"
        return await generate(state)

    return solve


def solver() -> list[Solver]:
    return [system_message(SYSTEM_MESSAGE), bloom_generate()]
