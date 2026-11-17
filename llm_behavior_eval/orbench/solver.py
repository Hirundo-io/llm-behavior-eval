"""OR-Bench solver. It sets no GenerateConfig for the model under test."""

from inspect_ai.model import get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from llm_behavior_eval.common.judge import GRADER_ROLE

PROMPT_PARITY = (
    "native chat; legacy decoded template and 1024-token truncation unresolved"
)


@solver
def orbench_generate() -> Solver:
    """Send the question alone, as the legacy user turn, with no system prompt.

    The legacy engine sent ``f"{question}\\n"`` as the only message. The grader
    role is checked first so a missing judge fails before any answers are
    generated.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        get_model(role=GRADER_ROLE, required=True)
        state.user_prompt.text = state.input_text + "\n"
        state.metadata["prompt_parity"] = PROMPT_PARITY
        return await generate(state)

    return solve
