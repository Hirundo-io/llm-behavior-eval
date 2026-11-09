"""UNQOVER solver. It passes no GenerateConfig: settings come from the user."""

from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    generate,
    solver,
    system_message,
)

from llm_behavior_eval.common.preprocessing import legacy_text_preprocessing
from llm_behavior_eval.unqover.prompts import SYSTEM_MESSAGE

# Legacy FAMILY_TOKEN_DEFAULTS["bias"]["enable_thinking"]: the chat-template
# variable the legacy engine rendered these prompts with.
LEGACY_THINKING = False


@solver
def unqover_user_turn() -> Solver:
    """Append the newline the legacy engine put after each question."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = state.input_text + "\n"
        return state

    return solve


def unqover_solver(
    legacy_tokenizer: str | None = None, legacy_tokenizer_revision: str | None = None
) -> list[Solver]:
    """Prompt steps, then generate; ``legacy_tokenizer`` opts in to preprocessing."""
    steps = [system_message(SYSTEM_MESSAGE), unqover_user_turn()]
    if legacy_tokenizer:
        steps.append(
            legacy_text_preprocessing(
                legacy_tokenizer,
                legacy_tokenizer_revision,
                legacy_thinking=LEGACY_THINKING,
                target=True,
                metadata_fields=("stereotyped_answer",),
            )
        )
    return [*steps, generate()]
