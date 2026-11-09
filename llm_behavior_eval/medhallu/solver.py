"""Med-Hallu solver. It sets no GenerateConfig for the model under test."""

from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    generate,
    solver,
    system_message,
)

from llm_behavior_eval.common.preprocessing import legacy_text_preprocessing

from .prompts import SYSTEM_MESSAGE

# Legacy FAMILY_TOKEN_DEFAULTS["hallucination"]["enable_thinking"]: the
# chat-template variable the legacy engine rendered these prompts with.
LEGACY_THINKING = False


@solver
def medhallu_user_prompt() -> Solver:
    """Append the trailing newline the legacy engine added to the user turn."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = f"{state.input_text}\n"
        return state

    return solve


def medhallu_solver(
    legacy_tokenizer: str | None = None, legacy_tokenizer_revision: str | None = None
) -> list[Solver]:
    """Prompt steps, then generate; ``legacy_tokenizer`` opts in to preprocessing."""
    steps = [system_message(SYSTEM_MESSAGE), medhallu_user_prompt()]
    if legacy_tokenizer:
        steps.append(
            legacy_text_preprocessing(
                legacy_tokenizer,
                legacy_tokenizer_revision,
                legacy_thinking=LEGACY_THINKING,
                target=True,
            )
        )
    return [*steps, generate()]
