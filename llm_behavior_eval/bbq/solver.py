"""BBQ solver chain. It sets no GenerateConfig for the model under test."""

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

# Legacy FAMILY_TOKEN_DEFAULTS["bias"]["enable_thinking"]: the chat-template
# variable the legacy engine rendered BBQ prompts with.
LEGACY_THINKING = False


@solver
def bbq_user_message() -> Solver:
    """Send the question with the legacy trailing newline (``f"{question}\\n"``)."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = state.input_text + "\n"
        return state

    return solve


def bbq_solver(
    legacy_tokenizer: str | None = None, legacy_tokenizer_revision: str | None = None
) -> list[Solver]:
    """Prompt steps, then generate.

    With ``legacy_tokenizer`` (opt-in, for parity), record the legacy decoded
    prompt and the target and stereotype round trips before generating.
    """
    steps = [system_message(SYSTEM_MESSAGE), bbq_user_message()]
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
