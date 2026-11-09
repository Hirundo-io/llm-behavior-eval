"""XSTest solver. It sets no GenerateConfig for the model under test.

Like the legacy engine, there is no system prompt (the default free-text
system prompt hurts refusal evaluation) and the user message is the question
followed by a newline.
"""

from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

from llm_behavior_eval.common.preprocessing import legacy_text_preprocessing

# Legacy FAMILY_TOKEN_DEFAULTS["refusal"]["enable_thinking"]: the chat-template
# variable the legacy engine rendered these prompts with.
LEGACY_THINKING = True


@solver
def xstest_user_prompt() -> Solver:
    """Send ``f"{question}\\n"`` as the only (user) message."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = state.input_text + "\n"
        return state

    return solve


def xstest_solver(
    legacy_tokenizer: str | None = None, legacy_tokenizer_revision: str | None = None
) -> list[Solver]:
    """Prompt, then generate; ``legacy_tokenizer`` opts in to preprocessing."""
    steps = [xstest_user_prompt()]
    if legacy_tokenizer:
        steps.append(
            legacy_text_preprocessing(
                legacy_tokenizer,
                legacy_tokenizer_revision,
                legacy_thinking=LEGACY_THINKING,
            )
        )
    return [*steps, generate()]
