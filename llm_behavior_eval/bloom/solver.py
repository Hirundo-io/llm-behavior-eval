"""BLOOM solver. It sets no GenerateConfig for the model under test."""

from inspect_ai.solver import Generate, Solver, TaskState, solver, system_message

from llm_behavior_eval.common.preprocessing import legacy_text_preprocessing

from .prompts import SYSTEM_MESSAGE

# Legacy FAMILY_TOKEN_DEFAULTS["bias"]["enable_thinking"]: the chat-template
# variable the legacy engine rendered these prompts with.
LEGACY_THINKING = False


@solver
def bloom_generate(
    legacy_tokenizer: str | None = None, legacy_tokenizer_revision: str | None = None
) -> Solver:
    """Generate with the legacy user message: the question plus a newline.

    ``legacy_tokenizer`` opts in to the legacy text preprocessing, which runs on
    the final messages just before generation and records the target and
    stereotype round trips.
    """
    preprocess = (
        legacy_text_preprocessing(
            legacy_tokenizer,
            legacy_tokenizer_revision,
            legacy_thinking=LEGACY_THINKING,
            target=True,
            metadata_fields=("stereotyped_answer",),
        )
        if legacy_tokenizer
        else None
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = state.input_text + "\n"
        if preprocess is not None:
            state = await preprocess(state, generate)
        return await generate(state)

    return solve


def solver(
    legacy_tokenizer: str | None = None, legacy_tokenizer_revision: str | None = None
) -> list[Solver]:
    return [
        system_message(SYSTEM_MESSAGE),
        bloom_generate(legacy_tokenizer, legacy_tokenizer_revision),
    ]
