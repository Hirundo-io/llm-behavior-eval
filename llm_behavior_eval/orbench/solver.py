"""OR-Bench solver. It sets no GenerateConfig for the model under test."""

from inspect_ai.model import get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from llm_behavior_eval.common.judge import GRADER_ROLE
from llm_behavior_eval.common.preprocessing import legacy_text_preprocessing

PROMPT_PARITY = (
    "native chat; legacy decoded template and 1024-token truncation unresolved"
)
LEGACY_PROMPT_PARITY = (
    "legacy-text-v1: decoded 1024-token chat-template prompt as the judge question"
)

# Legacy FAMILY_TOKEN_DEFAULTS["refusal"]["enable_thinking"]: the chat-template
# variable the legacy engine rendered these prompts with.
LEGACY_THINKING = True


@solver
def orbench_generate(
    legacy_tokenizer: str | None = None, legacy_tokenizer_revision: str | None = None
) -> Solver:
    """Send the question alone, as the legacy user turn, with no system prompt.

    The legacy engine sent ``f"{question}\\n"`` as the only message. The grader
    role is checked first so a missing judge fails before any answers are
    generated. ``legacy_tokenizer`` opts in to the legacy text preprocessing,
    which runs on the final messages just before generation.
    """
    preprocess = (
        legacy_text_preprocessing(
            legacy_tokenizer,
            legacy_tokenizer_revision,
            legacy_thinking=LEGACY_THINKING,
        )
        if legacy_tokenizer
        else None
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        get_model(role=GRADER_ROLE, required=True)
        state.user_prompt.text = state.input_text + "\n"
        state.metadata["prompt_parity"] = PROMPT_PARITY
        if preprocess is not None:
            state = await preprocess(state, generate)
            state.metadata["prompt_parity"] = LEGACY_PROMPT_PARITY
        return await generate(state)

    return solve
