"""CCPC answer prompt. Sets no GenerateConfig for the model under test."""

from inspect_ai.model import get_model
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

from llm_behavior_eval.common.judge import GRADER_ROLE
from llm_behavior_eval.common.preprocessing import legacy_text_preprocessing

PROMPT_PARITY = (
    "unresolved: legacy chat-template decoding and 1024-token input truncation"
)
LEGACY_PROMPT_PARITY = (
    "legacy-text-v1 recorded; the judge keeps the original question, as the "
    "legacy CCPC evaluator did"
)
# Legacy FAMILY_TOKEN_DEFAULTS["censorship"]["enable_thinking"]: the
# chat-template variable the legacy engine rendered CCPC prompts with.
LEGACY_THINKING = False


@solver
def ccpc_prompt(prompt_parity: str = PROMPT_PARITY) -> Solver:
    """Legacy answer prompt: the question plus a trailing newline, no system prompt.

    Resolves the required grader role first, so a run without
    ``--model-role grader=...`` fails before any answer is generated.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        get_model(role=GRADER_ROLE, required=True)
        state.user_prompt.text = state.input_text + "\n"
        state.metadata["prompt_parity"] = prompt_parity
        return state

    return solve


def solver(
    legacy_tokenizer: str | None = None, legacy_tokenizer_revision: str | None = None
) -> list[Solver]:
    """Prompt, then generate; ``legacy_tokenizer`` opts in to preprocessing.

    The opt-in step only records the legacy prompt view (for boundary
    evidence): CCPC's legacy judge saw the original question, so the scorer
    does not read it.
    """
    if not legacy_tokenizer:
        return [ccpc_prompt(), generate()]
    return [
        ccpc_prompt(LEGACY_PROMPT_PARITY),
        legacy_text_preprocessing(
            legacy_tokenizer,
            legacy_tokenizer_revision,
            legacy_thinking=LEGACY_THINKING,
        ),
        generate(),
    ]
