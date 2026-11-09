"""Benchmark-owned solver. It sets no GenerateConfig for the model under test.

The per-row system prompt and user message are already in each sample's input.
"""

from inspect_ai.solver import Solver, generate

from llm_behavior_eval.common.preprocessing import legacy_text_preprocessing

# Legacy FAMILY_TOKEN_DEFAULTS["prompt-injection"]["enable_thinking"]: the
# chat-template variable the legacy engine rendered these prompts with.
LEGACY_THINKING = False


def solver(
    legacy_tokenizer: str | None = None, legacy_tokenizer_revision: str | None = None
) -> list[Solver]:
    """Generate; ``legacy_tokenizer`` opts in to preprocessing first.

    The legacy judge saw the per-row judge question after a 256-token round
    trip, so the opt-in step records that view of ``judge_question``.
    """
    if not legacy_tokenizer:
        return [generate()]
    return [
        legacy_text_preprocessing(
            legacy_tokenizer,
            legacy_tokenizer_revision,
            legacy_thinking=LEGACY_THINKING,
            metadata_fields=("judge_question",),
        ),
        generate(),
    ]
