"""Legacy free-text preprocessing outputs for tests/test_inspect_preprocessing.py.

Runs the legacy ``free_text_preprocess_function`` on the rows of
``test_views_equal_legacy_preprocessing`` for the tiny tokenizer and the pinned
Gemma 3 and Qwen3 snapshots (required here, unlike in the tests), the legacy
Qwen3 renders of ``test_qwen3_thinking_render_matches_legacy``, the HaluEval
judge template and the legacy per-family thinking defaults.
"""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from scripts.parity.historical import legacy_module

SOURCES = [
    "llm_behavior_eval.evaluation_utils.custom_dataset:free_text_preprocess_function",
    "llm_behavior_eval.evaluation_utils.util_functions:safe_apply_chat_template",
    "llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator:GRADER_TEMPLATE",
    "llm_behavior_eval.evaluation_utils.eval_config:FAMILY_TOKEN_DEFAULTS",
]
TOKENIZERS = ("tiny", "gemma3", "qwen3")
TEXT_KEYS = {
    "answer": "gt_answers",
    "stereotyped_answer": "stereotyped_answers",
    "judge_question": "judge_questions",
}
QWEN3_MESSAGES = [
    {"role": "system", "content": "Be helpful."},
    {"role": "user", "content": "question\n"},
]


def views_row(long: bool) -> dict[str, str]:
    """The row of test_views_equal_legacy_preprocessing (same code there)."""
    repeat = 1400 if long else 1
    return dict(
        question=" First  question\n" + "long input " * repeat,
        answer=" target " * (500 if long else 1),
        stereotyped_answer=" stereotype " * (500 if long else 1),
        system_prompt="You are helpful.\n",
        judge_question=" Independent question " * (500 if long else 1),
    )


def row_digest(row: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def case_key(tokenizer: str, side: str, thinking: bool, long: bool) -> str:
    return f"{tokenizer}-{side}-{str(thinking).lower()}-{'long' if long else 'short'}"


def _views(directory: Path) -> dict[str, Any]:
    from llm_behavior_eval.common.preprocessing import (
        LEGACY_PROMPT_MAX_LENGTH,
        LEGACY_TEXT_MAX_LENGTH,
        load_tokenizer,
    )
    from tests.legacy_tokenizers import pinned_tokenizer, tiny_tokenizer

    preprocess = legacy_module(
        "llm_behavior_eval.evaluation_utils.custom_dataset"
    ).free_text_preprocess_function
    cases: dict[str, Any] = {}
    for name in TOKENIZERS:
        path = tiny_tokenizer(directory) if name == "tiny" else pinned_tokenizer(name)
        tokenizer = load_tokenizer(str(path))
        previous = tokenizer.truncation_side
        pad = tokenizer.pad_token_id
        try:
            for side in ("left", "right"):
                tokenizer.truncation_side = side
                for thinking in (False, True):
                    for long in (False, True):
                        row = views_row(long)
                        legacy = preprocess(
                            {key: [value] for key, value in row.items()},
                            tokenizer,
                            LEGACY_PROMPT_MAX_LENGTH,
                            LEGACY_TEXT_MAX_LENGTH,
                            True,
                            enable_thinking=thinking,
                        )

                        def ids(key: str, legacy=legacy, pad=pad) -> list[int]:
                            return [int(t) for t in legacy[key][0] if int(t) != pad]

                        cases[case_key(name, side, thinking, long)] = {
                            "row_sha256": row_digest(row),
                            "test_input_ids": ids("test_input_ids"),
                            **{key: ids(key) for key in TEXT_KEYS.values()},
                        }
        finally:
            tokenizer.truncation_side = previous
    return cases


def capture() -> dict[str, Any]:
    from llm_behavior_eval.common.preprocessing import load_tokenizer
    from tests.legacy_tokenizers import pinned_tokenizer

    util_functions = legacy_module("llm_behavior_eval.evaluation_utils.util_functions")
    qwen3 = load_tokenizer(str(pinned_tokenizer("qwen3")))
    with tempfile.TemporaryDirectory() as directory:
        views = _views(Path(directory))
    return {
        "views": views,
        "qwen3_render": {
            str(thinking).lower(): util_functions.safe_apply_chat_template(
                qwen3, [dict(m) for m in QWEN3_MESSAGES], enable_thinking=thinking
            )
            for thinking in (False, True)
        },
        "hallucination_grader_template": legacy_module(
            "llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator"
        ).GRADER_TEMPLATE,
        "family_token_defaults": legacy_module(
            "llm_behavior_eval.evaluation_utils.eval_config"
        ).FAMILY_TOKEN_DEFAULTS,
    }
