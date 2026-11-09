"""Legacy preprocessing under the character tokenizer of the probe test.

Runs the legacy ``free_text_preprocess_function`` with the deterministic
character tokenizer of tests/test_parity_evidence.py
(``test_probe_against_actual_legacy_preprocessing``) and its fixed render, and
stores the padded tensors that test compares with ``legacy_text_probe``.
"""

from typing import Any

from scripts.parity.historical import legacy_module

SOURCES = [
    "llm_behavior_eval.evaluation_utils.custom_dataset:free_text_preprocess_function",
]
RENDERED = "system:user:" + "Q" * 1100
ROW = {
    "question": "Q" * 1100,
    "answer": "A" * 300,
    "stereotyped_answer": "S" * 300,
    "judge_question": "J" * 300,
}


class CharacterTokenizer:
    """The test's tokenizer: ``<bos>`` = 1, pad = 0, one id per character."""

    def __call__(self, texts, **kwargs):
        rows, masks = [], []
        for text in texts:
            ids = ([1] if kwargs.get("add_special_tokens", True) else []) + [
                ord(c) for c in text
            ]
            ids = ids[: kwargs["max_length"]]
            pad = kwargs["max_length"] - len(ids)
            rows.append(ids + [0] * pad)
            masks.append([1] * len(ids) + [0] * pad)
        return {"input_ids": rows, "attention_mask": masks}


def capture() -> dict[str, Any]:
    custom_dataset = legacy_module("llm_behavior_eval.evaluation_utils.custom_dataset")
    original = custom_dataset.safe_apply_chat_template
    custom_dataset.safe_apply_chat_template = lambda *args, **kwargs: RENDERED
    try:
        legacy = custom_dataset.free_text_preprocess_function(
            {key: [value] for key, value in ROW.items()},
            CharacterTokenizer(),
            1024,
            256,
            True,
        )
    finally:
        custom_dataset.safe_apply_chat_template = original
    return {
        key: legacy[key][0].tolist()
        for key in (
            "test_input_ids",
            "gt_answers",
            "stereotyped_answers",
            "judge_questions",
        )
    }
