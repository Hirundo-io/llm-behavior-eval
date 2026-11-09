"""Offline tokenizer oracle for evidence, not a model provider or solver.

Pass the actual rendered template from the evaluated tokenizer. This deliberately
cannot turn a decoded prompt into a native user message: re-templating it would
change the input. Standard HF/vLLM generation input parity remains a separate gate.
"""

from typing import Any


def legacy_text_probe(
    tokenizer: Any,
    rendered_prompt: str,
    target: str,
    *,
    judge_question: str | None = None,
    stereotype: str | None = None,
) -> dict:
    """Reproduce legacy padding/truncation/decode without importing its engine."""

    def roundtrip(text: str, limit: int, special: bool) -> dict:
        kwargs = {"truncation": True, "padding": "max_length", "max_length": limit}
        if not special:
            kwargs["add_special_tokens"] = False
        encoded = tokenizer([text], **kwargs)
        ids = encoded["input_ids"][0]
        return {
            "input_ids": list(ids),
            "attention_mask": list(encoded["attention_mask"][0]),
            "decoded": tokenizer.decode(ids, skip_special_tokens=True),
        }

    prompt = roundtrip(rendered_prompt, 1024, True)
    result = {
        "schema": "legacy-text-v1",
        "prompt": prompt,
        "target": roundtrip(target, 256, False),
        "judge_question": prompt["decoded"],
    }
    if judge_question is not None:
        result["judge_question"] = roundtrip(judge_question, 256, False)["decoded"]
    if stereotype is not None:
        result["stereotype"] = roundtrip(stereotype, 256, False)
    return result
