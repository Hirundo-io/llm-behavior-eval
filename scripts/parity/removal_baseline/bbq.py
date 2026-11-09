"""Legacy BBQ expectations for tests/test_inspect_bbq.py."""

from typing import Any

from scripts.parity.removal_baseline import _bias

SOURCES = [
    *_bias.SOURCES,
    f"{_bias.LEGACY}.free_text_bias_evaluator:_JUDGE_OUTPUT_RE",
    f"{_bias.LEGACY}.free_text_bias_evaluator:_UNCERT_RE",
    f"{_bias.LEGACY}.free_text_bias_evaluator:FreeTextBiasEvaluator._match_llm_answers",
]
AGREEMENT_RAWS = [
    'gt_agree: "YES"',
    "gt_agree:no",
    "GT_AGREE: yesmore",
    "gt_agree: 'yes'",
    "",
    "nonsense",
]
UNCERTAINTY_RAWS = [
    'candidate_uncertain: "YES"',
    "candidate_uncertain:no",
    "",
    "candidate_uncertain: 'yes'",
]


def judge_messages() -> list[list[dict[str, str]]]:
    """Messages legacy ``_match_llm_answers`` renders for one sample."""
    module = _bias.bias()
    captured: list[list[dict[str, str]]] = []
    evaluator = object.__new__(module.FreeTextBiasEvaluator)
    evaluator.prepare_judge_tokenizer = lambda: None
    evaluator._get_judge_tokenizer = lambda: None

    def no_judgments(engine: Any, batch: list[Any]) -> list[list[dict[str, str]]]:
        return [[{"generated_text": ""}]] * len(batch)

    evaluator.run_judge_with_backoff = no_judgments
    original = module.safe_apply_chat_template
    module.safe_apply_chat_template = lambda tokenizer, messages: (
        captured.append(messages) or ""
    )
    try:
        evaluator._match_llm_answers(object(), ["a0"], ["person"], ["stereo"], ["q0"])
    finally:
        module.safe_apply_chat_template = original
    return captured


def capture() -> dict[str, Any]:
    module = _bias.bias()
    return {
        "constants": _bias.constants(),
        "patterns": {
            "agreement": {
                "pattern": module._JUDGE_OUTPUT_RE.pattern,
                "flags": int(module._JUDGE_OUTPUT_RE.flags),
            },
            "uncertainty": {
                "pattern": module._UNCERT_RE.pattern,
                "flags": int(module._UNCERT_RE.flags),
            },
        },
        "agreement": _bias.agreements(AGREEMENT_RAWS),
        "uncertainty": _bias.uncertainties(UNCERTAINTY_RAWS),
        "judge_messages": judge_messages(),
        "grade_inputs": _bias.GRADE_INPUTS,
        "grade": _bias.grades({"bias": "BIAS", "unbias": "UNBIAS"}),
        "settings": _bias.settings(),
    }
