"""Frozen legacy XSTest values for tests/test_inspect_xstest.py."""

from typing import Any

from scripts.parity.historical import legacy_module
from scripts.parity.removal_baseline._refusal import grade

SOURCES = [
    "llm_behavior_eval.evaluation_utils.refusal_utils:XSTEST_JUDGE_PROMPT",
    "llm_behavior_eval.evaluation_utils.refusal_utils:_REFUSAL_PREFIXES",
    "llm_behavior_eval.evaluation_utils.refusal_utils:REFUSAL_MARKERS",
    "llm_behavior_eval.evaluation_utils.refusal_utils:XSTEST_DATASET",
    "llm_behavior_eval.evaluation_utils.refusal_utils:normalize_refusal_label",
    "llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator:parse_xstest_judgment",
    "llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator:refusal_heuristic",
    "llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator:FreeTextRefusalEvaluator._grade_impl",
    "llm_behavior_eval.evaluation_utils.eval_config:FAMILY_TOKEN_DEFAULTS",
    "llm_behavior_eval.evaluation_utils.eval_config:EvaluationConfig",
]

PARSER_TEXTS = [
    "1_full_compliance",
    "2_FULL_REFUSAL",
    "3_partial_refusal",
    "CLASS: 3_Partial_Refusal because ...",
    "nothing",
    "x1_full_compliance",
    "3_partial_refusal then 1_full_compliance",
]
HEURISTIC_ANSWERS = [
    "  Sorry!",
    "I cannot do that",
    "That is illegal",
    "Sure, here",
    "",
    "<think>x</think> I will not",
    "No.",
]
LABELS: list[Any] = ["safe", " SAFE ", "Unsafe", 1, None, "other"]
# The deterministic inputs of test_legacy_metric_parity.
METRIC_INPUTS = {
    "labels": ["safe", "safe", "safe", "unsafe", "unsafe", "unsafe", "safe"],
    "answers": [
        "sure",
        "Sorry",
        "illegal",
        "I cannot",
        "sure",
        "I will not",
        "<think>x</think> sure",
    ],
    "stops": ["stop", "length", "weird", "stop", "length", None, "stop"],
    "raw": ["1_full_compliance", "3_partial_refusal", "2_full_refusal", "bad", "bad"],
}


def capture() -> dict[str, Any]:
    utils = legacy_module("llm_behavior_eval.evaluation_utils.refusal_utils")
    evaluator = legacy_module(
        "llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator"
    )
    eval_config = legacy_module("llm_behavior_eval.evaluation_utils.eval_config")
    labels = []
    for label in LABELS:
        try:
            labels.append(
                {"label": label, "normalized": utils.normalize_refusal_label(label)}
            )
        except ValueError as error:
            labels.append({"label": label, "error": str(error)})
    fields = eval_config.EvaluationConfig.model_fields
    return {
        "constants": {
            "XSTEST_JUDGE_PROMPT": utils.XSTEST_JUDGE_PROMPT,
            "REFUSAL_PREFIXES": list(utils._REFUSAL_PREFIXES),
            "REFUSAL_MARKERS": list(utils.REFUSAL_MARKERS),
            "XSTEST_DATASET": utils.XSTEST_DATASET,
        },
        "parser": [
            [text, evaluator.parse_xstest_judgment(text).value] for text in PARSER_TEXTS
        ],
        "heuristic": [
            [answer, evaluator.refusal_heuristic(answer)]
            for answer in HEURISTIC_ANSWERS
        ],
        "labels": labels,
        "metric_parity": grade(evaluator, METRIC_INPUTS, "xstest"),
        "family_token_defaults_refusal": dict(
            eval_config.FAMILY_TOKEN_DEFAULTS["refusal"]
        ),
        "evaluation_config_defaults": {
            "sample": fields["sample"].default,
            "max_samples": fields["max_samples"].default,
        },
    }
