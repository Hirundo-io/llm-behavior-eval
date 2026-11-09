"""Frozen legacy OR-Bench values for tests/test_inspect_orbench.py."""

from types import SimpleNamespace
from typing import Any

from scripts.parity.historical import legacy_module
from scripts.parity.removal_baseline._refusal import grade

SOURCES = [
    "llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator:FreeTextRefusalEvaluator._judge_prompt_template",
    "llm_behavior_eval.evaluation_utils.refusal_utils:_REFUSAL_PREFIXES",
    "llm_behavior_eval.evaluation_utils.refusal_utils:REFUSAL_MARKERS",
    "llm_behavior_eval.evaluation_utils.refusal_utils:normalize_refusal_dataset",
    "llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator:refusal_heuristic",
    "llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator:parse_or_bench_judgment",
    "llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator:FreeTextRefusalEvaluator._grade_impl",
    "llm_behavior_eval.evaluation_utils.eval_config:FAMILY_TOKEN_DEFAULTS",
    "llm_behavior_eval.evaluation_utils.eval_config:EvaluationConfig",
    "llm_behavior_eval.evaluation_utils.dataset_config:DatasetConfig",
]

DATASET_ID = "hirundo-io/or-bench"
HEURISTIC_ANSWERS = [
    "Sure, here it is.",
    "  Sorry, no.",
    "No, that is fine",
    "This is illegal",
    "",
]
PARSER_TEXTS = [
    "[[direct_answer]]",
    "[[ DIRECT_REFUSAL ]]",
    "[[indirect_refusal]]",
    "direct_refusal and indirect_refusal considered; [[direct_answer]]",
    "direct_answer considered; [[indirect_refusal]]",
    "The response is a direct_answer.",
    "Explanation mentions direct_answer but truncates at [[direct_",
    "[[other]]",
    "[[direct_refusal]] [[direct_answer]]",
    "",
]
# The source of test_conversion_reproduces_legacy_shuffled_selection.
SOURCE = {
    "prompt": [f"q{i}" for i in range(503)],
    "label": ["safe", "unsafe"] * 251 + ["safe"],
}
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
    "raw": [
        "[[direct_answer]]",
        "[[indirect_refusal]]",
        "[[direct_refusal]]",
        "bad",
        "Explanation considers direct_answer but truncates at [[direct_",
    ],
    "judge_stops": ["stop", "stop", "stop", "stop", "length"],
}


def capture() -> dict[str, Any]:
    from datasets import Dataset

    utils = legacy_module("llm_behavior_eval.evaluation_utils.refusal_utils")
    evaluator = legacy_module(
        "llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator"
    )
    eval_config = legacy_module("llm_behavior_eval.evaluation_utils.eval_config")
    dataset_config = legacy_module("llm_behavior_eval.evaluation_utils.dataset_config")
    instance = object.__new__(evaluator.FreeTextRefusalEvaluator)
    instance.dataset_config = SimpleNamespace(dataset_id=DATASET_ID)
    # Legacy selection: normalize the whole source, shuffle once with seed 42.
    shuffled = utils.normalize_refusal_dataset(Dataset.from_dict(SOURCE)).shuffle(
        seed=42
    )
    fields = eval_config.EvaluationConfig.model_fields
    return {
        "judge_prompt": {
            "dataset_id": DATASET_ID,
            "template": instance._judge_prompt_template(),
        },
        "heuristic_phrases": {
            "REFUSAL_PREFIXES": list(utils._REFUSAL_PREFIXES),
            "REFUSAL_MARKERS": list(utils.REFUSAL_MARKERS),
        },
        "heuristic": [
            [answer, evaluator.refusal_heuristic(answer)]
            for answer in HEURISTIC_ANSWERS
        ],
        "parser": [
            [text, evaluator.parse_or_bench_judgment(text).value]
            for text in PARSER_TEXTS
        ],
        "shuffled_selection": {
            "question": list(shuffled["question"]),
            "label": list(shuffled["label"]),
        },
        "metric_parity": grade(evaluator, METRIC_INPUTS, "orbench"),
        "family_token_defaults_refusal": dict(
            eval_config.FAMILY_TOKEN_DEFAULTS["refusal"]
        ),
        "evaluation_config_defaults": {
            "sample": fields["sample"].default,
            "max_samples": fields["max_samples"].default,
        },
        "dataset_config_seed": dataset_config.DatasetConfig.model_fields[
            "seed"
        ].default,
    }
