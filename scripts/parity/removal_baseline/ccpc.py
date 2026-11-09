"""Legacy CCPC expectations for test_inspect_ccpc."""

import math
from typing import Any

from scripts.parity.historical import legacy_module

SOURCES = [
    "llm_behavior_eval.evaluation_utils.censorship_utils:CCPC_*",
    "llm_behavior_eval.evaluation_utils.censorship_utils:parse_censorship_judgment",
    "llm_behavior_eval.evaluation_utils.censorship_utils:calculate_censorship_metrics",
    "llm_behavior_eval.evaluation_utils.censorship_utils:censorship_dataset_fingerprint",
    "llm_behavior_eval.evaluation_utils.eval_config:FAMILY_TOKEN_DEFAULTS",
    "llm_behavior_eval.evaluation_utils.eval_config:EvaluationConfig",
]
PARSER_VECTORS = [
    "True",
    " FALSE\n",
    "true.",
    "yes",
    "",
    "<think>x</think>True",
    "True\nFalse",
]
METRIC_VECTORS = [[], ["True"], ["True", "False"], ["True", "oops"]]


def _json_float(value: float | None) -> float | str | None:
    """NaN is not JSON; store it as the string "nan"."""
    return "nan" if value is not None and math.isnan(value) else value


def synthetic_rows(prefix: str) -> dict[str, list[str]]:
    """The tests' synthetic 500-row sources (``rows`` and the CLI snapshot)."""
    return {
        "question": [f"{prefix}{i}" for i in range(500)],
        "benchmark_id": [f"id-{i}" for i in range(500)],
    }


def capture() -> dict[str, Any]:
    from datasets import Dataset

    legacy = legacy_module("llm_behavior_eval.evaluation_utils.censorship_utils")
    eval_config = legacy_module("llm_behavior_eval.evaluation_utils.eval_config")
    fields = eval_config.EvaluationConfig.model_fields
    return {
        "contract": {
            key: value for key, value in vars(legacy).items() if key.startswith("CCPC_")
        },
        "parser": {
            text: legacy.parse_censorship_judgment(text) for text in PARSER_VECTORS
        },
        "metrics": [
            {
                "raw": raw,
                "metrics": [
                    _json_float(value)
                    for value in legacy.calculate_censorship_metrics(
                        [legacy.parse_censorship_judgment(x) for x in raw]
                    )
                ],
            }
            for raw in METRIC_VECTORS
        ],
        "fingerprints": {
            prefix: legacy.censorship_dataset_fingerprint(
                Dataset.from_dict(synthetic_rows(prefix))
            )
            for prefix in ["问题 ", "q"]
        },
        "family_token_defaults": dict(eval_config.FAMILY_TOKEN_DEFAULTS["censorship"]),
        "evaluation_config_defaults": {
            "sample": fields["sample"].default,
            "max_samples": fields["max_samples"].default,
        },
    }
