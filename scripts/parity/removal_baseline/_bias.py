"""Shared legacy captures for the bias benchmarks (BBQ, UNQOVER, BLOOM).

Every value comes from the legacy ``FreeTextBiasEvaluator`` and its module,
loaded only through ``legacy_module`` (frozen pre-removal checkout).
"""

from types import SimpleNamespace
from typing import Any

from scripts.parity.historical import legacy_module

LEGACY = "llm_behavior_eval.evaluation_utils"
SOURCES = [
    f"{LEGACY}.prompts:FREE_TEXT_SYSTEM_MESSAGE",
    f"{LEGACY}.prompts:UNKNOWN_ANSWERS",
    f"{LEGACY}.free_text_bias_evaluator:FreeTextBiasEvaluator."
    "AGREEMENT_CHECK_SYSTEM_PROMPT",
    f"{LEGACY}.free_text_bias_evaluator:FreeTextBiasEvaluator."
    "UNCERTAINTY_CHECK_SYSTEM_PROMPT",
    f"{LEGACY}.free_text_bias_evaluator:extract_agreements",
    f"{LEGACY}.free_text_bias_evaluator:map_uncertainty_outputs",
    f"{LEGACY}.free_text_bias_evaluator:FreeTextBiasEvaluator._grade_impl",
    f"{LEGACY}.eval_config:FAMILY_TOKEN_DEFAULTS",
    f"{LEGACY}.eval_config:EvaluationConfig.model_fields",
    f"{LEGACY}.dataset_config:DatasetConfig.model_fields",
]

# The deterministic grading fixture of the tests' test_legacy_grade_parity.
GRADE_INPUTS = {
    "targets": ["person", "Unknown", "person", "person", "person"],
    "gt": ["yes", "no", "unmatched", "no"],
    "uncertainty": ["no", "yes", "yes", "no"],
    "stereo": ["yes", "yes", "yes", "yes"],
    "finish_reasons": ["stop", "stop", "stop", "stop", "length"],
}


def bias() -> Any:
    return legacy_module(f"{LEGACY}.free_text_bias_evaluator")


def constants() -> dict[str, Any]:
    prompts = legacy_module(f"{LEGACY}.prompts")
    evaluator = bias().FreeTextBiasEvaluator
    return {
        "system_message": prompts.FREE_TEXT_SYSTEM_MESSAGE["content"],
        "unknown_answers": list(prompts.UNKNOWN_ANSWERS),
        "agreement_check_system_prompt": evaluator.AGREEMENT_CHECK_SYSTEM_PROMPT,
        "uncertainty_check_system_prompt": evaluator.UNCERTAINTY_CHECK_SYSTEM_PROMPT,
    }


def agreements(raws: list[str]) -> dict[str, str]:
    """``extract_agreements`` per raw judge text (the Agreement value)."""
    extract = bias().extract_agreements
    return {raw: extract([[{"generated_text": raw}]])[0].value for raw in raws}


def uncertainties(raws: list[str]) -> dict[str, str]:
    """``map_uncertainty_outputs`` per raw judge text."""
    parse = bias().map_uncertainty_outputs
    return {raw: parse([[{"generated_text": raw}]])[0] for raw in raws}


def settings() -> dict[str, Any]:
    eval_config = legacy_module(f"{LEGACY}.eval_config")
    dataset_config = legacy_module(f"{LEGACY}.dataset_config")
    fields = eval_config.EvaluationConfig.model_fields
    return {
        "family_token_defaults": dict(eval_config.FAMILY_TOKEN_DEFAULTS["bias"]),
        "evaluation_sample_default": fields["sample"].default,
        "evaluation_max_samples_default": fields["max_samples"].default,
        "dataset_seed_default": dataset_config.DatasetConfig.model_fields[
            "seed"
        ].default,
    }


def grade(dataset_type: str, has_stereotype: bool) -> dict[str, Any]:
    """Legacy ``_grade_impl`` save_results on the deterministic fixture."""
    module = bias()
    enums = legacy_module(f"{LEGACY}.enums")
    inputs = GRADE_INPUTS
    legacy = object.__new__(module.FreeTextBiasEvaluator)
    legacy.num_samples = 5
    legacy.has_stereotype = has_stereotype
    legacy.dataset_config = SimpleNamespace(
        dataset_type=getattr(enums.DatasetType, dataset_type)
    )
    legacy._format_answers = lambda answers: answers
    legacy._match_llm_answers = lambda *args: (
        [
            (module.Agreement(g), module.Agreement(s if has_stereotype else "no"))
            for g, s in zip(inputs["gt"], inputs["stereo"], strict=True)
        ],
        inputs["uncertainty"],
        [[{"generated_text": ""}]] * 4,
        [[{"generated_text": ""}]] * 4,
    )
    saved: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    legacy.save_results = lambda *args, **kwargs: saved.append((args, kwargs))
    legacy._grade_impl(
        [
            module._BiasGenerationRecord(
                questions=[f"q{i}" for i in range(5)],
                answers=[f"a{i}" for i in range(5)],
                correct_answers=inputs["targets"],
                stereotyped_answers=["stereo"] * 5 if has_stereotype else None,
                finish_reasons=inputs["finish_reasons"],
            )
        ],
        judge_engine=object(),
    )
    ((args, kwargs),) = saved
    return {
        "accuracy": args[1],
        "stereotype_bias": args[2],
        "empty_responses": args[3],
        "incomplete_response_rate": kwargs["incomplete_response_rate"],
    }


def grades(kinds: dict[str, str]) -> dict[str, Any]:
    """``{kind: {has_stereotype: saved}}`` with the legacy DatasetType per kind."""
    return {
        kind: {
            str(has_stereotype): grade(dataset_type, has_stereotype)
            for has_stereotype in (True, False)
        }
        for kind, dataset_type in kinds.items()
    }
