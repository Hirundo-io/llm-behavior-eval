"""Legacy Purple Llama prompt-injection expectations for test_inspect_prompt_injection."""

import math
from types import SimpleNamespace
from typing import Any

from scripts.parity.historical import legacy_module

SOURCES = [
    "llm_behavior_eval.evaluation_utils.prompts:FREE_TEXT_SYSTEM_MESSAGE",
    "llm_behavior_eval.evaluation_utils.eval_config:FAMILY_TOKEN_DEFAULTS",
    "llm_behavior_eval.evaluation_utils.eval_config:EvaluationConfig",
    "llm_behavior_eval.evaluation_utils.custom_dataset:free_text_preprocess_function",
    "llm_behavior_eval.evaluation_utils.free_text_injection_evaluator:"
    "FreeTextPromptInjectionEvaluator",
]
PARSER_VECTORS = ["Yes", "NO!", "yesterday", "", "No then yes", "YES/no", "  yes  "]
# test_native_legacy_grade_parity: answers, finish reasons and stub judge labels.
GRADE_ANSWERS = ["<think>secret</think>a0", "a1", "a2", "a3"]
GRADE_FINISH_REASONS = ["stop", "stop", "stop", "length"]
GRADE_JUDGE_LABELS = ["Yes", "No", "No"]
# test_current_preprocessor_and_explicit_truncation_gap
LONG_QUESTION = "q" * 1100


def _preprocessing() -> dict[str, Any]:
    custom_dataset = legacy_module("llm_behavior_eval.evaluation_utils.custom_dataset")
    captured: list[Any] = []
    calls: list[Any] = []

    def template(tokenizer: Any, messages: Any, **kwargs: Any) -> str:
        captured.append(messages)
        return "CHAT:" + messages[-1]["content"]

    class Tokenizer:
        def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
            calls.append({"texts": list(texts), "kwargs": dict(kwargs)})
            limit = kwargs["max_length"]
            return {
                "input_ids": [[ord(c) for c in t[:limit]] for t in texts],
                "attention_mask": [[1] * min(len(t), limit) for t in texts],
            }

    original = custom_dataset.safe_apply_chat_template
    custom_dataset.safe_apply_chat_template = template
    try:
        legacy = custom_dataset.free_text_preprocess_function(
            {"question": [LONG_QUESTION], "answer": ["a"]},
            Tokenizer(),
            max_length=1024,
            gt_max_length=128,
            has_stereotype=False,
        )
    finally:
        custom_dataset.safe_apply_chat_template = original
    return {
        "question": LONG_QUESTION,
        "chat_messages": captured[0],
        "tokenizer_calls": calls,
        "test_input_ids_width": int(legacy["test_input_ids"].shape[1]),
        "judge_questions_width": int(legacy["judge_questions"].shape[1]),
    }


def _grade() -> dict[str, Any]:
    module = legacy_module(
        "llm_behavior_eval.evaluation_utils.free_text_injection_evaluator"
    )
    legacy: Any = object.__new__(module.FreeTextPromptInjectionEvaluator)
    legacy.eval_config = SimpleNamespace(
        exclude_thinking_trace_for_judge=True,
        thinking_start_token="<think>",  # noqa: S106 - not a credential
        thinking_end_token="</think>",  # noqa: S106 - not a credential
    )
    legacy._grade_batch = lambda *args: list(GRADE_JUDGE_LABELS)
    saved: dict[str, Any] = {}
    legacy.save_results = lambda **kw: saved.update(kw)
    legacy._grade_impl(
        [
            module._InjectionGenerationRecord(
                input_texts=["q0", "q1", "q2", "q3"],
                gt_answers=["a"] * 4,
                answers=list(GRADE_ANSWERS),
                finish_reasons=list(GRADE_FINISH_REASONS),
                judge_questions=["q0", "q1", "q2", "q3"],
                response_tokens=[2] * 4,
            )
        ],
        judge_engine=object(),
    )
    accuracy = saved["accuracy"]
    if math.isnan(accuracy):
        raise ValueError("Legacy accuracy is NaN")
    return {
        "answers": GRADE_ANSWERS,
        "finish_reasons": GRADE_FINISH_REASONS,
        "judge_labels": GRADE_JUDGE_LABELS,
        "accuracy": accuracy,
        "incomplete_response_rate": saved["incomplete_response_rate"],
    }


def capture() -> dict[str, Any]:
    prompts = legacy_module("llm_behavior_eval.evaluation_utils.prompts")
    eval_config = legacy_module("llm_behavior_eval.evaluation_utils.eval_config")
    evaluator = legacy_module(
        "llm_behavior_eval.evaluation_utils.free_text_injection_evaluator"
    ).FreeTextPromptInjectionEvaluator
    fields = eval_config.EvaluationConfig.model_fields
    return {
        "family_token_defaults": dict(
            eval_config.FAMILY_TOKEN_DEFAULTS["prompt-injection"]
        ),
        "evaluation_config_defaults": {
            "sample": fields["sample"].default,
            "max_samples": fields["max_samples"].default,
        },
        "default_system_prompt": prompts.FREE_TEXT_SYSTEM_MESSAGE["content"],
        "prompt_judge_response": evaluator.PROMPT_JUDGE_RESPONSE,
        "parser": {
            raw: evaluator._map_judge_outputs_yes_no([[{"generated_text": raw}]])[0]
            for raw in PARSER_VECTORS
        },
        "preprocessing": _preprocessing(),
        "grade": _grade(),
    }
