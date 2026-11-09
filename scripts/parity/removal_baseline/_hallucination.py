"""Legacy hallucination-family values shared by the HaluEval and Med-Hallu baselines.

These run the legacy FreeTextHaluEvaluator with the same deterministic stand-ins
the native-vs-legacy tests used before removal (frozen checkout only).
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from scripts.parity.historical import legacy_module

HALLU = "llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator"
PROMPTS = "llm_behavior_eval.evaluation_utils.prompts"
BASE = "llm_behavior_eval.evaluation_utils.base_evaluator"
CUSTOM = "llm_behavior_eval.evaluation_utils.custom_dataset"
CONFIG = "llm_behavior_eval.evaluation_utils.eval_config"
DATASET_CONFIG = "llm_behavior_eval.evaluation_utils.dataset_config"

SOURCES = [
    f"{HALLU}:GRADER_TEMPLATE",
    f"{PROMPTS}:FREE_TEXT_SYSTEM_MESSAGE",
    f"{HALLU}:FreeTextHaluEvaluator._map_judge_outputs",
    f"{HALLU}:FreeTextHaluEvaluator._grade_impl",
    f"{BASE}:BaseEvaluator.prepare_dataloader",
    f"{CUSTOM}:free_text_preprocess_function",
    f"{CONFIG}:FAMILY_TOKEN_DEFAULTS",
    f"{CONFIG}:EvaluationConfig",
    f"{DATASET_CONFIG}:DatasetConfig",
]

# Judge replies whose legacy grade the native parser must reproduce.
PARSE_VECTORS = [
    "A",
    "B",
    "C",
    "garbage",
    "",
    "AA",
    "a",
    "B then A",
    "(A)",
    "<think>B</think>A",
]
# Aggregate fixture: native stop reasons and the stub judge grades of the four
# judged rows (the last three are skipped by both implementations).
STOPS = ["stop", "stop", "stop", "stop", "max_tokens", "unknown", None]
STUB_GRADES = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED", "NOT_ATTEMPTED"]


@contextmanager
def patched(module: Any, name: str, value: Any):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def prompts() -> dict[str, str]:
    return {
        "grader_template": legacy_module(HALLU).GRADER_TEMPLATE,
        "system_message": legacy_module(PROMPTS).FREE_TEXT_SYSTEM_MESSAGE["content"],
    }


def parse() -> dict[str, str]:
    evaluator = legacy_module(HALLU).FreeTextHaluEvaluator
    return {
        raw: evaluator._map_judge_outputs([[{"generated_text": raw}]])[0]
        for raw in PARSE_VECTORS
    }


def aggregate() -> dict[str, Any]:
    hallu = legacy_module(HALLU)
    legacy = hallu.FreeTextHaluEvaluator.__new__(hallu.FreeTextHaluEvaluator)
    legacy.num_samples = len(STOPS)
    legacy._format_answers = lambda answers: answers
    legacy._grade_batch = lambda *args: list(STUB_GRADES)
    saved: dict[str, Any] = {}
    legacy.save_results = lambda **kwargs: saved.update(kwargs)
    finish_reasons = ["length" if s == "max_tokens" else s for s in STOPS]
    legacy._grade_impl(
        [
            hallu._HalluGenerationRecord(
                answers=["answer"] * len(STOPS),
                input_texts=["question"] * len(STOPS),
                gt_answers=["gold"] * len(STOPS),
                finish_reasons=finish_reasons,
            )
        ],
        object(),
    )
    return {
        "stops": STOPS,
        "finish_reasons": finish_reasons,
        "stub_grades": STUB_GRADES,
        "response_grades": [r["grade"] for r in saved["responses"]],
        "saved": {k: v for k, v in saved.items() if k != "responses"},
    }


def tokenizer_gap() -> dict[str, Any]:
    """Legacy preprocessing of a long row with a character tokenizer stand-in."""
    custom = legacy_module(CUSTOM)

    class Tokenizer:
        def __call__(self, texts, *, max_length, **kwargs):
            ids = [[ord(c) for c in text[:max_length]] for text in texts]
            return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

    question, target = "q" * 1200, "a" * 300
    with patched(
        custom,
        "safe_apply_chat_template",
        lambda tokenizer, messages, **kwargs: "TEMPLATE:" + str(messages),
    ):
        legacy = custom.free_text_preprocess_function(
            {"question": [question], "answer": [target]}, Tokenizer(), 1024, 256, False
        )
    return {
        "question_length": len(question),
        "target_length": len(target),
        "test_input_ids_width": int(legacy["test_input_ids"].shape[1]),
        "gt_answers_width": int(legacy["gt_answers"].shape[1]),
        "decoded_prompt": "".join(chr(c) for c in legacy["test_input_ids"][0]),
    }


def settings() -> dict[str, Any]:
    config = legacy_module(CONFIG)
    fields = config.EvaluationConfig.model_fields
    return {
        "family_token_defaults": dict(config.FAMILY_TOKEN_DEFAULTS["hallucination"]),
        "sample_default": fields["sample"].default,
        "max_samples_default": fields["max_samples"].default,
        "dataset_seed_default": legacy_module(DATASET_CONFIG)
        .DatasetConfig.model_fields["seed"]
        .default,
    }


def prepare(legacy: Any, source: Any, *, stub_loader: bool = False) -> Any:
    """Run the legacy prepare_dataloader on ``source`` (preprocessing keeps order)."""
    base = legacy_module(BASE)
    custom = SimpleNamespace(preprocess=lambda *args, **kwargs: source)
    with patched(base, "CustomDataset", lambda *args, **kwargs: custom):
        if stub_loader:
            with patched(base, "DataLoader", lambda *args, **kwargs: None):
                base.BaseEvaluator.prepare_dataloader(legacy)
        else:
            base.BaseEvaluator.prepare_dataloader(legacy)
    return legacy.eval_dataset
