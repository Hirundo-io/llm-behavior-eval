"""Frozen legacy expectations for tests/test_inspect_halueval.py."""

from typing import Any

from scripts.parity.historical import legacy_module
from scripts.parity.removal_baseline import _hallucination as hallu

SOURCES = hallu.SOURCES

# test_limit_selects_legacy_samples: 30 rows, legacy max_samples=7.
LIMIT_ROWS = 30
LIMIT = 7


def selection() -> dict[str, Any]:
    from datasets import Dataset

    rows = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(LIMIT_ROWS)]
    evaluator = legacy_module(hallu.HALLU).FreeTextHaluEvaluator
    legacy = evaluator.__new__(evaluator)
    legacy.dataset_config = legacy_module(hallu.DATASET_CONFIG).DatasetConfig(
        file_path="hirundo-io/halueval",
        dataset_type=legacy_module(
            "llm_behavior_eval.evaluation_utils.enums"
        ).DatasetType.UNBIAS,
    )
    legacy.eval_config = type(
        "EvalConfig",
        (),
        {
            "max_samples": LIMIT,
            "model_token": None,
            "dataset_token": None,
            "max_answer_tokens": None,
            "enable_thinking": None,
            "enable_thinking_arg_name": None,
            "thinking_start_token": None,
            "thinking_end_token": None,
            "pass_max_answer_tokens": None,
        },
    )()
    legacy.trust_remote_code = False
    legacy.tokenizer = None
    legacy.data_collator = None

    class Engine:
        def set_dataset(self, dataset):
            pass

        def get_batch_size(self):
            return 1

    legacy.eval_engine = Engine()
    selected = hallu.prepare(legacy, Dataset.from_list(rows))
    return {"rows": LIMIT_ROWS, "limit": LIMIT, "questions": list(selected["question"])}


def capture() -> dict[str, Any]:
    return {
        "prompts": hallu.prompts(),
        "parse": hallu.parse(),
        "selection": selection(),
        "tokenizer_gap": hallu.tokenizer_gap(),
        "aggregate": hallu.aggregate(),
        "settings": hallu.settings(),
    }
