"""Frozen legacy expectations for tests/test_inspect_medhallu.py."""

from types import SimpleNamespace
from typing import Any

from scripts.parity.removal_baseline import _hallucination as hallu

SOURCES = hallu.SOURCES

# test_parity_cap_selects_legacy_samples: 502 rows, legacy seed 42 shuffle and
# max_samples=500.
CAP_ROWS = 502
CAP = 500


def selection() -> dict[str, Any]:
    from datasets import Dataset

    source = Dataset.from_list(
        [
            {"question": f"q{i}", "answer": f"a{i}", "source_index": i}
            for i in range(CAP_ROWS)
        ]
    )
    legacy = SimpleNamespace(
        dataset_config=SimpleNamespace(
            file_path="offline",
            dataset_type=None,
            dataset_id="hirundo-io/medhallu",
            preprocess_config=None,
            seed=42,
        ),
        eval_config=SimpleNamespace(
            model_token=None,
            dataset_token=None,
            max_answer_tokens=128,
            enable_thinking=False,
            enable_thinking_arg_name="enable_thinking",
            thinking_start_token="<think>",  # noqa: S106
            thinking_end_token="</think>",  # noqa: S106
            pass_max_answer_tokens=False,
            max_samples=CAP,
        ),
        tokenizer=None,
        trust_remote_code=False,
        data_collator=None,
        eval_engine=SimpleNamespace(
            set_dataset=lambda ds: None, get_batch_size=lambda: 1
        ),
    )
    selected = hallu.prepare(legacy, source, stub_loader=True)
    return {
        "rows": CAP_ROWS,
        "max_samples": CAP,
        "source_index": list(selected["source_index"]),
        "question": list(selected["question"]),
        "answer": list(selected["answer"]),
    }


def capture() -> dict[str, Any]:
    return {
        "prompts": hallu.prompts(),
        "parse": hallu.parse(),
        "selection": selection(),
        "tokenizer_gap": hallu.tokenizer_gap(),
        "aggregate": hallu.aggregate(),
        "settings": hallu.settings(),
    }
