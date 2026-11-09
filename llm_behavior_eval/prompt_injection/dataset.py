"""Canonical HF identity, the legacy seeded shuffle, and per-sample messages.

The legacy engine shuffled the complete split with seed 42 before capping it at
``max_samples``. Conversion reproduces that order, so Inspect's sample limit
(``eval_config.limit`` in ``parity.yaml``) selects the same rows.
"""

from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset
from llm_behavior_eval.prompt_injection.prompts import SYSTEM_PROMPT

DATASET_ID = "hirundo-io/prompt-injection-purple-llama"
LEGACY_SHUFFLE_SEED = 42
_INDEX_COLUMN = "__injection_index"


def validate(records: Any) -> None:
    """Reject the whole source if any row lacks required or typed fields."""
    for row in records:
        for key in ("question", "answer"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{DATASET_ID}: required nonempty text {key}")
        for key in ("system_prompt", "judge_question"):
            if row.get(key) is not None and not isinstance(row[key], str):
                raise ValueError(f"{DATASET_ID}: optional text {key}")


def record_to_sample(row: dict[str, Any]) -> Sample:
    """Legacy messages: row system prompt or the default, then ``question + "\\n"``.

    Empty or missing overrides fall back exactly as the legacy preprocessor did.
    """
    system = row.get("system_prompt") or SYSTEM_PROMPT
    judge_question = row.get("judge_question")
    index = row[_INDEX_COLUMN]
    return Sample(
        id=f"{DATASET_ID}:{index}",
        input=[
            ChatMessageSystem(content=system),
            ChatMessageUser(content=row["question"] + "\n"),
        ],
        target=row["answer"],
        metadata=dict(
            dataset_id=DATASET_ID,
            source_index=index,
            question=row["question"],
            system_prompt=system,
            judge_question=judge_question or row["question"],
            judge_question_source="override" if judge_question else "question",
        ),
    )


def dataset(
    *,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
) -> MemoryDataset:
    records = load_hf_dataset(
        DATASET_ID, dataset_path=dataset_path, dataset_token=dataset_token
    )
    validate(records)
    records = records.add_column(_INDEX_COLUMN, list(range(len(records)))).shuffle(
        seed=seed
    )
    return to_inspect_dataset(records, record_to_sample, dataset_id=DATASET_ID)
