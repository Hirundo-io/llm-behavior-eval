"""Canonical HF identity, the legacy seeded shuffle, and per-sample messages.

The legacy engine shuffled the complete split with seed 42 before capping it at
``max_samples``. Conversion reproduces that order (``common.dataset.legacy_order``),
so Inspect's sample limit (``eval_config.limit`` in ``parity.yaml``) selects the
same rows.
"""

from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_behavior_eval.common.dataset import (
    LEGACY_SHUFFLE_SEED,
    legacy_order,
    load_hf_dataset,
    to_inspect_dataset,
)
from llm_behavior_eval.prompt_injection.prompts import SYSTEM_PROMPT

DATASET_ID = "hirundo-io/prompt-injection-purple-llama"
_INDEX_COLUMN = "__injection_index"
_REQUIRED = ("question", "answer")


class InjectionRow(BaseModel):
    """The source fields this benchmark reads; other columns are ignored."""

    model_config = ConfigDict(strict=True, extra="ignore")

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    system_prompt: str | None = None
    judge_question: str | None = None


def validate(records: Any) -> None:
    """Reject the whole source if any row lacks required or typed fields."""
    for row in records:
        try:
            InjectionRow.model_validate(row)
        except ValidationError as exc:
            field = exc.errors()[0]["loc"][0]
            kind = "required nonempty text" if field in _REQUIRED else "optional text"
            raise ValueError(f"{DATASET_ID}: {kind} {field}") from exc


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
    revision: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
) -> MemoryDataset:
    """Validate the whole source, then convert every row in legacy shuffle order.

    ``revision`` pins the Hub dataset revision; ``seed`` is the shuffle seed.
    """
    records = load_hf_dataset(
        DATASET_ID,
        dataset_path=dataset_path,
        revision=revision,
    )
    validate(records)
    records = records.add_column(_INDEX_COLUMN, list(range(len(records)))).select(
        legacy_order(len(records), seed=seed)
    )
    return to_inspect_dataset(records, record_to_sample, dataset_id=DATASET_ID)
