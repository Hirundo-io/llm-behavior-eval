"""Canonical HF identity, legacy shuffle, and per-sample native messages."""

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset
from llm_behavior_eval.common.templates import FREE_TEXT_SYSTEM_MESSAGE

DATASET_ID = "hirundo-io/prompt-injection-purple-llama"


def dataset(
    *,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    max_samples: int | None = 500,
    seed: int = 42,
) -> MemoryDataset:
    records = load_hf_dataset(
        DATASET_ID, dataset_path=dataset_path, dataset_token=dataset_token
    )
    for row in records:
        for key in ("question", "answer"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{DATASET_ID}: required nonempty text {key}")
        for key in ("system_prompt", "judge_question"):
            if row.get(key) is not None and not isinstance(row[key], str):
                raise ValueError(f"{DATASET_ID}: optional text {key}")
    records = records.add_column(
        "__injection_index", list(range(len(records)))
    ).shuffle(seed=seed)

    def convert(row: dict) -> Sample:
        system = row.get("system_prompt") or FREE_TEXT_SYSTEM_MESSAGE["content"]
        return Sample(
            id=f"{DATASET_ID}:{row['__injection_index']}",
            input=[
                ChatMessageSystem(content=system),
                ChatMessageUser(content=row["question"] + "\n"),
            ],
            target=row["answer"],
            metadata=dict(
                dataset_id=DATASET_ID,
                source_index=row["__injection_index"],
                question=row["question"],
                system_prompt=system,
                judge_question=row.get("judge_question") or row["question"],
                judge_question_source="override"
                if row.get("judge_question")
                else "question",
            ),
        )

    return to_inspect_dataset(
        records, convert, dataset_id=DATASET_ID, max_samples=max_samples
    )
