"""Prepared LLMail-Inject records: a user's own email contexts and targets.

Each record of a JSON array or JSON Lines file holds non-empty strings
``id``, ``system``, ``query``, ``retrieved``, ``target_to`` and
``target_body``. ``retrieved`` is the whole result of the email search,
attacker email (or benign control) included, exactly as the model should see
it: nothing is filtered, ranked or formatted here. The conversation, tool and
scoring are those of ``llm_behavior_eval/llmail_inject``.
"""

from collections import Counter
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, MemoryDataset, Sample, json_dataset
from llm_behavior_eval.security.llmail_inject.scenario import conversation

FIELDS = ("id", "system", "query", "retrieved", "target_to", "target_body")
"""Required fields of a prepared record, each a non-empty string."""


def prepared_sample(record: dict[str, Any]) -> Sample:
    """One prepared record as the challenge's GPT-path conversation.

    Raises:
        ValueError: A field is missing, not a string or blank.
    """
    for field in FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Prepared LLMail-Inject record {record.get('id')!r}: {field!r} "
                f"must be a non-empty string; got {value!r}"
            )
    return Sample(
        id=record["id"],
        input=conversation(record["system"], record["query"], record["retrieved"]),
        metadata={
            "target_to": record["target_to"],
            "target_body": record["target_body"],
        },
    )


def prepared_dataset(dataset_path: str) -> Dataset:
    """Read and validate a prepared ``.json`` (array) or ``.jsonl`` file.

    Raises:
        ValueError: A record is invalid, IDs repeat, or there is no record.
    """
    samples = list(json_dataset(dataset_path, sample_fields=prepared_sample))
    if not samples:
        raise ValueError(f"{dataset_path} holds no prepared LLMail-Inject record")
    counts = Counter(sample.id for sample in samples)
    repeated = sorted(str(i) for i, count in counts.items() if count > 1)
    if repeated:
        raise ValueError(
            f"Prepared LLMail-Inject IDs must be unique; repeated: {repeated}"
        )
    return MemoryDataset(samples, name=Path(dataset_path).stem, location=dataset_path)
