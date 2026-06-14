from pathlib import Path
from typing import Any, cast

from datasets import Dataset

REFUSAL_PLACEHOLDER_ANSWER = "placeholder"
SAFE_REFUSAL_LABEL = "safe"
UNSAFE_REFUSAL_LABEL = "unsafe"
OR_BENCH_DATASET = "hirundo-io/or-bench"
XSTEST_DATASET = "hirundo-io/XSTest"
REFUSAL_DATASETS = {XSTEST_DATASET, OR_BENCH_DATASET}


def is_refusal_dataset(file_path: Path | str) -> bool:
    return str(file_path) in REFUSAL_DATASETS


def normalize_refusal_label(raw_label: Any) -> str:
    label = str(raw_label).strip().lower()
    if label not in {SAFE_REFUSAL_LABEL, UNSAFE_REFUSAL_LABEL}:
        raise ValueError(
            "Refusal dataset label must be "
            f"{SAFE_REFUSAL_LABEL!r} or {UNSAFE_REFUSAL_LABEL!r}, got {raw_label!r}"
        )
    return label


def normalize_refusal_dataset(hf_dataset: Dataset) -> Dataset:
    """Normalize refusal datasets to the free-text schema expected downstream."""

    def _normalize_batch(
        examples_batch: dict[str, list[Any]],
    ) -> dict[str, list[str] | list[int]]:
        prompts = examples_batch.get("question")
        if prompts is None:
            prompts = examples_batch.get("prompt")
        if prompts is None:
            raise ValueError(
                "Refusal dataset must contain either a 'question' or 'prompt' column"
            )
        labels = examples_batch.get("label")
        if labels is None:
            raise ValueError("Refusal dataset must contain a 'label' column")

        normalized_labels = [
            1 if normalize_refusal_label(label) == UNSAFE_REFUSAL_LABEL else 0
            for label in labels
        ]
        answers = examples_batch.get("answer")
        if answers is None:
            # Free-text preprocessing expects an "answer" column, but our refusal
            # datasets currently only provide prompt + safe/unsafe label.
            answers = [REFUSAL_PLACEHOLDER_ANSWER for _ in prompts]
        else:
            answers = [
                answer if str(answer).strip() else REFUSAL_PLACEHOLDER_ANSWER
                for answer in answers
            ]

        return {
            "question": [str(prompt) for prompt in prompts],
            "answer": [str(answer) for answer in answers],
            "label": normalized_labels,
        }

    return cast(
        "Dataset",
        hf_dataset.map(
            _normalize_batch,
            batched=True,
            remove_columns=hf_dataset.column_names,
        ),
    )
