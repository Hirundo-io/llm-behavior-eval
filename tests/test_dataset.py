import pytest
from datasets import Dataset, DatasetDict

import llm_behavior_eval.evaluation_utils.custom_dataset as custom_dataset_module
from llm_behavior_eval.evaluation_utils.custom_dataset import (
    REFUSAL_PLACEHOLDER_ANSWER,
    CustomDataset,
    normalize_refusal_dataset,
    validate_dataset_columns,
)
from llm_behavior_eval.evaluation_utils.enums import DatasetType


def test_validate_dataset_columns_pass_free_text():
    ds = Dataset.from_dict(
        {
            "question": ["q"],
            "answer": ["a"],
            "stereotyped_answer": ["s"],
        }
    )
    validate_dataset_columns(ds)


def test_validate_dataset_columns_fail():
    ds = Dataset.from_dict({"question": ["q"]})
    with pytest.raises(ValueError):
        validate_dataset_columns(ds)


def test_normalize_refusal_dataset_maps_prompt_and_label_columns():
    ds = Dataset.from_dict({"prompt": ["q1"], "label": ["unsafe"]})

    normalized = normalize_refusal_dataset(ds)

    assert set(normalized.column_names) == {"question", "answer", "label"}
    assert normalized["question"] == ["q1"]
    assert normalized["answer"] == [REFUSAL_PLACEHOLDER_ANSWER]
    assert normalized["label"] == ["unsafe"]
    validate_dataset_columns(normalized)


def test_normalize_refusal_dataset_preserves_existing_answers():
    ds = Dataset.from_dict({"question": ["q1"], "answer": ["given"], "label": ["safe"]})

    normalized = normalize_refusal_dataset(ds)

    assert normalized["answer"] == ["given"]
    assert normalized["label"] == ["safe"]


def test_normalize_refusal_dataset_rejects_unknown_labels():
    ds = Dataset.from_dict({"prompt": ["q1"], "label": ["maybe"]})

    with pytest.raises(ValueError, match="must be 'safe' or 'unsafe'"):
        normalize_refusal_dataset(ds)


def test_custom_dataset_uses_train_split_when_present(
    monkeypatch: pytest.MonkeyPatch,
):
    ds = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    monkeypatch.setattr(
        custom_dataset_module,
        "load_dataset",
        lambda _path: DatasetDict({"train": ds, "test": ds}),
    )

    custom_dataset = CustomDataset("repo/dataset", DatasetType.BIAS)

    assert custom_dataset.ds == ds


def test_custom_dataset_falls_back_to_only_available_split(
    monkeypatch: pytest.MonkeyPatch,
):
    ds = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    monkeypatch.setattr(
        custom_dataset_module,
        "load_dataset",
        lambda _path: DatasetDict({"test": ds}),
    )

    custom_dataset = CustomDataset("repo/dataset", DatasetType.BIAS)

    assert custom_dataset.ds == ds
