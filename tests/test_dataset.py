import pytest
from datasets import Dataset

from llm_behavior_eval.evaluation_utils.bbq_dataset import validate_dataset_columns


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
    ds = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    with pytest.raises(ValueError):
        validate_dataset_columns(ds)
