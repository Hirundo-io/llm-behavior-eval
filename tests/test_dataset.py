import pytest
from datasets import Dataset

from evaluation_utils.bbq_dataset import validate_dataset_columns
from evaluation_utils.enums import TextFormat


def test_validate_dataset_columns_pass_free_text():
    ds = Dataset.from_dict(
        {
            "question": ["q"],
            "answer": ["a"],
            "stereotyped_answer": ["s"],
        }
    )
    validate_dataset_columns(ds, TextFormat.FREE_TEXT)


def test_validate_dataset_columns_fail():
    ds = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    with pytest.raises(ValueError):
        validate_dataset_columns(ds, TextFormat.FREE_TEXT)
