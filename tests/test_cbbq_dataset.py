import pytest
from datasets import Dataset

from llm_behavior_eval.evaluation_utils.cbbq_dataset import validate_cbbq_columns


def test_validate_cbbq_columns_rejects_null_or_empty_required_text() -> None:
    dataset = Dataset.from_dict(
        {
            "context": ["valid context", None],
            "question": ["valid question", "   "],
            "ans0": ["A1", "A2"],
            "ans1": ["B1", "B2"],
            "ans2": ["C1", "C2"],
            "label": [0, 1],
            "question_polarity": ["neg", "non_neg"],
        }
    )

    with pytest.raises(ValueError, match="null/empty required text values"):
        validate_cbbq_columns(dataset)
