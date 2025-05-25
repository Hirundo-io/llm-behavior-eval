import pytest
pytest.importorskip("torch")
import torch

from llm_behavior_eval.evaluation_utils.base_evaluator import custom_collator


def test_custom_collator_padding() -> None:
    batch = [
        {"ids": [1, 2], "mask": [1]},
        {"ids": [3, 4, 5], "mask": [1, 1]},
    ]
    result = custom_collator(batch)
    assert result["ids"].shape == (2, 3)
    assert result["mask"].shape == (2, 2)
    assert torch.equal(result["ids"][0], torch.tensor([1, 2, 0]))
    assert torch.equal(result["ids"][1], torch.tensor([3, 4, 5]))
