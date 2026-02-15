from datasets import Dataset, DatasetDict

from llm_behavior_eval.evaluation_utils.custom_dataset import CustomDataset
from llm_behavior_eval.evaluation_utils.enums import DatasetType


def _make_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "question": ["q1"],
            "answer": ["a1"],
        }
    )


def test_custom_dataset_loads_from_dataset_dict(
    monkeypatch,
) -> None:
    dataset = _make_dataset()

    def mock_load_dataset(*args, **kwargs):
        return DatasetDict({"train": dataset})

    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.custom_dataset.load_dataset",
        mock_load_dataset,
    )

    custom_dataset = CustomDataset("dummy", DatasetType.BIAS)
    assert len(custom_dataset.ds) == 1
    assert custom_dataset.ds["question"][0] == "q1"


def test_custom_dataset_falls_back_to_hub_tabular_on_not_implemented(
    monkeypatch,
) -> None:
    dataset = _make_dataset()
    calls = {"fallback_called": False}

    def mock_load_dataset(*args, **kwargs):
        raise NotImplementedError("cache backend issue")

    def mock_fallback(self, original_exc):
        calls["fallback_called"] = True
        return dataset

    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.custom_dataset.load_dataset",
        mock_load_dataset,
    )
    monkeypatch.setattr(
        CustomDataset,
        "_load_dataset_hub_tabular_fallback",
        mock_fallback,
    )

    custom_dataset = CustomDataset("dummy", DatasetType.BIAS)
    assert len(custom_dataset.ds) == 1
    assert calls["fallback_called"] is True
