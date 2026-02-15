from datasets import Dataset, DatasetDict

from llm_behavior_eval.evaluation_utils.custom_dataset import (
    CustomDataset,
    free_text_preprocess_raw_function,
)
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


def test_free_text_preprocess_raw_function_truncates_to_limits() -> None:
    batch = {
        "question": ["one two three four five six"],
        "answer": ["a b c d e"],
        "stereotyped_answer": ["s1 s2 s3 s4"],
        "judge_question": ["j1 j2 j3 j4"],
    }

    processed = free_text_preprocess_raw_function(
        batch,
        has_stereotype=True,
        max_length=3,
        gt_max_length=2,
    )

    assert processed["questions"] == ["one two three"]
    assert processed["input_texts"] == ["one two three"]
    assert processed["gt_answers"] == ["a b"]
    assert processed["stereotyped_answers"] == ["s1 s2"]
    assert processed["judge_questions"] == ["j1 j2"]
    assert processed["test_messages"][0][1]["content"] == "one two three\n"


def test_free_text_preprocess_raw_function_uses_custom_text_truncator() -> None:
    batch = {
        "question": ["question text"],
        "answer": ["answer text"],
        "stereotyped_answer": ["stereotype text"],
        "judge_question": ["judge text"],
        "system_prompt": ["system text"],
    }
    calls: list[tuple[str, int]] = []

    def custom_truncator(text: str, max_tokens: int) -> str:
        calls.append((text, max_tokens))
        return f"[{max_tokens}]{text[:1]}"

    processed = free_text_preprocess_raw_function(
        batch,
        has_stereotype=True,
        max_length=10,
        gt_max_length=3,
        text_truncator=custom_truncator,
    )

    assert processed["questions"] == ["[10]q"]
    assert processed["input_texts"] == ["[10]q"]
    assert processed["gt_answers"] == ["[3]a"]
    assert processed["stereotyped_answers"] == ["[3]s"]
    assert processed["judge_questions"] == ["[3]j"]
    assert processed["test_messages"][0][0]["content"] == "[10]s"
    assert processed["test_messages"][0][1]["content"] == "[10]q\n"
    assert ("question text", 10) in calls
    assert ("answer text", 3) in calls
    assert ("stereotype text", 3) in calls
    assert ("judge text", 3) in calls
    assert ("system text", 10) in calls
