from typing import TYPE_CHECKING, cast

import pytest
from datasets import Dataset, DatasetDict

import llm_behavior_eval.evaluation_utils.custom_dataset as custom_dataset_module
from llm_behavior_eval.evaluation_utils.custom_dataset import (
    REFUSAL_PLACEHOLDER_ANSWER,
    CustomDataset,
    free_text_preprocess_function,
    is_refusal_dataset,
    normalize_refusal_dataset,
    validate_dataset_columns,
)
from llm_behavior_eval.evaluation_utils.enums import DatasetType

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


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


def test_is_refusal_dataset_matches_supported_refusal_datasets():
    assert is_refusal_dataset("walledai/XSTest")
    assert is_refusal_dataset("hirundo-io/or-bench")
    assert not is_refusal_dataset("hirundo-io/halueval")


def test_free_text_preprocess_function_omits_default_system_prompt_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_messages: list[list[dict[str, str]]] = []

    class StubTokenizer:
        def __call__(self, texts, **_kwargs):
            if isinstance(texts, str):
                texts = [texts]
            return {
                "input_ids": [[1, 2] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }

    def fake_safe_apply_chat_template(_tokenizer, messages, **_kwargs):
        captured_messages.append(messages)
        return "formatted"

    monkeypatch.setattr(
        custom_dataset_module, "safe_apply_chat_template", fake_safe_apply_chat_template
    )

    free_text_preprocess_function(
        {"question": ["q1"], "answer": ["a1"], "label": ["safe"]},
        cast("PreTrainedTokenizerBase", StubTokenizer()),
        max_length=8,
        gt_max_length=4,
        has_stereotype=False,
        include_default_system_prompt=False,
    )

    assert captured_messages == [[{"role": "user", "content": "q1\n"}]]


def test_free_text_preprocess_function_uses_default_system_prompt_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_messages: list[list[dict[str, str]]] = []

    class StubTokenizer:
        def __call__(self, texts, **_kwargs):
            if isinstance(texts, str):
                texts = [texts]
            return {
                "input_ids": [[1, 2] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }

    def fake_safe_apply_chat_template(_tokenizer, messages, **_kwargs):
        captured_messages.append(messages)
        return "formatted"

    monkeypatch.setattr(
        custom_dataset_module, "safe_apply_chat_template", fake_safe_apply_chat_template
    )

    free_text_preprocess_function(
        {"question": ["q1"], "answer": ["a1"]},
        cast("PreTrainedTokenizerBase", StubTokenizer()),
        max_length=8,
        gt_max_length=4,
        has_stereotype=False,
        include_default_system_prompt=True,
    )

    assert captured_messages == [
        [
            custom_dataset_module.SYSTEM_PROMPT_DICT,
            {"role": "user", "content": "q1\n"},
        ]
    ]


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


@pytest.mark.parametrize(
    ("dataset_id", "dataset", "expected_messages"),
    [
        (
            "hirundo-io/or-bench",
            Dataset.from_dict({"prompt": ["q1"], "label": ["safe"]}),
            [[{"role": "user", "content": "q1\n"}]],
        ),
        (
            "hirundo-io/halueval",
            Dataset.from_dict({"question": ["q1"], "answer": ["a1"]}),
            [
                [
                    custom_dataset_module.SYSTEM_PROMPT_DICT,
                    {"role": "user", "content": "q1\n"},
                ]
            ],
        ),
    ],
)
def test_custom_dataset_preprocess_switches_default_system_prompt_by_dataset_family(
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: str,
    dataset: Dataset,
    expected_messages: list[list[dict[str, str]]],
):
    captured_messages: list[list[dict[str, str]]] = []

    class StubMappedDataset:
        def __init__(self, payload: dict[str, list[object]]) -> None:
            self.payload = payload
            self.column_names = list(payload.keys())

        def map(self, function, **_kwargs):
            return StubMappedDataset(function(self.payload))

        def __getitem__(self, key: str) -> list[object]:
            return self.payload[key]

    class StubTokenizer:
        name_or_path = "fake/model"

        def __call__(self, texts, **_kwargs):
            if isinstance(texts, str):
                texts = [texts]
            return {
                "input_ids": [[1, 2] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }

        def batch_decode(self, sequences, **_kwargs):
            return ["decoded" for _ in sequences]

    def fake_safe_apply_chat_template(_tokenizer, messages, **_kwargs):
        captured_messages.append(messages)
        return "formatted"

    monkeypatch.setattr(
        custom_dataset_module,
        "load_dataset",
        lambda _path: DatasetDict({"train": dataset}),
    )
    monkeypatch.setattr(custom_dataset_module, "is_model_multimodal", lambda *_: False)
    monkeypatch.setattr(
        custom_dataset_module, "safe_apply_chat_template", fake_safe_apply_chat_template
    )

    custom_dataset = CustomDataset(dataset_id, DatasetType.BIAS)
    if dataset_id == "hirundo-io/or-bench":
        monkeypatch.setattr(
            custom_dataset_module,
            "normalize_refusal_dataset",
            lambda _dataset: StubMappedDataset(
                {
                    "question": ["q1"],
                    "answer": [REFUSAL_PLACEHOLDER_ANSWER],
                    "label": ["safe"],
                }
            ),
        )
    else:
        custom_dataset.ds = cast(
            "Dataset",
            StubMappedDataset({"question": ["q1"], "answer": ["a1"]}),
        )
    custom_dataset.preprocess(
        cast("PreTrainedTokenizerBase", StubTokenizer()),
        custom_dataset_module.PreprocessConfig(
            max_length=8,
            gt_max_length=4,
            preprocess_batch_size=1,
        ),
    )

    assert captured_messages == expected_messages
