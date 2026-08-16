from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers.data.data_collator import default_data_collator

import llm_behavior_eval.evaluation_utils.censorship_utils as censorship_utils_module
import llm_behavior_eval.evaluation_utils.custom_dataset as custom_dataset_module
import llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator as censorship_evaluator_module
from llm_behavior_eval.evaluation_utils.censorship_utils import (
    CENSORSHIP_JUDGE_PROMPT,
    CENSORSHIP_JUDGE_PROMPT_SHA256,
    CENSORSHIP_JUDGE_RUBRIC_VERSION,
    CHINESE_CENSORSHIP_DATASET_CONFIG,
    CHINESE_CENSORSHIP_DATASET_ID,
    CHINESE_CENSORSHIP_DATASET_REVISION,
    CHINESE_CENSORSHIP_DATASET_SOURCE,
    CHINESE_CENSORSHIP_DATASET_SPLIT,
    CHINESE_CENSORSHIP_GROUP_COUNT,
    CensorshipJudgment,
    parse_censorship_judgment,
)
from llm_behavior_eval.evaluation_utils.custom_dataset import (
    CustomDataset,
    free_text_preprocess_function,
    validate_dataset_columns,
)
from llm_behavior_eval.evaluation_utils.dataset_config import (
    DatasetConfig,
    PreprocessConfig,
)
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator import (
    FreeTextCensorshipEvaluator,
)
from llm_behavior_eval.evaluation_utils.refusal_utils import (
    OR_BENCH_DATASET,
    REFUSAL_PLACEHOLDER_ANSWER,
    normalize_refusal_dataset,
)
from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine


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
    assert normalized["label"] == [1]
    validate_dataset_columns(normalized)


def test_normalize_refusal_dataset_preserves_existing_answers():
    ds = Dataset.from_dict({"question": ["q1"], "answer": ["given"], "label": ["safe"]})

    normalized = normalize_refusal_dataset(ds)

    assert normalized["answer"] == ["given"]
    assert normalized["label"] == [0]


def test_normalize_refusal_dataset_rejects_unknown_labels():
    ds = Dataset.from_dict({"prompt": ["q1"], "label": ["maybe"]})

    with pytest.raises(ValueError, match="must be 'safe' or 'unsafe'"):
        normalize_refusal_dataset(ds)


def _censorship_rows() -> Dataset:
    ids = [f"locked-{index:03d}" for index in range(288)]
    return Dataset.from_dict(
        {
            "question": [f"q-{group_id}" for group_id in ids],
            "source_group_id": ids,
        }
    )


def test_dedicated_censorship_contract_is_pinned() -> None:
    assert CHINESE_CENSORSHIP_GROUP_COUNT == 288
    assert CHINESE_CENSORSHIP_DATASET_CONFIG == "default"
    assert CHINESE_CENSORSHIP_DATASET_SPLIT == "test"
    assert CHINESE_CENSORSHIP_DATASET_REVISION == (
        "bffb7b76440ed69fdb1792a20e6efac298ab2d94"
    )


def test_censorship_judge_contract_is_exact_and_strict():
    assert CENSORSHIP_JUDGE_RUBRIC_VERSION == "chinese_censorship_binary"
    assert (
        CENSORSHIP_JUDGE_PROMPT_SHA256
        == censorship_utils_module.hashlib.sha256(
            CENSORSHIP_JUDGE_PROMPT.encode()
        ).hexdigest()
    )
    assert parse_censorship_judgment(" True\n") == CensorshipJudgment.CENSORED
    assert parse_censorship_judgment("False") == CensorshipJudgment.NOT_CENSORED
    assert (
        parse_censorship_judgment("The verdict is True")
        == CensorshipJudgment.UNPARSEABLE
    )


def test_custom_dataset_loads_pinned_censorship_without_refusal_injection(
    monkeypatch: pytest.MonkeyPatch,
):
    test = _censorship_rows()
    captured: dict[str, object] = {}

    def fake_load_dataset(path, **kwargs):
        captured.update({"path": path, **kwargs})
        return DatasetDict({"test": test})

    monkeypatch.setattr(custom_dataset_module, "load_dataset", fake_load_dataset)

    custom_dataset = CustomDataset(
        CHINESE_CENSORSHIP_DATASET_SOURCE,
        DatasetType.BIAS,
        trust_remote_code=True,
        dataset_id=CHINESE_CENSORSHIP_DATASET_ID,
    )

    assert captured == {
        "path": CHINESE_CENSORSHIP_DATASET_SOURCE,
        "name": CHINESE_CENSORSHIP_DATASET_CONFIG,
        "revision": CHINESE_CENSORSHIP_DATASET_REVISION,
        "token": None,
        "trust_remote_code": False,
    }
    assert len(custom_dataset.dataset) == 288
    assert set(custom_dataset.dataset.column_names) == {
        "question",
        "censorship_group_index",
    }
    assert "answer" not in custom_dataset.dataset.column_names
    assert "label" not in custom_dataset.dataset.column_names
    assert custom_dataset.censorship_group_ids == list(test["source_group_id"])


def test_censorship_dataset_rejects_noncanonical_source() -> None:
    with pytest.raises(ValueError, match="pinned dataset source"):
        CustomDataset(
            "unapproved/dataset",
            DatasetType.BIAS,
            dataset_id=CHINESE_CENSORSHIP_DATASET_ID,
        )


def test_censorship_dataset_requires_pinned_test_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        custom_dataset_module,
        "load_dataset",
        lambda *_args, **_kwargs: DatasetDict({"train": _censorship_rows()}),
    )

    with pytest.raises(ValueError, match="pinned 'test' split"):
        CustomDataset(
            CHINESE_CENSORSHIP_DATASET_SOURCE,
            DatasetType.BIAS,
            dataset_id=CHINESE_CENSORSHIP_DATASET_ID,
        )


def test_censorship_preprocessing_and_generation_preserve_group_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubTokenizer:
        name_or_path = "fake/model"

        def __call__(self, texts, **_kwargs):
            if isinstance(texts, str):
                texts = [texts]
            return {
                "input_ids": [[1, 2, 3] for _ in texts],
                "attention_mask": [[1, 1, 1] for _ in texts],
            }

    dataset = _censorship_rows()
    monkeypatch.setattr(
        custom_dataset_module,
        "load_dataset",
        lambda *_args, **_kwargs: DatasetDict({"test": dataset}),
    )
    monkeypatch.setattr(
        custom_dataset_module, "is_model_multimodal", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        custom_dataset_module,
        "safe_apply_chat_template",
        lambda *_args, **_kwargs: "rendered prompt",
    )

    custom_dataset = CustomDataset(
        CHINESE_CENSORSHIP_DATASET_SOURCE,
        DatasetType.BIAS,
        dataset_id=CHINESE_CENSORSHIP_DATASET_ID,
    )
    processed = custom_dataset.preprocess(
        cast("PreTrainedTokenizerBase", StubTokenizer()),
        PreprocessConfig(max_length=8, gt_max_length=4, preprocess_batch_size=32),
    )

    assert set(processed.column_names) == {
        "test_input_ids",
        "test_attention_mask",
        "censorship_group_index",
    }
    shuffled = cast("Dataset", processed.shuffle(seed=17))
    shuffled_group_indexes = cast("list[int]", shuffled["censorship_group_index"])
    evaluator = object.__new__(FreeTextCensorshipEvaluator)
    evaluator.censorship_group_ids = custom_dataset.censorship_group_ids
    evaluator.censorship_questions = custom_dataset.censorship_questions
    evaluator.eval_group_ids = [
        custom_dataset.censorship_group_ids[index] for index in shuffled_group_indexes
    ]
    evaluator.num_samples = len(shuffled)
    evaluator.eval_loader = DataLoader(
        cast("TorchDataset", shuffled),
        batch_size=64,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    monkeypatch.setattr(
        evaluator, "ensure_test_model_ready", lambda: None, raising=False
    )
    monkeypatch.setattr(evaluator, "load_completed_generation_dicts", lambda: [])
    monkeypatch.setattr(
        evaluator,
        "generate_answers",
        lambda input_ids, attention_mask: (
            ["answer"] * len(input_ids),
            ["stop"] * len(attention_mask),
        ),
    )
    monkeypatch.setattr(evaluator, "save_generations", lambda items: None)

    generations = evaluator._collect_generations()

    assert [
        group_id
        for generation in generations
        for group_id in generation.source_group_ids
    ] == evaluator.eval_group_ids
    assert [
        question for generation in generations for question in generation.questions
    ] == [
        custom_dataset.censorship_questions[
            custom_dataset.censorship_group_ids.index(group_id)
        ]
        for group_id in evaluator.eval_group_ids
    ]


def test_censorship_preparation_preserves_generation_batch_size_during_grading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep model batch provenance when the judge replaces the model engine."""

    class StubDataset:
        censorship_group_ids = [
            f"locked-{index:03d}" for index in range(CHINESE_CENSORSHIP_GROUP_COUNT)
        ]
        censorship_questions = ["question"] * CHINESE_CENSORSHIP_GROUP_COUNT

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def preprocess(self, *_args: object, **_kwargs: object) -> Dataset:
            return Dataset.from_dict(
                {
                    "test_input_ids": [[1]] * CHINESE_CENSORSHIP_GROUP_COUNT,
                    "test_attention_mask": [[1]] * CHINESE_CENSORSHIP_GROUP_COUNT,
                    "censorship_group_index": list(
                        range(CHINESE_CENSORSHIP_GROUP_COUNT)
                    ),
                }
            )

    class StubEngine:
        def __init__(self, batch_size: int, is_judge: bool) -> None:
            self.batch_size = batch_size
            self.is_judge = is_judge

        def set_dataset(self, _dataset: Dataset) -> None:
            pass

        def get_batch_size(self) -> int:
            return self.batch_size

    monkeypatch.setattr(censorship_evaluator_module, "CustomDataset", StubDataset)
    evaluator = object.__new__(FreeTextCensorshipEvaluator)
    evaluator.dataset_config = DatasetConfig(
        file_path=CHINESE_CENSORSHIP_DATASET_SOURCE,
        dataset_id=CHINESE_CENSORSHIP_DATASET_ID,
        dataset_type=DatasetType.BIAS,
        seed=None,
    )
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=0,
        sampling_config=SamplingConfig(seed=17),
    )
    evaluator.trust_remote_code = False
    evaluator.tokenizer = cast("PreTrainedTokenizerBase", object())
    evaluator.data_collator = default_data_collator
    evaluator.eval_engine = cast("EvalEngine", StubEngine(batch_size=8, is_judge=False))

    evaluator.prepare_dataloader()
    generation_group_ids = evaluator.eval_group_ids
    evaluator.eval_engine = cast("EvalEngine", StubEngine(batch_size=2, is_judge=True))
    evaluator.prepare_dataloader()

    assert evaluator.effective_batch_size == 8
    assert evaluator.eval_group_ids == generation_group_ids


def test_custom_dataset_load_failure_surfaces_underlying_error(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_load_dataset(_path, **_kwargs):
        raise OSError("401 Client Error: Unauthorized for private repo")

    monkeypatch.setattr(custom_dataset_module, "load_dataset", fake_load_dataset)

    with pytest.raises(RuntimeError, match="401 Client Error") as exc_info:
        CustomDataset(OR_BENCH_DATASET, DatasetType.BIAS)

    assert "--model-token" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, OSError)


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
        {"question": ["q1"], "answer": ["a1"], "label": [0]},
        cast("PreTrainedTokenizerBase", StubTokenizer()),
        max_length=8,
        gt_max_length=4,
        has_stereotype=False,
        include_default_system_prompt=False,
    )

    assert captured_messages == [[{"role": "user", "content": "q1\n"}]]


def test_free_text_preprocess_function_emits_refusal_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubTokenizer:
        def __call__(self, texts, **_kwargs):
            if isinstance(texts, str):
                texts = [texts]
            return {
                "input_ids": [[1, 2] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }

    monkeypatch.setattr(
        custom_dataset_module,
        "safe_apply_chat_template",
        lambda *_args, **_kwargs: "formatted",
    )

    result = free_text_preprocess_function(
        {"question": ["q1"], "answer": ["a1"], "label": [1]},
        cast("PreTrainedTokenizerBase", StubTokenizer()),
        max_length=8,
        gt_max_length=4,
        has_stereotype=False,
        include_default_system_prompt=False,
    )

    assert "refusal_labels" in result
    assert "label" not in result
    assert result["refusal_labels"].tolist() == [1]


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


def test_custom_dataset_passes_auth_args_to_load_dataset(
    monkeypatch: pytest.MonkeyPatch,
):
    ds = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    captured: dict[str, object] = {}

    def fake_load_dataset(path, *, token=None, trust_remote_code=False):
        captured["path"] = path
        captured["token"] = token
        captured["trust_remote_code"] = trust_remote_code
        return DatasetDict({"train": ds})

    monkeypatch.setattr(custom_dataset_module, "load_dataset", fake_load_dataset)

    CustomDataset(
        "repo/gated-dataset",
        DatasetType.BIAS,
        trust_remote_code=True,
        token="hf_test_token",
    )

    assert captured == {
        "path": "repo/gated-dataset",
        "token": "hf_test_token",
        "trust_remote_code": True,
    }


def test_custom_dataset_uses_train_split_when_present(
    monkeypatch: pytest.MonkeyPatch,
):
    ds = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    monkeypatch.setattr(
        custom_dataset_module,
        "load_dataset",
        lambda _path, **_kwargs: DatasetDict({"train": ds, "test": ds}),
    )

    custom_dataset = CustomDataset("repo/dataset", DatasetType.BIAS)

    assert custom_dataset.dataset == ds


def test_custom_dataset_falls_back_to_only_available_split(
    monkeypatch: pytest.MonkeyPatch,
):
    ds = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    monkeypatch.setattr(
        custom_dataset_module,
        "load_dataset",
        lambda _path, **_kwargs: DatasetDict({"test": ds}),
    )

    custom_dataset = CustomDataset("repo/dataset", DatasetType.BIAS)

    assert custom_dataset.dataset == ds


@pytest.mark.parametrize(
    ("dataset_id", "dataset", "expected_messages"),
    [
        (
            OR_BENCH_DATASET,
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
        lambda _path, **_kwargs: DatasetDict({"train": dataset}),
    )
    monkeypatch.setattr(
        custom_dataset_module, "is_model_multimodal", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        custom_dataset_module, "safe_apply_chat_template", fake_safe_apply_chat_template
    )

    load_source = (
        "/opt/assets/or-bench" if dataset_id == OR_BENCH_DATASET else dataset_id
    )
    custom_dataset = CustomDataset(load_source, DatasetType.BIAS, dataset_id=dataset_id)
    if dataset_id == OR_BENCH_DATASET:
        monkeypatch.setattr(
            custom_dataset_module,
            "normalize_refusal_dataset",
            lambda _dataset: StubMappedDataset(
                {
                    "question": ["q1"],
                    "answer": [REFUSAL_PLACEHOLDER_ANSWER],
                    "label": [0],
                }
            ),
        )
    else:
        custom_dataset.dataset = cast(
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
