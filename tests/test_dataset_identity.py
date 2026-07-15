from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from llm_behavior_eval import (
    DatasetConfig,
    DatasetType,
    EvaluateFactory,
    expand_dataset_preset,
    list_dataset_presets,
)
from llm_behavior_eval.evaluation_utils.base_evaluator import BaseEvaluator
from llm_behavior_eval.evaluation_utils.custom_dataset import CustomDataset
from llm_behavior_eval.evaluation_utils.refusal_utils import OR_BENCH_DATASET


def test_dataset_id_defaults_to_loading_source() -> None:
    config = DatasetConfig(
        file_path="hirundo-io/halueval", dataset_type=DatasetType.BIAS
    )

    assert config.dataset_id == config.file_path


def test_dataset_id_assignment_cannot_leave_none() -> None:
    config = DatasetConfig(
        file_path="hirundo-io/halueval", dataset_type=DatasetType.BIAS
    )

    config.dataset_id = None  # type: ignore[assignment]

    assert config.dataset_id == config.file_path


def test_local_source_dispatches_by_logical_dataset_id() -> None:
    config = DatasetConfig(
        file_path="/opt/assets/halueval",
        dataset_id="hirundo-io/halueval",
        dataset_type=DatasetType.BIAS,
    )

    assert config.dataset_id is not None
    assert EvaluateFactory.get_evaluator_family(config.dataset_id) == "hallucination"


def test_output_and_provenance_retain_logical_identity() -> None:
    config = DatasetConfig(
        file_path="/opt/assets/halueval",
        dataset_id="hirundo-io/halueval",
        dataset_type=DatasetType.BIAS,
    )
    evaluator = SimpleNamespace(dataset_config=config)

    assert BaseEvaluator.get_dataset_slug(evaluator) == "halueval"  # type: ignore[arg-type]
    assert config.model_dump()["dataset_id"] == "hirundo-io/halueval"
    assert config.model_dump()["file_path"] == "/opt/assets/halueval"


def test_custom_dataset_loads_source_but_normalizes_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []

    def fake_load_dataset(source: str, **_: object) -> DatasetDict:
        loaded.append(source)
        return DatasetDict(
            {"train": Dataset.from_dict({"question": ["q"], "label": [0]})}
        )

    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.custom_dataset.load_dataset",
        fake_load_dataset,
    )
    dataset = CustomDataset(
        Path("/opt/assets/or-bench"),
        DatasetType.BIAS,
        dataset_id=OR_BENCH_DATASET,
    )

    assert loaded == ["/opt/assets/or-bench"]
    assert dataset.dataset_id == OR_BENCH_DATASET


def test_catalog_is_complete_and_expansions_are_unique() -> None:
    presets = list_dataset_presets()

    assert presets
    assert len({preset.name for preset in presets}) == len(presets)
    assert expand_dataset_preset("bias:all") == [
        f"hirundo-io/bbq-{bias_type}-bias-free-text"
        for bias_type in sorted(
            {"age", "gender", "nationality", "physical", "race", "religion"}
        )
    ]
    assert expand_dataset_preset("refusal:all") == [
        "hirundo-io/XSTest",
        "hirundo-io/or-bench",
    ]
