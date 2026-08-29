from typing import Literal

import pytest
from pydantic import ValidationError

from llm_behavior_eval.evaluation_utils.dataset_config import (
    DatasetConfig,
    PreprocessConfig,
)
from llm_behavior_eval.evaluation_utils.enums import DatasetType


def test_preprocess_config_defaults() -> None:
    defaults = PreprocessConfig()
    assert defaults.max_length == 1024
    assert defaults.gt_max_length == 256
    assert defaults.preprocess_batch_size == 128


@pytest.mark.parametrize("ccpc_source_mode", [None, "historical"])
@pytest.mark.parametrize(
    ("expected_row_count", "expected_sha256"),
    [
        (500, None),
        (None, "a" * 64),
        (500, "a" * 64),
    ],
)
def test_historical_ccpc_source_rejects_local_contracts(
    ccpc_source_mode: Literal["historical"] | None,
    expected_row_count: int | None,
    expected_sha256: str | None,
) -> None:
    with pytest.raises(ValidationError, match="apply only when"):
        DatasetConfig(
            file_path="chinese_censorship",
            dataset_type=DatasetType.BIAS,
            ccpc_source_mode=ccpc_source_mode,
            expected_row_count=expected_row_count,
            expected_sha256=expected_sha256,
        )


def test_local_ccpc_source_requires_expected_row_count() -> None:
    with pytest.raises(ValidationError, match="requires expected_row_count"):
        DatasetConfig(
            file_path="chinese_censorship",
            dataset_id="chinese_censorship",
            dataset_type=DatasetType.BIAS,
            ccpc_source_mode="local",
        )


def test_local_ccpc_source_allows_optional_sha256() -> None:
    config = DatasetConfig(
        file_path="chinese_censorship",
        dataset_id="chinese_censorship",
        dataset_type=DatasetType.BIAS,
        ccpc_source_mode="local",
        expected_row_count=500,
        expected_sha256="a" * 64,
    )
    assert config.expected_row_count == 500
    assert config.expected_sha256 == "a" * 64


def test_omitted_and_historical_source_modes_accept_absent_local_contracts() -> None:
    omitted = DatasetConfig(
        file_path="chinese_censorship", dataset_type=DatasetType.BIAS
    )
    historical = DatasetConfig(
        file_path="chinese_censorship",
        dataset_type=DatasetType.BIAS,
        ccpc_source_mode="historical",
    )
    assert omitted.ccpc_source_mode is None
    assert omitted.expected_row_count is None
    assert omitted.expected_sha256 is None
    assert historical.ccpc_source_mode == "historical"
    assert historical.expected_row_count is None
    assert historical.expected_sha256 is None
