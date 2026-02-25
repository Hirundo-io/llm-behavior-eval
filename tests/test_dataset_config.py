from llm_behavior_eval.evaluation_utils.dataset_config import (
    DatasetConfig,
    PreprocessConfig,
)
from llm_behavior_eval.evaluation_utils.enums import AnswerFormat, DatasetType


def test_preprocess_config_defaults() -> None:
    defaults = PreprocessConfig()
    assert defaults.max_length == 1024
    assert defaults.gt_max_length == 256
    assert defaults.preprocess_batch_size == 128


def test_dataset_config_defaults_answer_format() -> None:
    defaults = DatasetConfig(
        file_path="hirundo-io/halueval",
        dataset_type=DatasetType.BIAS,
    )
    assert defaults.answer_format == AnswerFormat.FREE_TEXT
