from pydantic_settings import BaseSettings, SettingsConfigDict

from .enums import AnswerFormat, DatasetType


class PreprocessConfig(BaseSettings):
    """
    PreprocessConfig is a configuration class for defining the settings of a dataset preprocessing including tokenization, batching and the train labels.

    Attributes:
        max_length: The maximum length of the text data.
        gt_max_length: The maximum length for ground truth data.
        preprocess_batch_size: The batch size for preprocessing the dataset.
    """

    model_config = SettingsConfigDict(env_prefix="bias_preprocess_")

    max_length: int = 1024
    gt_max_length: int = 256
    preprocess_batch_size: int = 128


class DatasetConfig(BaseSettings):
    """
    DatasetConfig is a configuration class for defining the settings of a dataset.

    Attributes:
        file_path: The HuggingFace repo id of the dataset file.
        dataset_type: The type of the dataset, represented as an enum.
        preprocess_config: Configuration for preprocessing the dataset.
        answer_format: Response format expected by the evaluator.
        seed: The random seed for reproducibility.
    """

    model_config = SettingsConfigDict(env_prefix="bias_dataset_")

    file_path: str
    dataset_type: DatasetType
    preprocess_config: PreprocessConfig = PreprocessConfig()
    answer_format: AnswerFormat = AnswerFormat.FREE_TEXT
    seed: int | None = 42
