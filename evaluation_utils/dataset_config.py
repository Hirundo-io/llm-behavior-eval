from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from evaluation_utils.enums import DatasetType, TextFormat


class PreprocessConfig(BaseSettings):
    """
    PreprocessConfig is a configuration class for defining the settings of a dataset preprocessing including tokenization, batching and the train labels.

    Attributes:
        max_length: The maximum length of the text data.
        gt_max_length: The maximum length for ground truth data.
        preprocess_batch_size: The batch size for preprocessing the dataset.
    """

    model_config = SettingsConfigDict(env_prefix="bias_preprocess_")

    max_length: int = 512
    gt_max_length: int = 32
    preprocess_batch_size: int = 128


class DatasetConfig(BaseSettings):
    """
    DatasetConfig is a configuration class for defining the settings of a dataset.

    Attributes:
        file_path: The hugging face repo id of the dataset file.
        dataset_type: The type of the dataset, represented as an enum.
        text_format: The format of the text in the dataset.
        preprocess_config: Configuration for preprocessing the dataset.
        seed: The random seed for reproducibility.
    """

    model_config = SettingsConfigDict(env_prefix="bias_dataset_")

    file_path: str
    dataset_type: DatasetType
    text_format: TextFormat
    preprocess_config: PreprocessConfig = PreprocessConfig()
    seed: int = 42
