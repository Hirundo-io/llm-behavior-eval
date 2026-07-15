from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .enums import DatasetType


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
        file_path: Local path or Hugging Face repository used to load the dataset.
        dataset_id: Stable logical identity used for evaluator selection and results.
            Defaults to ``file_path`` for backward compatibility.
        dataset_type: The type of the dataset, represented as an enum.
        preprocess_config: Configuration for preprocessing the dataset.
        seed: The random seed for reproducibility.
    """

    model_config = SettingsConfigDict(
        env_prefix="bias_dataset_", validate_assignment=True, validate_default=True
    )

    file_path: str
    dataset_id: str = ""
    dataset_type: DatasetType
    preprocess_config: PreprocessConfig = PreprocessConfig()
    seed: int | None = 42

    @field_validator("dataset_id", mode="before")
    @classmethod
    def default_dataset_id(cls, value: object, info: ValidationInfo) -> object:
        """Use the loading source as the logical identity for legacy callers."""
        if value is None or value == "":
            return info.data["file_path"]
        return value
