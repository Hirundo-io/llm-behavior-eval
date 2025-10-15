import os

from pydantic_settings import BaseSettings, SettingsConfigDict

from .enums import DatasetType

CPU_COUNT = os.cpu_count() or 1


class PreprocessConfig(BaseSettings):
    """
    PreprocessConfig is a configuration class for defining the settings of a dataset preprocessing including tokenization, batching and the train labels.

    Attributes:
        max_length: The maximum length of the text data.
        gt_max_length: The maximum length for ground truth data.
        preprocess_batch_size: The batch size for preprocessing the dataset.
        num_proc: Number of parallel processes for HuggingFace Dataset.map; 1 disables multiprocessing.
        skip_decode_validation: If True, skip post-tokenization batch_decode used only for logging sanity checks.
    """

    model_config = SettingsConfigDict(env_prefix="bias_preprocess_")

    max_length: int = 1024
    gt_max_length: int = 256
    preprocess_batch_size: int = 128
    num_proc: int = CPU_COUNT // 2
    skip_decode_validation: bool = True


class DatasetConfig(BaseSettings):
    """
    DatasetConfig is a configuration class for defining the settings of a dataset.

    Attributes:
        file_path: The HuggingFace repo id of the dataset file.
        dataset_type: The type of the dataset, represented as an enum.
        preprocess_config: Configuration for preprocessing the dataset.
        seed: The random seed for reproducibility.
    """

    model_config = SettingsConfigDict(env_prefix="bias_dataset_")

    file_path: str
    dataset_type: DatasetType
    preprocess_config: PreprocessConfig = PreprocessConfig()
    seed: int = 42
