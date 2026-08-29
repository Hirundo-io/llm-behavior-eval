from typing import Literal

from pydantic import ValidationInfo, field_validator, model_validator
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
        dataset_revision: HuggingFace revision pinning the loaded dataset.
        ccpc_source_mode: Explicit CCPC cohort source selection. ``None``
            (omitted from run identity) is the reusable default; CCPC local
            cohorts set ``"local"``. ``"historical"`` is accepted as an alias
            of the omitted historical CCPC source.
        expected_row_count: Optional explicit local-cohort row-count contract.
        expected_sha256: Optional exact local-cohort SHA-256 contract.
    """

    model_config = SettingsConfigDict(
        env_prefix="bias_dataset_", validate_assignment=True, validate_default=True
    )

    file_path: str
    dataset_id: str = ""
    dataset_type: DatasetType
    preprocess_config: PreprocessConfig = PreprocessConfig()
    seed: int | None = 42
    dataset_revision: str | None = None
    ccpc_source_mode: Literal["historical", "local"] | None = None
    expected_row_count: int | None = None
    expected_sha256: str | None = None

    @field_validator("dataset_id", mode="before")
    @classmethod
    def default_dataset_id(cls, value: object, info: ValidationInfo) -> object:
        """Use the loading source as the logical identity for legacy callers."""
        if value is None or value == "":
            return info.data.get("file_path", value)
        return value

    @model_validator(mode="after")
    def validate_ccpc_source_contracts(self) -> "DatasetConfig":
        """Reject mixed historical and local CCPC source contracts.

        Returns:
            This config when historical and local contracts do not mix.

        Raises:
            ValueError: If a local source omits ``expected_row_count``, or a
                historical/omitted source includes local row-count or SHA fields.
        """
        local_contract_present = (
            self.expected_row_count is not None or self.expected_sha256 is not None
        )
        if self.ccpc_source_mode == "local":
            if self.expected_row_count is None:
                raise ValueError(
                    "An explicit local CCPC-Bench source requires expected_row_count."
                )
            return self
        if local_contract_present:
            raise ValueError(
                "expected_row_count and expected_sha256 apply only when "
                "ccpc_source_mode='local'."
            )
        return self
