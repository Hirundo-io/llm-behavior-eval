import logging
from collections.abc import Callable
from copy import copy
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, DatasetDict, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .dataset_config import PreprocessConfig
from .enums import DatasetType
from .prompts import SYSTEM_PROMPT_DICT
from .util_functions import truncate_text_by_whitespace


def validate_dataset_columns(hf_dataset: Dataset) -> None:
    """
    Validates that the dataset contains the required columns based on the text format.
    Raises a ValueError if any required columns are missing.
    """
    # Minimum required columns for free-text
    required = {"question", "answer"}

    missing = required - set(hf_dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}; found {hf_dataset.column_names}"
        )


def free_text_preprocess_raw_function(
    examples_batch: dict[str, list[str]],
    has_stereotype: bool,
    max_length: int,
    gt_max_length: int,
    text_truncator: Callable[[str, int], str] | None = None,
) -> dict[str, list[Any]]:
    """
    Preprocess a batch for API models without a tokenizer.

    Builds message lists and keeps ground-truth fields as strings.
    Since no tokenizer is available in this path, truncation is approximated by
    limiting whitespace-delimited tokens.
    """

    truncate_text = text_truncator or truncate_text_by_whitespace

    rows = [
        dict(zip(examples_batch.keys(), vals, strict=True))
        for vals in zip(*examples_batch.values(), strict=True)
    ]
    for row in rows:
        if not row.get("question") or not row.get("answer"):
            raise ValueError("Free text row must contain 'question' and 'answer'")

    test_messages: list[list[dict[str, str]]] = []
    questions: list[str] = []
    answer_strings: list[str] = []
    stereotyped_strings: list[str] = []
    judge_questions: list[str] = []

    for row in rows:
        question_text = truncate_text(row["question"], max_length)
        answer_text = truncate_text(row["answer"], gt_max_length)
        stereotyped_text = (
            truncate_text(row.get("stereotyped_answer") or "", gt_max_length)
            if has_stereotype
            else None
        )
        system_override = row.get("system_prompt")
        judge_question_override = truncate_text(
            row.get("judge_question") or row["question"],
            gt_max_length,
        )
        if system_override:
            system_override = truncate_text(system_override, max_length)

        user_msg = {"role": "user", "content": f"{question_text}\n"}
        system_msg = (
            {"role": "system", "content": system_override}
            if system_override
            else copy(SYSTEM_PROMPT_DICT)
        )
        test_messages.append([system_msg, user_msg])
        questions.append(question_text)
        answer_strings.append(answer_text)
        if has_stereotype:
            stereotyped_strings.append(stereotyped_text or "")
        judge_questions.append(judge_question_override)

    result: dict[str, list[Any]] = {
        "test_messages": test_messages,
        "questions": questions,
        "input_texts": questions,
        "gt_answers": answer_strings,
        "judge_questions": judge_questions,
    }
    if has_stereotype:
        result["stereotyped_answers"] = stereotyped_strings
    return result


class CustomDataset:
    """
    A custom dataset that loads data from a CSV file having only the fields "question" and "answer",
    """

    def __init__(
        self,
        file_path: Path | str,
        dataset_type: DatasetType,
    ):
        """
        Initializes the custom dataset with a specified dataset type and behavior type.

        Args:
            file_path: The local path or HuggingFace name of the dataset csv file.
            dataset_type: The type of the dataset (e.g., BIAS or UNBIAS).
        """
        self.file_path = file_path
        self.dataset_type = dataset_type
        try:
            raw = load_dataset(str(self.file_path))
        except NotImplementedError as exc:
            raw = self._load_dataset_hub_tabular_fallback(exc)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to load dataset '{self.file_path}'. "
                "Check that the identifier is correct."
            ) from exc
        if isinstance(raw, DatasetDict):
            self.ds = cast("Dataset", raw["train"])
        elif isinstance(raw, Dataset):
            self.ds = raw
        else:
            raise ValueError(f"Expected DatasetDict or Dataset, got {type(raw)}")
        self.has_stereotype: bool = "stereotyped_answer" in self.ds.column_names

    @staticmethod
    def _select_preferred_tabular_file(tabular_files: list[str]) -> str:
        """Pick the most likely training split file from dataset repo files."""
        if not tabular_files:
            raise ValueError("tabular_files must not be empty")

        exact_train_names = {"train.csv", "train.parquet", "train.json", "train.jsonl"}

        def rank(path: str) -> tuple[int, str]:
            file_name = Path(path).name.lower()
            stem = Path(path).stem.lower()
            parts = [part.lower() for part in Path(path).parts]

            if file_name in exact_train_names:
                return (0, file_name)
            if file_name.startswith(("train-", "train_", "train.")):
                return (1, file_name)
            if stem == "train" or "train" in parts:
                return (2, file_name)
            if file_name.startswith(("validation", "valid", "dev")):
                return (4, file_name)
            if file_name.startswith("test"):
                return (5, file_name)
            return (3, file_name)

        return min(tabular_files, key=rank)

    def _load_dataset_hub_tabular_fallback(self, original_exc: Exception) -> Dataset:
        """
        Fallback for environments where `datasets.load_dataset` cannot read
        cached LocalFileSystem datasets.
        """
        dataset_ref = str(self.file_path)
        logging.warning(
            "Primary dataset loading failed for '%s' (%s). Falling back to direct CSV loading.",
            dataset_ref,
            original_exc,
        )
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError as exc:
            raise RuntimeError(
                "Fallback dataset loading requires `huggingface_hub`."
            ) from exc

        try:
            repo_files = list_repo_files(dataset_ref, repo_type="dataset")
            tabular_files = [
                f
                for f in repo_files
                if f.lower().endswith((".parquet", ".csv", ".json", ".jsonl"))
            ]
            if not tabular_files:
                raise RuntimeError(
                    f"No supported dataset files found in dataset repository '{dataset_ref}'."
                )
            preferred_file = self._select_preferred_tabular_file(tabular_files)
            local_file_path = hf_hub_download(
                repo_id=dataset_ref,
                filename=preferred_file,
                repo_type="dataset",
            )
            import pandas as pd

            lower_name = preferred_file.lower()
            if lower_name.endswith(".parquet"):
                df = pd.read_parquet(local_file_path)
            elif lower_name.endswith(".csv"):
                df = pd.read_csv(local_file_path)
            else:
                df = pd.read_json(local_file_path, lines=lower_name.endswith(".jsonl"))
            return Dataset.from_pandas(df, preserve_index=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_ref}' via fallback file path."
            ) from exc

    def preprocess(
        self,
        tokenizer: PreTrainedTokenizerBase | None,
        preprocess_config: PreprocessConfig,
        trust_remote_code: bool = False,
        max_answer_tokens: int | None = None,
        reasoning: bool = False,
        pass_max_answer_tokens: bool = False,
        token: str | None = None,
        raw_text_truncator: Callable[[str, int], str] | None = None,
    ) -> Dataset:
        """
        Preprocess custom datasets into raw message/text fields.

        This path is raw-text only and is consumed by eval engines that format/tokenize
        prompts at generation time. Unused tokenizer-era parameters are retained for
        call-site compatibility within this PR scope.

        Args:
            tokenizer: Unused; retained for compatibility.
            preprocess_config: Configuration for preprocessing the dataset.
            trust_remote_code: Unused; retained for compatibility.
            max_answer_tokens: Unused; retained for compatibility.
            reasoning: Unused; retained for compatibility.
            pass_max_answer_tokens: Unused; retained for compatibility.
            token: Unused; retained for compatibility.

        Returns:
            A test dataset with raw message and text fields.
        """
        del tokenizer
        del trust_remote_code
        del max_answer_tokens
        del reasoning
        del pass_max_answer_tokens
        del token

        validate_dataset_columns(self.ds)
        old_columns = self.ds.column_names
        processed_dataset = self.ds.map(
            lambda examples: free_text_preprocess_raw_function(
                examples,
                has_stereotype=self.has_stereotype,
                max_length=preprocess_config.max_length,
                gt_max_length=preprocess_config.gt_max_length,
                text_truncator=raw_text_truncator,
            ),
            batched=True,
            remove_columns=old_columns,
            batch_size=preprocess_config.preprocess_batch_size,
            num_proc=1,
        )
        if len(processed_dataset) > 0:
            logging.info("Validation text: %s", processed_dataset["questions"][0])
        return processed_dataset
