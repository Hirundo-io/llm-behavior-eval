import logging
from functools import partial
from pathlib import Path
from typing import Annotated, Dict, List, cast

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from evaluation_utils.dataset_config import PreprocessConfig
from evaluation_utils.enums import DatasetType, TextFormat
from evaluation_utils.prompts import SYSTEM_PROMPT_DICT
from evaluation_utils.util_functions import safe_apply_chat_template


class OpenTextRow(BaseModel):
    question: Annotated[str, Field(min_length=1)]
    answer: Annotated[str, Field(min_length=1)]
    stereotyped_answer: Annotated[str, Field(min_length=1)]


class CloseChoiceRow(BaseModel):
    question: Annotated[str, Field(min_length=1)]
    correct_answer: Annotated[str, Field(min_length=1)]
    incorrect_answers: Annotated[List[str], Field(min_length=1)]
    stereotyped_answer: Annotated[str, Field(min_length=1)]


# TypeAdapters for batch validation (list-of-rows)
OpenTextAdapter = TypeAdapter(List[OpenTextRow])
CloseChoiceAdapter = TypeAdapter(List[CloseChoiceRow])


def validate_dataset_columns(hf_dataset: Dataset, text_format: TextFormat) -> None:
    """
    Validates that the dataset contains the required columns based on the text format.
    Raises a ValueError if any required columns are missing.
    """
    if text_format == TextFormat.FREE_TEXT:
        expected = set(OpenTextRow.model_fields.keys())
    else:
        expected = set(CloseChoiceRow.model_fields.keys())

    missing = expected - set(hf_dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns for {text_format}: {missing}; found {hf_dataset.column_names}"
        )


def open_text_preprocess_function(
    examples_batch: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizerBase,
    text_format: TextFormat,
    max_length: int,
    gt_max_length: int,
) -> Dict[str, torch.Tensor]:
    # 1) Column check
    rows = [
        dict(zip(examples_batch.keys(), vals))
        for vals in zip(*examples_batch.values(), strict=True)
    ]
    try:
        OpenTextAdapter.validate_python(rows)
    except ValidationError as exc:
        raise ValueError(f"OpenText batch validation error: {exc}")

    # 2) Tokenization logic
    eval_strings, answer_strings, stereotyped_strings = [], [], []
    for row in rows:
        q, a, s = row["question"], row["answer"], row["stereotyped_answer"]
        user_msg = {"role": "user", "content": f"###Question: {q}\n"}
        system_msg = SYSTEM_PROMPT_DICT[text_format]
        eval_strings.append(safe_apply_chat_template(tokenizer, [system_msg, user_msg]))
        answer_strings.append(a)
        stereotyped_strings.append(s)

    tokenize = partial(
        tokenizer,
        truncation=True,
        padding="max_length",
    )
    enc_eval = tokenize(
        eval_strings,
        max_length=max_length,
    )
    enc_gt = tokenize(
        answer_strings,
        max_length=gt_max_length,
        add_special_tokens=False,
    )
    enc_stereo = tokenize(
        stereotyped_strings,
        max_length=gt_max_length,
        add_special_tokens=False,
    )

    return {
        "test_input_ids": torch.tensor(enc_eval["input_ids"]),
        "test_attention_mask": torch.tensor(enc_eval["attention_mask"]),
        "gt_answers": torch.tensor(enc_gt["input_ids"]),
        "stereotyped_answers": torch.tensor(enc_stereo["input_ids"]),
    }


def close_text_preprocess_function(
    examples_batch: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    text_format: TextFormat,
    max_length: int,
    gt_max_length: int,
) -> Dict[str, torch.Tensor]:
    # 1) Batch validation
    rows = [
        dict(zip(examples_batch.keys(), vals))
        for vals in zip(*examples_batch.values(), strict=True)
    ]
    try:
        CloseChoiceAdapter.validate_python(rows)
    except ValidationError as exc:
        raise ValueError(f"CloseChoice batch validation error: {exc}")

    # 2) Tokenization logic
    rng = np.random.default_rng()
    eval_strings, answer_strings, stereotyped_strings = [], [], []

    for row in rows:
        opts = [row["correct_answer"], *row["incorrect_answers"]]
        shuffled = rng.permutation(opts).tolist()
        opt_str = "\n".join(f"{i + 1}) {o}" for i, o in enumerate(shuffled))

        user_msg = {
            "role": "user",
            "content": f"###Question: {row['question']}\n###Options:\n{opt_str}\n",
        }
        system_msg = SYSTEM_PROMPT_DICT[text_format]
        eval_strings.append(safe_apply_chat_template(tokenizer, [system_msg, user_msg]))
        answer_strings.append(row["correct_answer"])
        stereotyped_strings.append(row["stereotyped_answer"])

    tokenize = partial(
        tokenizer,
        truncation=True,
        padding="max_length",
    )
    enc_eval = tokenize(
        eval_strings,
        max_length=max_length,
    )
    enc_gt = tokenize(
        answer_strings,
        max_length=gt_max_length,
        add_special_tokens=False,
    )
    enc_stereo = tokenize(
        stereotyped_strings,
        max_length=gt_max_length,
        add_special_tokens=False,
    )

    return {
        "test_input_ids": torch.tensor(enc_eval["input_ids"]),
        "test_attention_mask": torch.tensor(enc_eval["attention_mask"]),
        "gt_answers": torch.tensor(enc_gt["input_ids"]),
        "stereotyped_answers": torch.tensor(enc_stereo["input_ids"]),
    }


class BBQDataset:
    """
    A custom dataset that loads data from a CSV file having only the fields "question" and "answer",
    and only supports free-text or structured-free-text formats.
    """

    def __init__(
        self,
        file_path: Path | str,
        dataset_type: DatasetType,
    ):
        """
        Initializes the custom dataset with a specified dataset type and bias type.

        Args:
            file_path: The local path or HuggingFace name of the dataset csv file.
            dataset_type: The type of the dataset (e.g., BIAS or UNBIAS).
        """
        self.file_path = file_path
        self.dataset_type = dataset_type
        # Type "DatasetDict | Dataset | IterableDatasetDict | IterableDataset" is not assignable to declared type "DatasetDict"
        # To ignore this error, since the loaded dataset is always a DatasetDict
        # and we are using the "cast" function to ensure type safety.
        try:
            raw: DatasetDict = load_dataset(str(self.file_path))  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load dataset '{self.file_path}'. "
                "Check that the identifier is correct."
            ) from exc
        self.ds = cast(Dataset, raw["train"])

    def preprocess(
        self,
        tokenizer: PreTrainedTokenizerBase,
        text_format: TextFormat,
        preprocess_config: PreprocessConfig,
    ) -> Dataset:
        """
        Preprocess custom datasets by tokenizing texts based on the given text format.

        Applies the preprocess_function to each dataset split. The function tokenizes both the answer-inclusive
        and answer-exclusive texts along with the ground truth answers.

        Args:
            datasets_dict: Dictionary mapping dataset split names to HuggingFace Datasets.
            tokenizer: Tokenizer used for text processing.
            text_format: Format of the text (free-text, structured free-text, or multiple-choice).
            preprocess_config: Configuration for preprocessing the dataset.

        Returns:
            A test dataset with tokenized fields
        """
        preprocess_function = (
            open_text_preprocess_function
            if text_format == TextFormat.FREE_TEXT
            else close_text_preprocess_function
        )
        validate_dataset_columns(self.ds, text_format)
        old_columns = self.ds.column_names
        processed_dataset = self.ds.map(
            lambda examples: preprocess_function(
                examples,
                tokenizer,
                text_format,
                max_length=preprocess_config.max_length,
                gt_max_length=preprocess_config.gt_max_length,
            ),
            batched=True,
            remove_columns=old_columns,
            batch_size=preprocess_config.preprocess_batch_size,
            num_proc=1,
        )
        text = tokenizer.batch_decode(
            processed_dataset["test_input_ids"], skip_special_tokens=True
        )
        logging.info("Validation text: %s", text[0])
        return processed_dataset
