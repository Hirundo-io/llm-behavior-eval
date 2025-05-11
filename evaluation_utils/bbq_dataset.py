import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from evaluation_utils.dataset_config import PreprocessConfig
from evaluation_utils.enums import DatasetType, TextFormat
from evaluation_utils.prompts import SYSTEM_PROMPT_DICT
from evaluation_utils.util_functions import safe_apply_chat_template


class CustomOpenTextExamples(TypedDict):
    question: list[str]
    answer: list[str]
    stereotyped_answer: list[str]


class CustomCloseChoiceExamples(TypedDict):
    question: list[str]
    correct_answer: list[str]
    incorrect_answers: list[list[str]]
    stereotyped_answer: list[str]


def validate_dataset_columns(hf_dataset: Dataset, text_format: TextFormat):
    """
    Validates that the dataset contains the required columns based on the text format.
    Raises a ValueError if any required columns are missing.
    """
    if text_format == TextFormat.FREE_TEXT:
        required_columns = set(CustomOpenTextExamples.__annotations__.keys())
    else:  # assume multiple-choice close text format
        required_columns = set(CustomCloseChoiceExamples.__annotations__.keys())

    dataset_columns = set(hf_dataset.column_names)
    missing = required_columns - dataset_columns
    if missing:
        raise ValueError(
            f"Dataset is missing the following required columns for {text_format} format: {missing}. "
            f"Found columns: {dataset_columns}"
        )


def open_text_preprocess_function(
    examples_batch: CustomOpenTextExamples,
    tokenizer: PreTrainedTokenizerBase,
    text_format: TextFormat,
    max_length: int,
    gt_max_length: int,
) -> dict[str, torch.Tensor]:
    """
    Preprocesses a batch of open text examples for training and evaluation by tokenizing input texts and ground truth answers.

    Args:
        examples_batch: A type dict of question and answer.
        tokenizer: The tokenizer to be used for processing the texts.
        text_format: The format of the text (free-text, structured free-text, or multiple-choice).
        max_length: The maximum length for tokenization.
        gt_max_length: The maximum length for ground truth answer.

    Returns:
        dict: A dictionary with tokenized fields:
              - "input_ids": List of tokenized input IDs with answer appended.
              - "attention_mask": List of corresponding attention masks.
              - "test_input_ids": List of tokenized input IDs without answer (for evaluation).
              - "test_attention_mask": List of corresponding attention masks for test text.
              - "gt_answers": List of tokenized ground truth answer IDs.
    """
    eval_strings, answer_strings, stereotyped_answers_strings = [], [], []

    for question, answer, stereotyped_answer in zip(
        examples_batch["question"],
        examples_batch["answer"],
        examples_batch["stereotyped_answer"],
        strict=True,
    ):
        user_msg = {"role": "user", "content": f"###Question: {question}\n"}
        system_msg = SYSTEM_PROMPT_DICT[text_format]
        # question only
        eval_conv = [system_msg, user_msg]
        eval_strings.append(safe_apply_chat_template(tokenizer, eval_conv))
        # bare answer
        answer_strings.append(answer)
        stereotyped_answers_strings.append(stereotyped_answer)

    enc_eval = tokenizer(
        eval_strings, truncation=True, padding="max_length", max_length=max_length
    )
    enc_stereotyped = tokenizer(
        stereotyped_answers_strings,
        truncation=True,
        padding="max_length",
        max_length=gt_max_length,
        add_special_tokens=False,
    )
    enc_gt = tokenizer(
        answer_strings,
        truncation=True,
        padding="max_length",
        max_length=gt_max_length,
        add_special_tokens=False,
    )
    results = {
        "test_input_ids": torch.tensor(enc_eval["input_ids"]),
        "test_attention_mask": torch.tensor(enc_eval["attention_mask"]),
        "gt_answers": torch.tensor(enc_gt["input_ids"]),
        "stereotyped_answers": torch.tensor(enc_stereotyped["input_ids"]),
    }
    return results


def close_text_preprocess_function(
    examples_batch: CustomCloseChoiceExamples,
    tokenizer: PreTrainedTokenizerBase,
    text_format: TextFormat,
    max_length: int,
    gt_max_length: int,
) -> dict[str, torch.Tensor]:
    """
    Preprocesses a batch of multiple-choice text examples for training and evaluation by tokenizing input texts and ground truth answers.
    """
    rng = np.random.default_rng()
    eval_strings: list[str] = []
    answer_strings: list[str] = []
    stereotyped_answer_strings: list[str] = []

    for question, correct_answer, incorrect_answers, stereotyped_answer in zip(
        examples_batch["question"],
        examples_batch["correct_answer"],
        examples_batch["incorrect_answers"],
        examples_batch["stereotyped_answer"],
        strict=True,
    ):
        # Randomise option order once per question
        shuffled_answers = rng.permutation(
            [correct_answer, *incorrect_answers]
        ).tolist()
        options_string = "\n".join(
            f"{i + 1}) {opt}" for i, opt in enumerate(shuffled_answers)
        )
        # Message objects used by the chat template
        user_msg = {
            "role": "user",
            "content": f"###Question: {question}\n###Options:\n{options_string}\n",
        }
        system_msg = SYSTEM_PROMPT_DICT[text_format]
        # Build evaluation (question‑only) conversation once per question
        eval_conv = [system_msg, user_msg]
        eval_string = safe_apply_chat_template(tokenizer, eval_conv)
        eval_strings.append(str(eval_string))
        answer_strings.append(correct_answer)
        stereotyped_answer_strings.append(stereotyped_answer)

    enc_eval = tokenizer.batch_encode_plus(
        eval_strings, truncation=True, padding="max_length", max_length=max_length
    )
    enc_stereotyped_answer = tokenizer.batch_encode_plus(
        stereotyped_answer_strings,
        truncation=True,
        padding="max_length",
        max_length=gt_max_length,
        add_special_tokens=False,
    )
    enc_gt = tokenizer.batch_encode_plus(
        answer_strings,
        truncation=True,
        padding="max_length",
        max_length=gt_max_length,
        add_special_tokens=False,
    )

    results = {
        "test_input_ids": torch.tensor(enc_eval["input_ids"]),
        "test_attention_mask": torch.tensor(enc_eval["attention_mask"]),
        "gt_answers": torch.tensor(enc_gt["input_ids"]),
        "stereotyped_answers": torch.tensor(enc_stereotyped_answer["input_ids"]),
    }
    return results


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
            file_path: The local path or hugging face name of the dataset csv file.
            dataset_type: The type of the dataset (e.g., BIAS or UNBIAS).
        """
        self.file_path = file_path
        self.dataset_type = dataset_type
        self.ds: Dataset = load_dataset(str(self.file_path))["train"]  # type: ignore

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
            datasets_dict: Dictionary mapping dataset split names to Hugging Face Datasets.
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
