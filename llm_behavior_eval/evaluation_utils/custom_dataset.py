import logging
from copy import copy
from functools import partial
from pathlib import Path
from typing import cast

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .dataset_config import PreprocessConfig
from .enums import DatasetType
from .prompts import SYSTEM_PROMPT_DICT
from .util_functions import is_model_multimodal, safe_apply_chat_template


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


def free_text_preprocess_function(
    examples_batch: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    gt_max_length: int,
    has_stereotype: bool,
    is_multimodal: bool = False,
    max_answer_tokens: int | None = None,
    reasoning: bool = False,
    pass_max_answer_tokens: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Preprocesses a batch of examples for free-text datasets.

    Args:
        examples_batch: A batch of examples to preprocess.
        tokenizer: The tokenizer to use for tokenization.
        max_length: The maximum length of the input sequence.
        gt_max_length: The maximum length of the ground truth sequence.
        has_stereotype: Whether the dataset has stereotyped answers.
        is_multimodal: Whether the model is multimodal.
        max_answer_tokens: The maximum number of tokens to allow for the answer.
        reasoning: Whether to use reasoning.
        pass_max_answer_tokens: Whether to pass max_answer_tokens to the chat template.

    Returns:
        A dictionary containing the tokenized input and ground truth sequences.
    """
    # 1) Column check
    rows = [
        dict(zip(examples_batch.keys(), vals, strict=True))
        for vals in zip(*examples_batch.values(), strict=True)
    ]
    # Validate minimally required fields only
    for row in rows:
        if not row.get("question") or not row.get("answer"):
            raise ValueError("Free text row must contain 'question' and 'answer'")
    # 2) Apply chat template to dataset messages
    eval_strings, answer_strings = [], []
    stereotyped_strings: list[str] = []
    judge_questions: list[str] = []
    for row in rows:
        question_text = row["question"]
        answer_text = row["answer"]
        stereotyped_text = row.get("stereotyped_answer") if has_stereotype else None
        # Optional overrides for prompt-injection datasets
        system_override = row.get("system_prompt")
        judge_question_override = row.get("judge_question")

        user_msg = {"role": "user", "content": f"{question_text}\n"}
        system_msg = (
            {"role": "system", "content": system_override}
            if system_override
            else copy(SYSTEM_PROMPT_DICT)
        )
        eval_strings.append(
            safe_apply_chat_template(
                tokenizer,
                [system_msg, user_msg],
                is_multimodal=is_multimodal,
                max_answer_tokens=max_answer_tokens,
                reasoning=reasoning,
                pass_max_answer_tokens=pass_max_answer_tokens,
            )
        )
        answer_strings.append(answer_text)
        if has_stereotype:
            stereotyped_strings.append(stereotyped_text or "")
        judge_questions.append(judge_question_override or question_text)
    # 3) Tokenization
    tokenize = partial(
        tokenizer,
        truncation=True,
        padding="max_length",
    )
    tokenized_eval = tokenize(
        eval_strings,
        max_length=max_length,
    )
    tokenized_gt = tokenize(
        answer_strings,
        max_length=gt_max_length,
        add_special_tokens=False,
    )
    tokenized_judge_questions = tokenize(
        judge_questions,
        max_length=gt_max_length,
        add_special_tokens=False,
    )
    tokenized_stereotype = None
    if has_stereotype:
        tokenized_stereotype = tokenize(
            stereotyped_strings,
            max_length=gt_max_length,
            add_special_tokens=False,
        )
    # 4) Result
    result = {
        "test_input_ids": torch.tensor(tokenized_eval["input_ids"]),
        "test_attention_mask": torch.tensor(tokenized_eval["attention_mask"]),
        "gt_answers": torch.tensor(tokenized_gt["input_ids"]),
        "judge_questions": torch.tensor(tokenized_judge_questions["input_ids"]),
    }
    if has_stereotype and tokenized_stereotype is not None:
        result["stereotyped_answers"] = torch.tensor(tokenized_stereotype["input_ids"])

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
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to load dataset '{self.file_path}'. "
                "Check that the identifier is correct."
            ) from exc
        if not isinstance(raw, DatasetDict):
            raise ValueError(f"Expected DatasetDict, got {type(raw)}")
        self.ds = cast("Dataset", raw["train"])
        self.has_stereotype: bool = "stereotyped_answer" in self.ds.column_names

    def preprocess(
        self,
        tokenizer: PreTrainedTokenizerBase,
        preprocess_config: PreprocessConfig,
        trust_remote_code: bool = False,
        max_answer_tokens: int | None = None,
        reasoning: bool = False,
        pass_max_answer_tokens: bool = False,
        token: str | None = None,
    ) -> Dataset:
        """
        Preprocess custom datasets by tokenizing texts based on the given text format.

        Applies the preprocess_function to each dataset split. The function tokenizes both the answer-inclusive
        and answer-exclusive texts along with the ground truth answers.

        Args:
            datasets_dict: Dictionary mapping dataset split names to HuggingFace Datasets.
            tokenizer: Tokenizer used for text processing.
            preprocess_config: Configuration for preprocessing the dataset.
            trust_remote_code: Whether to trust remote code.
            max_answer_tokens: Maximum number of tokens to allow for the answer.
            reasoning: Whether to use reasoning.
            pass_max_answer_tokens: Whether to pass max_answer_tokens to the chat template.
            token: The HuggingFace token to use for accessing gated models.

        Returns:
            A test dataset with tokenized fields
        """
        preprocess_function = free_text_preprocess_function
        validate_dataset_columns(self.ds)
        old_columns = self.ds.column_names
        # Compute once to avoid per-batch remote config lookups
        is_multimodal = is_model_multimodal(
            tokenizer.name_or_path, trust_remote_code, token
        )
        processed_dataset = self.ds.map(
            lambda examples: preprocess_function(
                examples,
                tokenizer,
                max_length=preprocess_config.max_length,
                gt_max_length=preprocess_config.gt_max_length,
                has_stereotype=self.has_stereotype,
                is_multimodal=is_multimodal,
                max_answer_tokens=max_answer_tokens,
                reasoning=reasoning,
                pass_max_answer_tokens=pass_max_answer_tokens,
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
