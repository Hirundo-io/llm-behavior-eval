import logging
from copy import copy
from functools import partial
from pathlib import Path
from typing import cast

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .censorship_utils import (
    CHINESE_CENSORSHIP_DATASET_CONFIG,
    CHINESE_CENSORSHIP_DATASET_ID,
    CHINESE_CENSORSHIP_DATASET_REVISION,
    CHINESE_CENSORSHIP_DATASET_SOURCE,
    CHINESE_CENSORSHIP_DATASET_SPLIT,
)
from .dataset_config import PreprocessConfig
from .enums import DatasetType
from .prompts import SYSTEM_PROMPT_DICT
from .refusal_utils import REFUSAL_DATASETS, normalize_refusal_dataset
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


def censorship_preprocess_function(
    examples_batch: dict[str, list[str] | list[int]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    is_multimodal: bool = False,
    max_answer_tokens: int | None = None,
    enable_thinking: bool = False,
    enable_thinking_arg_name: str | None = None,
    pass_max_answer_tokens: bool = False,
    thinking_start_token: str | None = None,
    thinking_end_token: str | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize censorship prompts without refusal labels or ground-truth answers.

    Args:
        examples_batch: Batch of locked benchmark questions and group indexes.
        tokenizer: Tokenizer used to render and encode user prompts.
        max_length: Maximum input token length.
        is_multimodal: Whether the tokenizer requires multimodal prompt formatting.
        max_answer_tokens: Optional answer-length hint for supported chat templates.
        enable_thinking: Whether to enable the model's reasoning mode.
        enable_thinking_arg_name: Template argument used to enable reasoning mode.
        pass_max_answer_tokens: Whether to pass the answer-length hint to the template.
        thinking_start_token: Optional token that starts a reasoning trace.
        thinking_end_token: Optional token that ends a reasoning trace.

    Returns:
        Token IDs, attention masks, and stable censorship group indexes.

    Raises:
        ValueError: If a batch lacks a non-empty question.
    """
    questions = [str(value) for value in examples_batch["question"]]
    if not questions or any(not question for question in questions):
        raise ValueError("Censorship rows must contain a non-empty 'question'")
    rendered = [
        safe_apply_chat_template(
            tokenizer,
            [{"role": "user", "content": f"{question}\n"}],
            is_multimodal=is_multimodal,
            max_answer_tokens=max_answer_tokens,
            enable_thinking=enable_thinking,
            enable_thinking_arg_name=enable_thinking_arg_name,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
            pass_max_answer_tokens=pass_max_answer_tokens,
        )
        for question in questions
    ]
    tokenized = tokenizer(
        rendered, truncation=True, padding="max_length", max_length=max_length
    )
    return {
        "test_input_ids": torch.tensor(tokenized["input_ids"]),
        "test_attention_mask": torch.tensor(tokenized["attention_mask"]),
        "censorship_group_index": torch.tensor(
            examples_batch["censorship_group_index"], dtype=torch.long
        ),
    }


def free_text_preprocess_function(
    examples_batch: dict[str, list[str] | list[int]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    gt_max_length: int,
    has_stereotype: bool,
    is_multimodal: bool = False,
    max_answer_tokens: int | None = None,
    enable_thinking: bool = False,
    enable_thinking_arg_name: str | None = None,
    pass_max_answer_tokens: bool = False,
    thinking_start_token: str | None = None,
    thinking_end_token: str | None = None,
    include_default_system_prompt: bool = True,
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
        enable_thinking: Whether to enable thinking.
        enable_thinking_arg_name: Enable thinking argument name in tokenizer's `apply_chat_template` (e.g. 'enable_thinking').
        thinking_start_token: Thinking start token to use for the model (e.g. '<think>').
        thinking_end_token: Thinking end token to use for the model (e.g. '</think>').
        pass_max_answer_tokens: Whether to pass max_answer_tokens to the chat template.
        include_default_system_prompt: Whether to prepend the shared free-text system prompt
            when no per-row system prompt override is provided.

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
        messages = [user_msg]
        if system_override:
            messages = [
                {"role": "system", "content": system_override},
                user_msg,
            ]
        elif include_default_system_prompt:
            messages = [copy(SYSTEM_PROMPT_DICT), user_msg]
        eval_strings.append(
            safe_apply_chat_template(
                tokenizer,
                messages,
                is_multimodal=is_multimodal,
                max_answer_tokens=max_answer_tokens,
                enable_thinking=enable_thinking,
                enable_thinking_arg_name=enable_thinking_arg_name,
                thinking_start_token=thinking_start_token,
                thinking_end_token=thinking_end_token,
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
    if "label" in examples_batch:
        result["refusal_labels"] = torch.tensor(
            examples_batch["label"], dtype=torch.long
        )

    return result


class CustomDataset:
    """
    Loads a HuggingFace dataset for free-text evaluation.

    Supports optional columns such as ``stereotyped_answer``, ``system_prompt``,
    ``judge_question``, and ``label`` (for refusal benchmarks).
    """

    def __init__(
        self,
        file_path: Path | str,
        dataset_type: DatasetType,
        trust_remote_code: bool = False,
        token: str | None = None,
        dataset_id: str | None = None,
    ) -> None:
        """Initialize the custom dataset with a specified dataset type.

        ``chinese_censorship`` loads its dedicated pinned dataset directly. Other
        datasets use the normal split selection and preprocessing path.

        Args:
            file_path: Local path or HuggingFace dataset identifier.
            dataset_type: The type of the dataset (e.g., BIAS or UNBIAS).
            trust_remote_code: Whether to trust remote code when loading the dataset.
            token: HuggingFace token for gated or private dataset repos.
            dataset_id (optional): Logical dataset identity. Defaults to ``file_path``.

        Raises:
            RuntimeError: If the dataset cannot be loaded (e.g. an unknown
                identifier or a missing token for a private repo). The original
                error is chained as the cause.
            ValueError: If the loaded dataset does not satisfy its split or schema
                contract.
        """
        self.file_path = file_path
        self.dataset_id = dataset_id or str(file_path)
        self.dataset_type = dataset_type
        self.trust_remote_code = trust_remote_code
        self.token = token
        self.dataset: Dataset
        self.censorship_group_ids: list[str] = []
        self.censorship_questions: list[str] = []
        is_censorship_dataset = self.dataset_id == CHINESE_CENSORSHIP_DATASET_ID
        if (
            is_censorship_dataset
            and str(self.file_path) != CHINESE_CENSORSHIP_DATASET_SOURCE
        ):
            raise ValueError(
                "chinese_censorship must use the pinned dataset source "
                f"{CHINESE_CENSORSHIP_DATASET_SOURCE!r}."
            )
        try:
            if is_censorship_dataset:
                loaded_dataset = load_dataset(
                    CHINESE_CENSORSHIP_DATASET_SOURCE,
                    name=CHINESE_CENSORSHIP_DATASET_CONFIG,
                    revision=CHINESE_CENSORSHIP_DATASET_REVISION,
                    token=token,
                    trust_remote_code=False,
                )
            else:
                loaded_dataset = load_dataset(
                    str(self.file_path),
                    token=token,
                    trust_remote_code=trust_remote_code,
                )
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to load dataset '{self.file_path}': {exc}. "
                "Check that the identifier is correct and, for private or gated "
                "datasets, that a Hugging Face token with access "
                "to the repo is provided via --model-token."
            ) from exc
        if not isinstance(loaded_dataset, DatasetDict):
            raise ValueError(f"Expected DatasetDict, got {type(loaded_dataset)}")
        if is_censorship_dataset:
            if CHINESE_CENSORSHIP_DATASET_SPLIT not in loaded_dataset:
                raise ValueError(
                    "Chinese censorship dataset must contain the pinned 'test' split"
                )
            censorship_dataset = loaded_dataset[CHINESE_CENSORSHIP_DATASET_SPLIT]
            required_columns = {"question", "source_group_id"}
            missing_columns = required_columns - set(censorship_dataset.column_names)
            if missing_columns:
                raise ValueError(
                    "Chinese censorship dataset is missing required columns: "
                    f"{sorted(missing_columns)}."
                )
            self.censorship_group_ids = [
                str(value) for value in censorship_dataset["source_group_id"]
            ]
            self.censorship_questions = [
                str(value) for value in censorship_dataset["question"]
            ]
            without_ids = cast(
                "Dataset", censorship_dataset.remove_columns("source_group_id")
            )
            self.dataset = cast(
                "Dataset",
                without_ids.add_column(
                    "censorship_group_index", list(range(len(censorship_dataset)))
                ),
            )
        elif "train" in loaded_dataset:
            self.dataset = loaded_dataset["train"]
        elif len(loaded_dataset) == 1:
            self.dataset = next(iter(loaded_dataset.values()))
        else:
            raise ValueError(
                f"Expected dataset '{self.file_path}' to contain a 'train' split or exactly one split, found {list(loaded_dataset.keys())}"
            )
        self.has_stereotype: bool = "stereotyped_answer" in self.dataset.column_names

    def preprocess(
        self,
        tokenizer: PreTrainedTokenizerBase,
        preprocess_config: PreprocessConfig,
        max_answer_tokens: int | None = None,
        enable_thinking: bool = False,
        enable_thinking_arg_name: str | None = None,
        thinking_start_token: str | None = None,
        thinking_end_token: str | None = None,
        pass_max_answer_tokens: bool = False,
        model_revision: str | None = None,
    ) -> Dataset:
        """
        Tokenize the loaded dataset for free-text evaluation.

        Args:
            tokenizer: Tokenizer used for text processing.
            preprocess_config: Configuration for preprocessing the dataset.
            max_answer_tokens: Maximum number of tokens to allow for the answer.
            enable_thinking: Whether to enable thinking.
            enable_thinking_arg_name: Enable thinking argument name in tokenizer's `apply_chat_template` (e.g. 'enable_thinking').
            thinking_start_token: Thinking start token to use for the model (e.g. '<think>').
            thinking_end_token: Thinking end token to use for the model (e.g. '</think>').
            pass_max_answer_tokens: Whether to pass max_answer_tokens to the chat template.
            model_revision: Optional immutable Hugging Face model revision used for
                tokenizer and multimodality lookups. Pinned evaluations such as
                ``chinese_censorship`` requires a 40-64 character hexadecimal
                revision.

        Returns:
            A tokenized dataset ready for evaluation.
        """
        refusal_dataset = self.dataset_id in REFUSAL_DATASETS
        censorship_dataset = self.dataset_id == CHINESE_CENSORSHIP_DATASET_ID
        dataset = (
            normalize_refusal_dataset(self.dataset) if refusal_dataset else self.dataset
        )
        if censorship_dataset:
            old_columns = dataset.column_names
            is_multimodal = is_model_multimodal(
                tokenizer.name_or_path,
                self.trust_remote_code,
                self.token,
                revision=model_revision,
            )
            processed_dataset = dataset.map(
                lambda examples: censorship_preprocess_function(
                    examples,
                    tokenizer,
                    max_length=preprocess_config.max_length,
                    is_multimodal=is_multimodal,
                    max_answer_tokens=max_answer_tokens,
                    enable_thinking=enable_thinking,
                    enable_thinking_arg_name=enable_thinking_arg_name,
                    thinking_start_token=thinking_start_token,
                    thinking_end_token=thinking_end_token,
                    pass_max_answer_tokens=pass_max_answer_tokens,
                ),
                batched=True,
                remove_columns=old_columns,
                batch_size=preprocess_config.preprocess_batch_size,
                num_proc=1,
            )
            return cast("Dataset", processed_dataset)
        validate_dataset_columns(dataset)
        old_columns = dataset.column_names
        # Compute once to avoid per-batch remote config lookups
        is_multimodal = is_model_multimodal(
            tokenizer.name_or_path,
            self.trust_remote_code,
            self.token,
            revision=model_revision,
        )
        processed_dataset = dataset.map(
            lambda examples: free_text_preprocess_function(
                examples,
                tokenizer,
                max_length=preprocess_config.max_length,
                gt_max_length=preprocess_config.gt_max_length,
                has_stereotype=self.has_stereotype,
                is_multimodal=is_multimodal,
                max_answer_tokens=max_answer_tokens,
                enable_thinking=enable_thinking,
                enable_thinking_arg_name=enable_thinking_arg_name,
                thinking_start_token=thinking_start_token,
                thinking_end_token=thinking_end_token,
                pass_max_answer_tokens=pass_max_answer_tokens,
                include_default_system_prompt=not refusal_dataset,
                # ⬆️ The default system prompt is detrimental to refusal evaluation, and therefore is avoided for refusal datasets.
            ),
            batched=True,
            remove_columns=old_columns,
            batch_size=preprocess_config.preprocess_batch_size,
            num_proc=1,
        )
        text = tokenizer.batch_decode(
            cast("list[list[int]]", list(processed_dataset["test_input_ids"])),
            skip_special_tokens=True,
        )
        if text:
            logging.info("Validation text: %s", text[0])
        return cast("Dataset", processed_dataset)
