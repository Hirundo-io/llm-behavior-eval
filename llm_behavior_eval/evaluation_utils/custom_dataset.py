import logging
from collections.abc import Mapping, Sequence
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
from .refusal_utils import REFUSAL_DATASETS, normalize_refusal_dataset
from .util_functions import is_model_multimodal, safe_apply_chat_template

_OPTIONAL_METADATA_TYPES: dict[str, tuple[type[object], ...]] = {
    "label": (int, str),
    "technique": (str,),
    "protected_value": (str,),
}


def _validate_optional_metadata(
    columns: Mapping[str, Sequence[object]], source: str
) -> None:
    for key, types in _OPTIONAL_METADATA_TYPES.items():
        if key not in columns:
            continue
        for value in columns[key]:
            if type(value) not in types:
                allowed = " or ".join(type_.__name__ for type_ in types)
                raise ValueError(
                    f"{source} field '{key}' must contain {allowed} values; "
                    f"got {value!r} ({type(value).__name__})"
                )

    label_values = columns.get("label")
    if label_values and len({type(value) for value in label_values}) > 1:
        raise ValueError(f"{source} field 'label' must use one consistent value type")


def get_dataset_slug(dataset_id: Path | str) -> str:
    """Return the normalized basename used for dataset-specific routing.

    Args:
        dataset_id: Dataset ID, URI, or filesystem path as a string or ``Path``.

    Returns:
        The lowercase final path segment after trailing slashes are removed.
    """
    return str(dataset_id).rstrip("/").rsplit("/", 1)[-1].lower()


def validate_dataset_columns(hf_dataset: Dataset) -> None:
    """Validate required columns and optional free-text metadata.

    Args:
        hf_dataset: Dataset whose columns and prompt-injection metadata are validated.

    Returns:
        None.

    Raises:
        ValueError: If required columns are absent, optional metadata has an invalid
            type, or the label column mixes integer and string representations.
    """
    # Minimum required columns for free-text
    required = {"question", "answer"}

    missing = required - set(hf_dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}; found {hf_dataset.column_names}"
        )

    metadata = {
        key: list(hf_dataset[key])
        for key in _OPTIONAL_METADATA_TYPES
        if key in hf_dataset.column_names
    }
    _validate_optional_metadata(metadata, "Dataset")


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
) -> dict[str, torch.Tensor | list[str]]:
    """
    Preprocesses a batch of examples for free-text datasets.

    Args:
        examples_batch: A batch of examples to preprocess. Optional prompt-injection
            columns are ``label``, ``technique``, and ``protected_value``. Integer
            labels are refusal targets; string labels are prompt-injection metadata.
            All columns must have equal lengths; mismatches raise ``ValueError``.
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
        A dictionary containing tokenized inputs, ground truths, and judge questions.
        Integer labels produce ``refusal_labels``. Non-integer labels, techniques,
        and protected values produce ``labels``, ``techniques``, and the raw-string
        ``protected_values`` respectively when their source columns are present.
    """
    # 1) Column check
    rows = [
        dict(zip(examples_batch.keys(), row_values, strict=True))
        for row_values in zip(*examples_batch.values(), strict=True)
    ]
    _validate_optional_metadata(examples_batch, "Free text")
    # Validate minimally required fields only
    for row in rows:
        if not row.get("question") or not row.get("answer"):
            raise ValueError("Free text row must contain 'question' and 'answer'")
    # 2) Apply chat template to dataset messages
    eval_strings, answer_strings = [], []
    stereotyped_strings: list[str] = []
    judge_questions: list[str] = []
    labels: list[str] = []
    techniques: list[str] = []
    protected_values: list[str] = []
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
        labels.append(str(row["label"]) if "label" in row else "")
        techniques.append(str(row["technique"]) if "technique" in row else "")
        protected_values.append(
            str(row["protected_value"]) if "protected_value" in row else ""
        )
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
    label_values = examples_batch.get("label")
    tokenized_stereotype = None
    if has_stereotype:
        tokenized_stereotype = tokenize(
            stereotyped_strings,
            max_length=gt_max_length,
            add_special_tokens=False,
        )
    # 4) Result
    result: dict[str, torch.Tensor | list[str]] = {
        "test_input_ids": torch.tensor(tokenized_eval["input_ids"]),
        "test_attention_mask": torch.tensor(tokenized_eval["attention_mask"]),
        "gt_answers": torch.tensor(tokenized_gt["input_ids"]),
        "judge_questions": torch.tensor(tokenized_judge_questions["input_ids"]),
    }
    if has_stereotype and tokenized_stereotype is not None:
        result["stereotyped_answers"] = torch.tensor(tokenized_stereotype["input_ids"])
    if label_values is not None and all(type(label) is str for label in label_values):
        result["labels"] = labels
    if "technique" in examples_batch:
        result["techniques"] = techniques
    if "protected_value" in examples_batch:
        result["protected_values"] = protected_values
    if "label" in examples_batch and all(
        type(label) is int for label in examples_batch["label"]
    ):
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
    ):
        """
        Initializes the custom dataset with a specified dataset type and behavior type.

        Args:
            file_path: Local path or HuggingFace dataset identifier.
            dataset_type: The type of the dataset (e.g., BIAS or UNBIAS).
            trust_remote_code: Whether to trust remote code when loading the dataset.
            token: HuggingFace token for gated or private dataset repos.
        """
        self.file_path = file_path
        self.dataset_type = dataset_type
        self.trust_remote_code = trust_remote_code
        self.token = token
        try:
            raw = load_dataset(
                str(self.file_path),
                token=token,
                trust_remote_code=trust_remote_code,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to load dataset '{self.file_path}'. "
                "Check that the identifier is correct."
            ) from exc
        if not isinstance(raw, DatasetDict):
            raise ValueError(f"Expected DatasetDict, got {type(raw)}")
        if "train" in raw:
            self.ds = raw["train"]
        elif len(raw) == 1:
            self.ds = next(iter(raw.values()))
        else:
            raise ValueError(
                f"Expected dataset '{self.file_path}' to contain a 'train' split or exactly one split, found {list(raw.keys())}"
            )
        self.has_stereotype: bool = "stereotyped_answer" in self.ds.column_names

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

        Returns:
            A tokenized dataset ready for evaluation.
        """
        refusal_dataset = str(self.file_path) in REFUSAL_DATASETS
        dataset = normalize_refusal_dataset(self.ds) if refusal_dataset else self.ds
        validate_dataset_columns(dataset)
        old_columns = dataset.column_names
        # Compute once to avoid per-batch remote config lookups
        is_multimodal = is_model_multimodal(
            tokenizer.name_or_path, self.trust_remote_code, self.token
        )
        load_from_cache_file = not get_dataset_slug(self.file_path).startswith(
            "bloom-prompt-injection-"
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
            load_from_cache_file=load_from_cache_file,
        )
        # Dataset column typing is broader than the tokenizer-produced nested token IDs.
        text = tokenizer.batch_decode(
            cast("list[list[int]]", list(processed_dataset["test_input_ids"])),
            skip_special_tokens=True,
        )
        if text:
            logging.info("Validation text: %s", text[0])
        return cast("Dataset", processed_dataset)
