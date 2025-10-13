import logging
from typing import Any

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.quantization_config import BitsAndBytesConfig


def safe_apply_chat_template(
    tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]]
) -> str:
    """
    Applies the chat template to the messages, ensuring that the system message is handled correctly.
    This is particularly important for models like Gemma v1, where the system message needs to be merged with the user message.
    Old Gemma models are deliberately strict about the roles they accept in a chat prompt.
    The official Jinja chat‑template that ships with the tokenizer throws an exception as soon as the first message is tagged "system".
    This function checks if the tokenizer is an old Gemma model and handles the system message accordingly.

    Args:
        tokenizer: The tokenizer to use for applying the chat template.
        messages: The list of messages to format.

    Returns:
        The formatted string after applying the chat template.
    """
    is_gemma_v1 = (
        tokenizer.name_or_path.startswith("google/gemma-")
        and "System role not supported" in tokenizer.chat_template
    )

    if is_gemma_v1 and messages and messages[0]["role"] == "system":
        # merge system into next user turn or retag
        # Gemma v1 models do not support system messages in their chat templates.
        # To handle this, we merge the system message into the next user message or retag it as a user message.
        sys_msg = messages.pop(0)["content"]
        if messages and messages[0]["role"] == "user":
            messages[0]["content"] = f"{sys_msg}\n\n{messages[0]['content']}"
        else:
            messages.insert(0, {"role": "user", "content": sys_msg})

    # Build chat_messages with correct structure for multimodal tokenizers
    chat_messages: list[dict[str, Any]] = []
    for message in messages:
        current_content = message["content"]
        chat_messages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": str(current_content)}],
            }
        )
    print(chat_messages)
    return str(
        tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
    )


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """
    Load a tokenizer by first trying the standard method and, if a ValueError
    is encountered, retry loading from a local path.
    """
    try:
        # Attempt to load the tokenizer normally
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("Tokenizer loaded successfully from the remote repository.")
    except ValueError as error:
        # Print or log the error details if desired
        logging.info(
            "Standard loading failed: %s. Falling back to local loading using 'local_files_only=True'.",
            error,
        )
        # Retry loading with local_files_only flag
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        logging.info("Tokenizer loaded successfully from the local files.")

    return tokenizer


def pick_best_dtype(device: str, prefer_bf16: bool = True) -> torch.dtype:
    """
    Robust dtype checker that adapts to the hardware:
      • chooses bf16→fp16→fp32 automatically
    """
    if device == "cuda" and prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        # fp16 is universally supported on CUDA GPUs ≥ sm_50
        return torch.float16
    # CPU or MPS → stay in full precision for safety
    return torch.float32


def use_vision_architecture(repo_id: str) -> bool:
    """
    Decide whether the model should be loaded with a vision-capable architecture.

    This checks the model configuration for multimodal/vision hints and explicitly
    enables the vision architecture when the config's model_type is "qwen2_5_vl".
    """
    cfg = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
    cfg_dict = cfg.to_dict()

    # Explicit rule: Qwen2.5-VL family
    if cfg_dict.get("model_type") == "qwen2_5_vl":
        return True

    return False


def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = False,
    device_map: str | dict[str, int] | None = "auto",
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Load a tokenizer and a causal language model based on the model name/path,
    using the model's configuration to determine the correct class to instantiate.

    Optionally load the model in 4-bit precision (using bitsandbytes) instead
    of the default 16-bit precision.

    Args:
        model_name: The repo-id or local path of the model to load.
        use_4bit: If True, load the model in 4-bit mode using bitsandbytes.
        device_map: The device map to use for the model.

    Returns:
        A tuple containing the loaded tokenizer and model.

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_best_dtype(device)
    logging.info("Using dtype: %s", dtype)

    # Load tokenizer
    tokenizer = load_tokenizer(model_name)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not supported!")

    # Optionally adjust the tokenizer settings (e.g., for padding)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if use_4bit:
        # Prepare the quantization configuration for 4-bit loading.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if use_vision_architecture(model_name):
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )

    return tokenizer, model
