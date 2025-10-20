import logging
from inspect import Parameter, signature
from typing import Any

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.quantization_config import BitsAndBytesConfig


def safe_apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    is_multimodal: bool = False,
    reasoning: bool = False,
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
        is_multimodal: Whether to format messages for a multimodal model.
        reasoning: Whether to enable tokenizer chat-template reasoning (if supported).

    Returns:
        The formatted string after applying the chat template.
    """
    # Cache for checking whether a given tokenizer's apply_chat_template supports
    # the `reasoning` kwarg. Keyed by id(tokenizer) to avoid holding strong refs.
    # This avoids repeated try/except in hot paths.
    _CHAT_TEMPLATE_SUPPORTS_REASONING: dict[int, bool] = getattr(
        safe_apply_chat_template, "_CHAT_TEMPLATE_SUPPORTS_REASONING", {}
    )
    setattr(
        safe_apply_chat_template,
        "_CHAT_TEMPLATE_SUPPORTS_REASONING",
        _CHAT_TEMPLATE_SUPPORTS_REASONING,
    )

    def _supports_reasoning_kwarg(tokenizer: PreTrainedTokenizerBase) -> bool:
        cache_key = id(tokenizer)
        cached = _CHAT_TEMPLATE_SUPPORTS_REASONING.get(cache_key)
        if cached is not None:
            return cached

        apply_fn = getattr(tokenizer, "apply_chat_template", None)
        supports = False
        if apply_fn is not None:
            try:
                sig = signature(apply_fn)
                has_kwargs = any(
                    p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
                supports = has_kwargs or ("reasoning" in sig.parameters)
            except (TypeError, ValueError):
                # Some callables may not expose a Python signature; fall back to False
                supports = False

        _CHAT_TEMPLATE_SUPPORTS_REASONING[cache_key] = supports
        return supports

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

    # Choose formatting based on whether the model is multimodal
    def _apply_chat_template(
        messages_like: list[dict[str, Any]] | list[dict[str, str]],
    ):
        """
        Call tokenizer.apply_chat_template while remaining compatible with
        tokenizers that don't support the `reasoning` kwarg.
        """
        if _supports_reasoning_kwarg(tokenizer):
            return tokenizer.apply_chat_template(
                messages_like,
                tokenize=False,
                add_generation_prompt=True,
                reasoning=reasoning,
            )
        return tokenizer.apply_chat_template(
            messages_like,
            tokenize=False,
            add_generation_prompt=True,
        )

    if is_multimodal:
        # Multimodal: list-of-parts with type "text"
        multimodal_chat_messages: list[dict[str, Any]] = []
        for message in messages:
            current_content = message["content"]
            multimodal_chat_messages.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": str(current_content)}],
                }
            )
        return str(_apply_chat_template(multimodal_chat_messages))

    # Unimodal: plain string content
    chat_messages_text: list[dict[str, str]] = []
    for message in messages:
        chat_messages_text.append(
            {"role": message["role"], "content": str(message["content"])}
        )
    return str(_apply_chat_template(chat_messages_text))


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


def is_model_multimodal(repo_id: str, trust_remote_code: bool = False) -> bool:
    """
    Decide whether the model should be loaded with a vision-capable architecture.

    This checks the model configuration for multimodal/vision hints and explicitly
    enables the vision architecture when the config's model_type is in the SUPPORTED_MULTIMODAL_MODELS list.

    Args:
        repo_id: The repo-id or local path of the model to load.
        trust_remote_code: Whether to trust remote code.

    Returns:
        True if the model should be loaded with a vision-capable architecture, False otherwise.
    """
    try:
        # Prefer local cache to avoid network calls during preprocessing
        config = AutoConfig.from_pretrained(
            repo_id, local_files_only=True, trust_remote_code=trust_remote_code
        )
    except Exception:
        # Fallback to remote if not cached locally
        config = AutoConfig.from_pretrained(
            repo_id, trust_remote_code=trust_remote_code
        )
    config_dict = config.to_dict()

    if config_dict.get("model_type") in list(
        MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()
    ):
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
    model_by_nvidia = model_name.startswith("nvidia/")
    trust_remote_code = True if model_by_nvidia else False

    if use_4bit:
        # Prepare the quantization configuration for 4-bit loading.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if is_model_multimodal(model_name, trust_remote_code):
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
        )

    return tokenizer, model
