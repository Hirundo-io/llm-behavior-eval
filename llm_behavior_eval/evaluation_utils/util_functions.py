import logging
import sys
from pathlib import Path

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.quantization_config import BitsAndBytesConfig

sys.path.append(f"{Path(__file__).parents[4].as_posix()}/hirundo_core/h_core")
sys.path.append(f"{Path(__file__).parents[4].as_posix()}/hirundo_core")
sys.path.append(Path(__file__).parents[4].as_posix())

from hirundo_core.h_core.debias.methods.plugin_model import PluginModelForCausalLM


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
    return str(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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


def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = False,
    plugin_model_path: str | None = None,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Load a tokenizer and a causal language model based on the model name/path,
    using the model's configuration to determine the correct class to instantiate.

    Optionally load the model in 4-bit precision (using bitsandbytes) instead
    of the default 16-bit precision.

    If plugin_model_path is provided, loads a PluginModelForCausalLM instead of a regular model.
    The base model path will be read from the plugin model's saved config.

    Args:
        model_name: The repo-id or local path of the model to load. Only used when plugin_model_path is None.
        use_4bit: If True, load the model in 4-bit mode using bitsandbytes.
        plugin_model_path: Optional path to plugin model. If provided, loads PluginModelForCausalLM
                          and the base model path is read from the plugin's config.

    Returns:
        A tuple containing the loaded tokenizer and model.

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_best_dtype(device)
    logging.info("Using dtype: %s", dtype)

    # If plugin model is requested, load PluginModelForCausalLM
    if plugin_model_path is not None:
        logging.info("Loading plugin model from %s", plugin_model_path)

        # Load config to get base model path
        cfg = AutoConfig.from_pretrained(plugin_model_path)

        # Get base model path from config
        base_name = cfg.base_model_name_or_path
        if not base_name:
            raise ValueError(
                "base_model_name_or_path must be in the plugin model's saved config. "
                "Make sure the plugin model was saved with the base model path."
            )

        # Load PluginModelForCausalLM using from_pretrained
        # Note: This loads both plugin and base models simultaneously
        # Using low_cpu_mem_usage to reduce peak memory during loading
        # device_map="auto" will distribute models across available GPUs/CPU
        model = PluginModelForCausalLM.from_pretrained(
            plugin_model_path,
            base_model_name_or_path=base_name,
            device_map="auto",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        
        # Log memory usage after loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logging.info(
                    f"GPU {i} memory after plugin model load: "
                    f"{allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
                )

        # Load tokenizer from base model path (from config)
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError("Tokenizer is not supported!")

        # Set pad_token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer, model

    # Regular model loading
    # Load tokenizer
    tokenizer = load_tokenizer(model_name)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not supported!")

    # Load the configuration to inspect the model type
    config = AutoConfig.from_pretrained(model_name)
    model_type = config.model_type.lower() if config.model_type else ""

    # Optionally adjust the tokenizer settings (e.g., for padding)
    if model_type == "llama":
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        # Prepare the quantization configuration for 4-bit loading.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=quantization_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    return tokenizer, model
