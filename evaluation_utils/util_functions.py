import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.gemma import GemmaForCausalLM
from transformers.models.gemma2 import Gemma2ForCausalLM
from transformers.models.gemma3 import Gemma3ForConditionalGeneration
from transformers.models.granite.modeling_granite import GraniteForCausalLM
from transformers.models.llama import LlamaForCausalLM
from transformers.models.llama4 import Llama4ForConditionalGeneration
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
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
        sys_msg = messages.pop(0)["content"]
        if messages and messages[0]["role"] == "user":
            messages[0]["content"] = f"{sys_msg}\n\n{messages[0]['content']}"
        else:
            messages.insert(0, {"role": "user", "content": sys_msg})
    return str(tokenizer.apply_chat_template(messages, tokenize=False))


SUPPORTED_MODELS = ["llama", "granite", "gemma", "qwen2", "gemma3"]
SUPPORTED_ARCHITECTURES = {
    "Llama4ForConditionalGeneration": Llama4ForConditionalGeneration,
    "LlamaForCausalLM": LlamaForCausalLM,
    "GraniteForCausalLM": GraniteForCausalLM,
    "GemmaForCausalLM": GemmaForCausalLM,
    "Gemma3ForConditionalGeneration": Gemma3ForConditionalGeneration,
    "Gemma2ForCausalLM": Gemma2ForCausalLM,
    "Qwen2ForCausalLM": Qwen2ForCausalLM,
}


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """
    Load a tokenizer by first trying the standard method and, if a ValueError
    is encountered, retry loading from a local path.
    """
    try:
        # Attempt to load the tokenizer normally
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully from the remote repository.")
    except ValueError as e:
        # Print or log the error details if desired
        print(
            f"Standard loading failed: {e}. Falling back to local loading using 'local_files_only=True'."
        )
        # Retry loading with local_files_only flag
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print("Tokenizer loaded successfully from the local files.")

    return tokenizer


def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = False,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Load a tokenizer and a causal language model based on the model name/path,
    using the model's configuration to determine the correct class to instantiate.

    Optionally load the model in 4-bit precision (using bitsandbytes) instead
    of the default 16-bit precision.

    Args:
        model_name: The repo-id or local path of the model to load.
        use_4bit: If True, load the model in 4-bit mode using bitsandbytes.

    Returns:
        A tuple containing the loaded tokenizer and model.

    Raises:
        ValueError: If the tokenizer is unsupported or the model type is not supported.
    """
    # Load tokenizer
    tokenizer = load_tokenizer(model_name)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not supported!")

    # Load the configuration to inspect the model type
    config = AutoConfig.from_pretrained(model_name)
    model_type = config.model_type.lower() if config.model_type else ""
    architecture_name = config.architectures[0] if config.architectures else ""

    # Optionally adjust the tokenizer settings (e.g., for padding)
    if model_type == "llama":
        tokenizer.pad_token = tokenizer.eos_token

    # Determine the appropriate model class based on model type.
    if model_type in SUPPORTED_MODELS and architecture_name in SUPPORTED_ARCHITECTURES:
        architecture_model = SUPPORTED_ARCHITECTURES[architecture_name]
        if use_4bit:
            # Prepare the quantization configuration for 4-bit loading.
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = architecture_model.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
            )
        else:
            model = architecture_model.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
    else:
        raise ValueError(
            f"Model type '{model_type}' with architecture class {architecture_name} is not yet supported."
        )

    return tokenizer, model
