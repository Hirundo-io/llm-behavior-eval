from __future__ import annotations

import hashlib
import logging
import os
import shutil
from contextlib import contextmanager
from inspect import Parameter, signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import torch
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.quantization_config import BitsAndBytesConfig

from llm_behavior_eval.evaluation_utils.vllm_config import VllmConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .vllm_types import TokenizerModeOption

VLLMDType = Literal["bfloat16", "float16", "float32"]
VLLMQuantization = Literal[
    "awq",
    "deepspeedfp",
    "tpu_int8",
    "fp8",
    "ptpc_fp8",
    "bitsandbytes",
]

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from vllm import LLM

DIGEST_SIZE_FOR_PEFT_PATHS = 8


def empty_cuda_cache_if_available() -> None:
    """Free CUDA cache if GPUs are available."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def _hf_token(token: str | None) -> Iterator[None]:
    """
    If `token` is truthy, temporarily set HF_TOKEN for Hugging Face API calls;
    otherwise, do nothing.
    """
    if not token:
        yield
        return

    prev: str | None = os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = token
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = prev


class SafeApplyChatTemplate:
    _CHAT_TEMPLATE_SUPPORTS_REASONING: dict[int, bool] = {}

    def __call__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        messages: list[dict[str, str]],
        is_multimodal: bool = False,
        max_answer_tokens: int | None = None,
        reasoning: bool = False,
        pass_max_answer_tokens: bool = False,
    ) -> str:
        """
        Applies the chat template to the messages, ensuring that the system message is handled correctly.
        This is particularly important for models like Gemma v1, where the system message needs to be merged with the user message.
        Old Gemma models are deliberately strict about the roles they accept in a chat prompt.
        The official Jinja chat-template that ships with the tokenizer throws an exception as soon as the first message is tagged "system".
        This function checks if the tokenizer is an old Gemma model and handles the system message accordingly.

        Args:
            tokenizer: The tokenizer to use for applying the chat template.
            messages: The list of messages to format.
            is_multimodal: Whether to format messages for a multimodal model.
            max_answer_tokens: The maximum number of tokens to allow for the answer.
            reasoning: Whether to enable tokenizer chat-template reasoning (if supported).
            pass_max_answer_tokens: Whether to pass the max_answer_tokens to the chat template.

        Returns:
            The formatted string after applying the chat template.
        """

        # Cache for checking whether a given tokenizer's apply_chat_template supports
        # the `reasoning` kwarg. Keyed by id(tokenizer) to avoid holding strong refs.
        # This avoids repeated try/except in hot paths.
        def _supports_reasoning_kwarg(tokenizer: PreTrainedTokenizerBase) -> bool:
            cache_key = id(tokenizer)
            cached = self._CHAT_TEMPLATE_SUPPORTS_REASONING.get(cache_key)
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

            self._CHAT_TEMPLATE_SUPPORTS_REASONING[cache_key] = supports
            return supports

        is_gemma_v1 = (
            tokenizer.name_or_path.startswith("google/gemma-")
            and isinstance(tokenizer.chat_template, str)
            and "System role not supported" in tokenizer.chat_template
        )
        uses_nothink = (
            isinstance(tokenizer.chat_template, str)
            and "/no_think" in tokenizer.chat_template
        )

        if messages and messages[0]["role"] == "system":
            if pass_max_answer_tokens and max_answer_tokens is not None:
                messages[0]["content"] = (
                    f"{messages[0]['content']} Respond in no more than {max_answer_tokens} tokens."
                )
            if is_gemma_v1:
                # merge system into next user turn or retag
                # Gemma v1 models do not support system messages in their chat templates.
                # To handle this, we merge the system message into the next user message or retag it as a user message.
                sys_msg = messages.pop(0)["content"]
                if messages and messages[0]["role"] == "user":
                    messages[0]["content"] = f"{sys_msg}\n\n{messages[0]['content']}"
                else:
                    messages.insert(0, {"role": "user", "content": sys_msg})
            elif uses_nothink and not reasoning:
                messages[0]["content"] = f"/no_think {messages[0]['content']}"

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


safe_apply_chat_template = SafeApplyChatTemplate()


def load_tokenizer_with_transformers(
    model_name: str,
    token: str | None = None,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizerBase:
    """
    Load a tokenizer by first trying the standard method and, if a ValueError
    is encountered, retry loading from a local path.

    Args:
        model_name: The repo-id or local path of the model to load.
        token: The HuggingFace token to use for the model.
        trust_remote_code: Whether to trust remote code.
    """
    try:
        # Attempt to load the tokenizer normally
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        logging.info("Tokenizer loaded successfully from the remote repository.")
    except ValueError as error:
        # Print or log the error details if desired
        logging.info(
            "Standard loading failed: %s. Falling back to local loading using 'local_files_only=True'.",
            error,
        )
        # Retry loading with local_files_only flag
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
            token=token,
            trust_remote_code=trust_remote_code,
        )
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


def is_model_multimodal(
    repo_id: str, trust_remote_code: bool = False, token: str | None = None
) -> bool:
    """
    Decide whether the model should be loaded with a vision-capable architecture.

    This checks the model configuration for multimodal/vision hints and explicitly
    enables the vision architecture when the config's model_type is in the SUPPORTED_MULTIMODAL_MODELS list.

    Args:
        repo_id: The repo-id or local path of the model to load.
        trust_remote_code: Whether to trust remote code.
        token: The HuggingFace token to use for accessing gated models.

    Returns:
        True if the model should be loaded with a vision-capable architecture, False otherwise.
    """
    try:
        # Prefer local cache to avoid network calls during preprocessing
        config = AutoConfig.from_pretrained(
            repo_id,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            token=token,
        )
    except Exception:
        # Fallback to remote if not cached locally
        config = AutoConfig.from_pretrained(
            repo_id, trust_remote_code=trust_remote_code, token=token
        )
    config_dict = config.to_dict()

    if config_dict.get("model_type") in list(
        MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()
    ):
        return True

    return False


def torch_dtype_to_str(dtype: torch.dtype) -> VLLMDType:
    """Translate a ``torch.dtype`` into the string literal expected by vLLM."""

    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unsupported dtype for vLLM: {dtype}")


def _get_default_from_vllm(parameter_name: str):
    from vllm import LLM

    return signature(LLM.__init__).parameters[parameter_name].default


def load_vllm_model(
    model_name: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    batch_size: int,
    token: str | None = None,
    tensor_parallel_size: int | None = None,
    enforce_eager: bool = False,
    quantization: VLLMQuantization | None = None,
    max_model_len: int | None = None,
    tokenizer_mode: TokenizerModeOption | None = None,
    config_format: str | None = None,
    load_format: str | None = None,
    gpu_memory_utilization: float = 0.9,
    enable_lora: bool = False,
    max_lora_rank: int = VllmConfig.model_fields["max_lora_rank"].default,
) -> LLM:
    """Load a vLLM model engine.

    Args:
        model_name: Model identifier or local path.
        dtype: Torch dtype to request from vLLM.
        trust_remote_code: Whether to allow remote code execution when loading the model.
        batch_size: The batch size to use for the model.
        token: The HuggingFace token to use for the model.
        tensor_parallel_size: Optional tensor parallelism degree passed to vLLM.
        enforce_eager: Whether to enforce eager execution (useful for CPU-only setups).
        quantization: Optional quantization backend (for example ``"bitsandbytes"`` for 4-bit inference).
        max_model_len: Optional maximum model length passed to vLLM.
        tokenizer_mode: Optional tokenizer mode string forwarded to vLLM.
        config_format: Optional config format string forwarded to vLLM.
        load_format: Optional checkpoint load format string forwarded to vLLM.
        gpu_memory_utilization: Optional GPU memory utilization passed to vLLM.
        enable_lora: Whether to enable LoRA.
        max_lora_rank: The maximum LoRA rank (do not set too high to avoid wasting memory).
    Returns:
        An initialized ``vllm.LLM`` instance.
    """

    try:
        from vllm import LLM
        from vllm.config import CompilationConfig
    except ImportError as exc:
        raise ImportError(
            "vLLM is not installed. Install it with `uv pip install llm-behavior-eval[vllm]` to enable vllm for inference (e.g. when using the --inference-engine argument)."
        ) from exc

    dtype_literal = torch_dtype_to_str(dtype)

    tensor_parallel = tensor_parallel_size
    if tensor_parallel is None:
        gpu_count = torch.cuda.device_count()
        tensor_parallel = gpu_count if gpu_count > 0 else None

    default_tokenizer_mode = _get_default_from_vllm("tokenizer_mode")
    default_tensor_parallel = _get_default_from_vllm("tensor_parallel_size")

    with _hf_token(token):
        llm_instance = LLM(
            model=model_name,
            trust_remote_code=trust_remote_code,
            dtype=dtype_literal,
            enforce_eager=enforce_eager,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel
            if tensor_parallel is not None
            else default_tensor_parallel,
            max_num_seqs=batch_size,
            hf_token=token,
            max_model_len=max_model_len,
            tokenizer_mode=tokenizer_mode or default_tokenizer_mode,
            config_format=config_format,
            load_format=load_format,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=enable_lora,
            max_lora_rank=max_lora_rank,
            compilation_config=CompilationConfig(cudagraph_specialize_lora=False),
        )
    return llm_instance


def build_vllm_prompt_token_ids(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> list[list[int]]:
    """Convert a padded batch of token ids into prompt-only sequences for vLLM."""

    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must share the same shape")

    prompt_token_ids: list[list[int]] = []
    for tokens, mask in zip(input_ids, attention_mask, strict=True):
        token_tensor = tokens.detach().cpu()
        mask_tensor = mask.detach().cpu().to(dtype=torch.bool)
        valid_tokens = token_tensor[mask_tensor]
        prompt_token_ids.append([int(token) for token in valid_tokens.tolist()])

    return prompt_token_ids


def load_transformers_model_and_tokenizer(
    model_name: str,
    token: str | None = None,
    use_4bit: bool = False,
    device_map: str | dict[str, int] | None = "auto",
    trust_remote_code: bool = False,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Load a tokenizer and a causal language model based on the model name/path,
    using the model's configuration to determine the correct class to instantiate.

    Optionally load the model in 4-bit precision (using bitsandbytes) instead
    of the default 16-bit precision.

    Args:
        model_name: The repo-id or local path of the model to load.
        token: The HuggingFace token to use for the model.
        use_4bit: If True, load the model in 4-bit mode using bitsandbytes.
        device_map: The device map to use for the model.
        trust_remote_code: Whether to trust remote code.

    Returns:
        A tuple containing the loaded tokenizer and model.

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_best_dtype(device)
    logging.info("Using dtype: %s", dtype)

    # Load tokenizer
    tokenizer = load_tokenizer_with_transformers(
        model_name, token=token, trust_remote_code=trust_remote_code
    )
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

    if is_model_multimodal(model_name, trust_remote_code, token):
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
            token=token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
            token=token,
        )

    return tokenizer, model


def maybe_download_adapter(
    adapter_ref: str,
    cache_dir: str | None = None,
    mlflow_tracking_uri: str | None = None,
) -> str:
    """
    If `adapter_ref` is one of:
      - mlflow://...
      - http://... / https://... where scheme+netloc match the MLflow tracking URI
      - git://...
      - s3://...
      - gs://...
      - Other schemes which fsspec supports

    ...download/clone it into a deterministic local cache directory and return that local path.

    Otherwise (local path, HF repo id, etc.), return `adapter_ref` unchanged.

    Behavior:
      - Deterministic cache location: <cache_dir or ~/.cache>/peft_adapters/<scheme>_<hash>/
      - Overwrite: re-download whether or not the destination exists. It's the user's responsibility to manage the cache.

    MLflow special-case:
      - If adapter_ref is exactly mlflow://<run_id> (no artifact subpath),
        or http://... / https://... where scheme+netloc match the MLflow tracking URI
        Under the run id, the adapter should reside in a directory by the following name:
          1) hf_checkpoint/peft   (try first)
          2) hf_checkpoint        (fallback)

    Optional deps (lazy imports):
      - mlflow               (for mlflow:// or http://... / https://... matching MLflow tracking URI)
      - gitpython            (for git://)
      - fsspec + s3fs/gcsfs  (for s3://, gs://)
      - Other user deps for other schemes which fsspec supports
    """
    adapter_ref = adapter_ref.strip()
    if not adapter_ref:
        raise ValueError("adapter_ref must be a non-empty string")

    # Do not allow fragments because mlflow can have "#" in the path
    parsed_url = urlparse(adapter_ref, allow_fragments=False)
    scheme = parsed_url.scheme.lower()
    if not scheme and not parsed_url.netloc:
        # Assume local path or HF repo id
        logging.info(f"Assuming {adapter_ref} is a local path or HF repo id")
        return adapter_ref  # downstream handles local paths / HF repo ids / etc.

    base_cache = Path(cache_dir).expanduser() if cache_dir else (Path.home() / ".cache")
    cache_root = base_cache / "peft_adapters"
    cache_root.mkdir(parents=True, exist_ok=True)

    digest = hashlib.blake2b(
        adapter_ref.encode("utf-8"), digest_size=DIGEST_SIZE_FOR_PEFT_PATHS
    ).hexdigest()
    digest_path = cache_root / f"{scheme}_{digest}"

    digest_path.mkdir(parents=True, exist_ok=True)

    tracking_uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    is_mlflow_url = scheme == "mlflow"
    if not is_mlflow_url and tracking_uri:
        # Check if scheme+netloc match the MLflow tracking URI
        tracking_parsed = urlparse(tracking_uri)
        adapter_scheme_netloc = f"{parsed_url.scheme}://{parsed_url.netloc}"
        tracking_scheme_netloc = f"{tracking_parsed.scheme}://{tracking_parsed.netloc}"
        is_mlflow_url = adapter_scheme_netloc == tracking_scheme_netloc

    if is_mlflow_url:
        try:
            import mlflow
            from mlflow.artifacts import download_artifacts
        except Exception as mlflow_exception:
            raise ImportError(
                "mlflow is required for mlflow:// refs. Install: [uv] pip install mlflow"
            ) from mlflow_exception

        run_id = (
            parsed_url.netloc if scheme == "mlflow" else parsed_url.path.split("/")[-1]
        )
        if not run_id:
            raise ValueError(f"Invalid mlflow ref (missing run id): {adapter_ref!r}")

        artifact_path = parsed_url.path.lstrip("/") or None

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Default when user supplies only mlflow://<run_id>
        if artifact_path is None:
            try:
                local_path = download_artifacts(
                    run_id=run_id,
                    artifact_path="hf_checkpoint/peft",
                    dst_path=str(digest_path),
                )
            except Exception:
                local_path = download_artifacts(
                    run_id=run_id,
                    artifact_path="hf_checkpoint",
                    dst_path=str(digest_path),
                )
        else:
            local_path = download_artifacts(
                run_id=run_id, artifact_path=artifact_path, dst_path=str(digest_path)
            )

        local_dir = Path(local_path)
        if not local_dir.is_dir():
            raise ValueError(
                f"MLflow artifact resolved to a file; expected directory: {local_dir}"
            )
        return str(local_dir)

    if scheme == "git":
        try:
            from git import Repo  # GitPython
        except Exception as e:
            raise ImportError(
                "gitpython is required for git:// refs. Install: [uv] pip install gitpython"
            ) from e

        git_parsed_url = urlparse(adapter_ref, allow_fragments=True)

        # Minimal: map git://host/path -> https://host/path
        repo_url = f"https://{git_parsed_url.netloc}{git_parsed_url.path}"

        rev = None
        subdir = None
        if git_parsed_url.fragment:
            # "#rev:subdir" or "#rev"
            if ":" in git_parsed_url.fragment:
                rev, subdir = git_parsed_url.fragment.split(":", 1)
                rev = rev or None
                subdir = subdir or None
            else:
                rev = git_parsed_url.fragment or None

        if not (digest_path / ".git").exists():
            Repo.clone_from(repo_url, str(digest_path))
        repo = Repo(str(digest_path))
        if rev:
            repo.git.fetch("--all", "--tags")
            repo.git.checkout(rev)
            if not repo.head.is_detached:
                try:
                    repo.git.pull()
                except Exception as error:
                    raise ValueError(
                        f"Failed to pull repository {repo_url} at revision {rev}"
                    ) from error

        return str(digest_path / subdir) if subdir else str(digest_path)

    # s3:// or gs:// (or others) via fsspec
    try:
        import fsspec
    except Exception as e:
        raise ImportError(
            "fsspec is required for s3:// and gs:// refs. Install: [uv] pip install fsspec"
        ) from e

    fs, base = fsspec.core.url_to_fs(adapter_ref)
    if fs.isfile(base):
        raise ValueError(
            f"{adapter_ref!r} points to a file; expected a directory/prefix containing adapter files."
        )

    base = base.rstrip("/")
    for remote in fs.find(base):
        if fs.isdir(remote):
            continue
        relative_path = remote[len(base) :].lstrip("/")
        if not relative_path:
            continue
        local_path = digest_path / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with fs.open(remote, "rb") as remote_file, open(local_path, "wb") as local_file:
            shutil.copyfileobj(remote_file, local_file)

    return str(digest_path)
