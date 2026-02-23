from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, cast

import torch

from .eval_engine import (
    EngineInputMode,
    EvalDataset,
    JudgePrompt,
    PromptEvalEngine,
)
from .util_functions import (
    load_tokenizer_with_transformers,
    load_vllm_model,
    maybe_download_adapter,
    pick_best_dtype,
    safe_apply_chat_template,
)
from .vllm_config import VllmConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.inputs.data import PromptType

    from .eval_config import EvaluationConfig
    from .sampling_config import SamplingConfig


class VllmEvalEngine(PromptEvalEngine):
    def generation_input_mode(self) -> EngineInputMode:
        return EngineInputMode.PROMPT

    def __init__(
        self,
        eval_config: EvaluationConfig,
        is_judge: bool = False,
    ) -> None:
        self.eval_config = eval_config
        self.is_judge = is_judge
        self.max_model_len = self._resolve_max_model_len(eval_config, is_judge)

        model_path_or_repo_id = (
            eval_config.judge_path_or_repo_id
            if is_judge
            else eval_config.model_path_or_repo_id
        )
        tokenizer_path_or_repo_id = (
            (
                eval_config.judge_tokenizer_path_or_repo_id
                or eval_config.judge_path_or_repo_id
            )
            if is_judge
            else (
                eval_config.model_tokenizer_path_or_repo_id
                or eval_config.model_path_or_repo_id
            )
        )
        lora_path_or_repo_id = (
            eval_config.lora_path_or_repo_id if not self.is_judge else None
        )
        model_token = eval_config.judge_token if is_judge else eval_config.model_token
        use_4bit = eval_config.use_4bit_judge if is_judge else eval_config.use_4bit
        batch_size = (
            eval_config.judge_batch_size if is_judge else eval_config.batch_size
        ) or 256

        self.tokenizer = load_tokenizer_with_transformers(
            tokenizer_path_or_repo_id,
            model_token,
            trust_remote_code=eval_config.trust_remote_code,
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = pick_best_dtype(device)
        quantization = "bitsandbytes" if use_4bit else None
        # Extract vLLM configuration
        vllm_config = eval_config.vllm_config or VllmConfig()

        logging.info(
            "Initializing vLLM with max_num_seqs=%s and gpu_memory_utilization=%s",
            batch_size,
            vllm_config.gpu_memory_utilization,
        )

        self.model = load_vllm_model(
            model_path_or_repo_id,
            dtype,
            eval_config.trust_remote_code,
            batch_size,
            model_token,
            enforce_eager=vllm_config.enforce_eager,
            quantization=quantization,
            max_model_len=self.max_model_len,
            tokenizer_mode=vllm_config.tokenizer_mode,
            config_format=vllm_config.config_format,
            load_format=vllm_config.load_format,
            gpu_memory_utilization=vllm_config.gpu_memory_utilization,
            enable_lora=vllm_config.enable_lora and not self.is_judge,
            max_lora_rank=vllm_config.max_lora_rank,
        )
        self._vllm_sampling_params = None
        if lora_path_or_repo_id is not None:
            try:
                from vllm.lora.request import LoRARequest
            except ImportError as exc:
                raise ImportError(
                    "vLLM is not installed. Install it with `uv pip install llm-behavior-eval[vllm]` to enable vllm for inference (e.g. when using the --inference-engine argument)."
                ) from exc
            try:
                mlflow_tracking_uri = (
                    self.eval_config.mlflow_config.mlflow_tracking_uri
                    if self.eval_config.mlflow_config
                    else None
                )
                lora_path_or_repo_id = maybe_download_adapter(
                    lora_path_or_repo_id, mlflow_tracking_uri=mlflow_tracking_uri
                )
                self.lora_request = LoRARequest("adapter", 1, lora_path_or_repo_id)
            except Exception as e:
                raise ValueError(
                    f"Failed to load LoRA from path {lora_path_or_repo_id}. Verify that the path is either a local path, a HF repo or a remote location with a valid scheme."
                ) from e
        else:
            self.lora_request = None

    @staticmethod
    def _resolve_max_model_len(
        eval_config: EvaluationConfig,
        is_judge: bool,
    ) -> int | None:
        if not eval_config.vllm_config:
            return None
        if is_judge:
            return (
                eval_config.vllm_config.judge_max_model_len
                if eval_config.vllm_config.judge_max_model_len is not None
                else eval_config.vllm_config.max_model_len
            )
        return eval_config.vllm_config.max_model_len

    def set_dataset(self, eval_dataset: EvalDataset) -> None:
        self.eval_dataset = eval_dataset

    def _get_vllm_sampling_params(
        self,
        sampling_config: SamplingConfig,
    ):
        """
        Get the sampling parameters for vLLM.

        Args:
            do_sample: Whether to sample from the model.
            temperature: The temperature for sampling. None means the default vLLM temperature is used. Overrides the do_sample argument.
            top_p: The top-p value for sampling. Defaults to 1.0.
            top_k: The top-k value for sampling. Defaults to 0.
            seed: The seed for sampling. None means no seed is set.

        Returns:
            The sampling parameters for vLLM.
        """
        from vllm import SamplingParams

        if sampling_config.do_sample is None:
            do_sample = (
                self.eval_config.sample_judge
                if self.is_judge
                else self.eval_config.sample
            )
        else:
            do_sample = sampling_config.do_sample
        max_new_tokens = (
            self.eval_config.max_judge_tokens
            if self.is_judge
            else self.eval_config.max_answer_tokens
        )
        if sampling_config.temperature is None:
            temperature = 1.0 if do_sample else 0.0
        else:
            temperature = sampling_config.temperature
        top_p = sampling_config.top_p if sampling_config.top_p is not None else 1.0
        top_k = sampling_config.top_k if sampling_config.top_k is not None else 0
        stop_token_ids = self._collect_stop_token_ids()
        return SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_token_ids=stop_token_ids,
            seed=sampling_config.seed,
        )

    def _collect_stop_token_ids(self) -> list[int] | None:
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            return None
        if isinstance(eos_token_id, list):
            return [int(token) for token in eos_token_id]
        return [int(eos_token_id)]

    def get_batch_size(self) -> int:
        batch_size = (
            self.eval_config.judge_batch_size
            if self.is_judge
            else self.eval_config.batch_size
        )

        if batch_size is None:
            batch_size = max(1, min(len(self.eval_dataset), 8))
            logging.info("Defaulting to batch size %s for vLLM backend", batch_size)
        return batch_size

    def free_model(self) -> None:
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
        self.model.llm_engine.engine_core.shutdown()
        del self.model
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()

    def format_prompt(self, messages: list[dict[str, str]]) -> JudgePrompt:
        """Apply chat template to format messages into a tokenized prompt string."""
        return safe_apply_chat_template(self.tokenizer, messages)

    def generate_answers_from_prompts(
        self,
        prompts: list[JudgePrompt],
        sampling_config: SamplingConfig,
    ) -> list[str]:
        """Generate from preformatted string prompts."""
        if not prompts:
            return []
        string_prompts = self.normalize_prompts_to_strings(prompts)
        sampling_params = self._get_vllm_sampling_params(sampling_config)
        outputs = self.model.generate(
            prompts=cast("Sequence[PromptType]", string_prompts),
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=self.lora_request,
        )
        responses: list[str] = []
        for output in outputs:
            candidates = getattr(output, "outputs", [])
            if not candidates:
                responses.append("")
                continue
            first_candidate = candidates[0]
            responses.append(getattr(first_candidate, "text", ""))
        return responses
