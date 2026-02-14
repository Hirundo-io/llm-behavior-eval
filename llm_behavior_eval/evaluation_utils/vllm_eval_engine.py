from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, cast

import torch

from .eval_engine import EvalDataset, JudgePrompt, PromptEvalEngine, TensorEvalEngine
from .util_functions import (
    build_vllm_prompt_token_ids,
    load_tokenizer_with_transformers,
    load_vllm_model,
    pick_best_dtype,
    safe_apply_chat_template,
)

if TYPE_CHECKING:
    from vllm.inputs.data import PromptType

    from .eval_config import EvaluationConfig
    from .sampling_config import SamplingConfig


class VllmEvalEngine(TensorEvalEngine, PromptEvalEngine):
    def __init__(
        self,
        eval_config: EvaluationConfig,
        is_judge: bool = False,
        max_model_len: int | None = None,
    ) -> None:
        self.eval_config = eval_config
        self.is_judge = is_judge
        self.max_model_len = max_model_len

        model_path_or_repo_id = self._get_model_path_or_repo_id(eval_config, is_judge)
        tokenizer_path_or_repo_id = self._get_tokenizer_path_or_repo_id(
            eval_config, is_judge
        )
        model_token = self._get_model_token(eval_config, is_judge)
        use_4bit = self._get_use_4bit(eval_config, is_judge)
        batch_size_config = self._get_batch_size_from_config(eval_config, is_judge)
        batch_size = batch_size_config or 256

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
        vllm_config = eval_config.vllm_config
        tokenizer_mode = vllm_config.tokenizer_mode if vllm_config else None
        config_format = vllm_config.config_format if vllm_config else None
        load_format = vllm_config.load_format if vllm_config else None
        gpu_memory_utilization = (
            vllm_config.gpu_memory_utilization if vllm_config else 0.9
        )
        logging.info(
            "Initializing vLLM with max_num_seqs=%s and gpu_memory_utilization=%s",
            batch_size,
            gpu_memory_utilization,
        )

        self.model = load_vllm_model(
            model_path_or_repo_id,
            dtype,
            eval_config.trust_remote_code,
            batch_size,
            model_token,
            enforce_eager=not torch.cuda.is_available(),
            quantization=quantization,
            max_model_len=max_model_len,
            tokenizer_mode=tokenizer_mode,
            config_format=config_format,
            load_format=load_format,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self._vllm_sampling_params = None

    def set_dataset(self, eval_dataset: EvalDataset) -> None:
        self.eval_dataset = eval_dataset

    def generate_answers_from_tensors(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_config: SamplingConfig,
    ) -> list[str]:
        prompt_token_ids = build_vllm_prompt_token_ids(input_ids, attention_mask)
        prompts: list[PromptType] = [
            {"prompt_token_ids": tokens} for tokens in prompt_token_ids
        ]
        sampling_params = self._get_vllm_sampling_params(sampling_config)
        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
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
            do_sample = self._get_sample_from_config(self.eval_config, self.is_judge)
        else:
            do_sample = sampling_config.do_sample
        max_new_tokens = self._get_max_new_tokens(self.eval_config, self.is_judge)
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
        batch_size = self._get_batch_size_from_config(self.eval_config, self.is_judge)

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
        """Tokenize string prompts and generate answers."""
        if not prompts:
            return []
        string_prompts = self.normalize_prompts_to_strings(prompts)
        tokenized = self.tokenizer(
            string_prompts,
            return_tensors="pt",
            padding=True,
        )
        input_ids = cast("torch.Tensor", tokenized["input_ids"])
        attention_mask = cast("torch.Tensor", tokenized["attention_mask"])
        return self.generate_answers_from_tensors(
            input_ids, attention_mask, sampling_config
        )
