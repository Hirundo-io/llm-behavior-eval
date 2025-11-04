import contextlib
import logging

import torch
from datasets import Dataset

from .eval_config import EvaluationConfig
from .eval_engine import EvalEngine
from .util_functions import (
    build_vllm_prompt_token_ids,
    load_tokenizer_with_transformers,
    load_vllm_model,
    pick_best_dtype,
)


class VllmEvalEngine(EvalEngine):
    def __init__(
        self,
        eval_dataset: Dataset,
        eval_config: EvaluationConfig,
    ) -> None:
        self.eval_dataset = eval_dataset
        self.eval_config = eval_config
        self.tokenizer = load_tokenizer_with_transformers(
            eval_config.model_path_or_repo_id,
            eval_config.model_token,
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = pick_best_dtype(device)
        quantization = "bitsandbytes" if eval_config.use_4bit else None
        trust_remote_code = eval_config.model_path_or_repo_id.startswith("nvidia/")
        self.model = load_vllm_model(
            eval_config.model_path_or_repo_id,
            dtype,
            trust_remote_code,
            eval_config.batch_size or 256,
            eval_config.model_token,
            enforce_eager=not torch.cuda.is_available(),
            quantization=quantization,
        )
        self._vllm_sampling_params = None

    def generate_answers(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> list[str]:
        prompt_token_ids = build_vllm_prompt_token_ids(input_ids, attention_mask)
        sampling_params = self._get_vllm_sampling_params()
        outputs = self.model.generate(
            prompt_token_ids=prompt_token_ids,
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

    def _get_vllm_sampling_params(self):
        if getattr(self, "_vllm_sampling_params", None) is None:
            from vllm import SamplingParams

            temperature = 1.0 if self.eval_config.sample else 0.0
            stop_token_ids = self._collect_stop_token_ids()
            self._vllm_sampling_params = SamplingParams(
                max_tokens=self.eval_config.answer_tokens,
                temperature=temperature,
                top_p=1.0,
                stop_token_ids=stop_token_ids,
            )
        return self._vllm_sampling_params

    def _collect_stop_token_ids(self) -> list[int] | None:
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            return None
        if isinstance(eos_token_id, list):
            return [int(token) for token in eos_token_id]
        return [int(eos_token_id)]

    def get_batch_size(self) -> int:
        # If batch_size is None, auto-detect the largest batch size to fit on current hardware.
        batch_size = self.eval_config.batch_size
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
