from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from google import genai
from google.genai.types import GenerateContentConfig

from .eval_engine import EvalEngine
from .util_functions import load_tokenizer_with_transformers

if TYPE_CHECKING:
    from datasets import Dataset

    from .eval_config import EvaluationConfig
    from .sampling_config import SamplingConfig

logger = logging.getLogger(__name__)


class GeminiEvalEngine(EvalEngine):
    """
    EvalEngine backend that calls the Google Gemini API instead of a local model.
    """

    def __init__(
        self,
        eval_config: EvaluationConfig,
        is_judge: bool = False,
    ) -> None:
        self.eval_config = eval_config
        self.is_judge = is_judge

        model_path_or_repo_id = self._get_model_path_or_repo_id(eval_config, is_judge)
        model_token = self._get_model_token(eval_config, is_judge)

        tokenizer_path = (
            getattr(eval_config, "tokenizer_path_or_repo_id", None)
            or model_path_or_repo_id
        )
        self.tokenizer = load_tokenizer_with_transformers(
            tokenizer_path,
            model_token,
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        api_key_env_var = getattr(eval_config, "gemini_api_key_env_var", "GOOGLE_API_KEY")
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env_var} is not set in environment variables. "
                "Set it to a valid Gemini API key."
            )
        self.client = genai.Client(api_key=api_key)
        if not self.client:
            raise RuntimeError("Failed to create Gemini client.")

        self.gemini_model_name: str = getattr(
            eval_config,
            "gemini_model_name",
            "gemini-2.0-flash",
        )
        self.eval_dataset: Dataset | None = None

    def set_dataset(self, eval_dataset: Dataset) -> None:
        self.eval_dataset = eval_dataset

    def generate_answers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_config: SamplingConfig,
    ) -> list[str]:
        prompts = self._build_prompts_from_input_ids(input_ids, attention_mask)
        responses: list[str] = []

        do_sample = (
            sampling_config.do_sample
            if sampling_config.do_sample is not None
            else self._get_sample_from_config(self.eval_config, self.is_judge)
        )
        max_output_tokens = self._get_max_new_tokens(self.eval_config, self.is_judge)
        temperature = (
            sampling_config.temperature
            if sampling_config.temperature is not None
            else (1.0 if do_sample else 0.0)
        )

        for idx, prompt in enumerate(prompts):
            logger.info(
                "GeminiEvalEngine: calling Gemini for sample %d/%d "
                "(model=%s, max_output_tokens=%d)",
                idx + 1,
                len(prompts),
                self.gemini_model_name,
                max_output_tokens,
            )
            answer = self._call_gemini_single(
                prompt, max_output_tokens=max_output_tokens, temperature=temperature
            )
            responses.append(answer)

        return responses

    def get_batch_size(self) -> int:
        batch_size = self._get_batch_size_from_config(self.eval_config, self.is_judge)
        if batch_size is None:
            if self.eval_dataset is None:
                batch_size = 8
            else:
                batch_size = max(1, min(len(self.eval_dataset), 8))
            logger.info("Defaulting to batch size %s for Gemini backend", batch_size)
        return batch_size

    def free_model(self) -> None:
        self.client = None

    def _build_prompts_from_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[str]:
        prompts: list[str] = []
        input_ids = input_ids.cpu()
        attention_mask = attention_mask.cpu()

        for ids_row, mask_row in zip(input_ids, attention_mask, strict=False):
            valid_ids = ids_row[mask_row.bool()]
            prompt = self.tokenizer.decode(
                valid_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            prompts.append(prompt)

        return prompts

    def _call_gemini_single(
        self, prompt: str, *, max_output_tokens: int, temperature: float
    ) -> str:
        if self.client is None:
            raise RuntimeError("Gemini client has been freed. Cannot generate answers.")

        cfg = GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        try:
            resp = self.client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt,
                config=cfg,
            )
            text = getattr(resp, "text", None)
            return (text or "").strip()
        except Exception as exc:  # pragma: no cover - API errors are runtime dependent
            logger.error("Gemini call failed: %s", exc, exc_info=True)
            return ""

