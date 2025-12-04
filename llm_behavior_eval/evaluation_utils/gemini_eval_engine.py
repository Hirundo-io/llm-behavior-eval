from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

from .eval_engine import EvalEngine
from .util_functions import load_tokenizer_with_transformers

if TYPE_CHECKING:
    import torch
    from datasets import Dataset

    from .eval_config import EvaluationConfig

logger = logging.getLogger(__name__)


class GeminiEvalEngine(EvalEngine):
    """
    EvalEngine backend that calls the Google Gemini API instead of a local model.

    Responsibilities:
    - Use HF tokenizer (from model_path_or_repo_id) to:
        * Decode input_ids + attention_mask -> text prompts
        * Allow downstream code to re-tokenize Gemini outputs for token counting
    - Use EvaluationConfig to control:
        * answer_tokens -> max_output_tokens
        * sample flag -> temperature (0.0 when deterministic)
        * (optionally) gemini_model_name, else a default is used
    """

    def __init__(
        self,
        eval_config: EvaluationConfig,
    ) -> None:
        self.eval_config = eval_config

        # Load tokenizer based on HF model path. This is *not* Gemini's tokenizer,
        # but your "reference" tokenizer (e.g., plugin base) used for prompts + token counts.
        self.tokenizer = load_tokenizer_with_transformers(
            eval_config.tokenizer_path_or_repo_id or eval_config.model_path_or_repo_id,
            eval_config.model_token,
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Gemini client from env
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set in environment variables.")
        self.client = genai.Client(api_key=api_key)
        if not self.client:
            raise RuntimeError("Failed to create Gemini client.")

        # Allow specifying Gemini model name in config; otherwise use a sensible default
        # You can add this field to EvaluationConfig if you want:
        #   gemini_model_name: str = "gemini-2.5-flash-lite"
        self.gemini_model_name: str = getattr(
            eval_config,
            "gemini_model_name",
            "gemini-2.5-flash-lite",
        )

        # Dataset reference (for get_batch_size default)
        self.eval_dataset: Dataset | None = None

    # -------------------------------------------------------------------------
    # EvalEngine interface
    # -------------------------------------------------------------------------

    def set_dataset(self, eval_dataset: Dataset) -> None:
        self.eval_dataset = eval_dataset

    def generate_answers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[str]:
        """
        Generate one Gemini answer per row of input_ids.

        `input_ids` / `attention_mask` are expected to come from your collator
        (as in TransformersEvalEngine / VllmEvalEngine). We:
          1) Decode to text prompts using the HF tokenizer
          2) Call Gemini once per prompt
          3) Return list[str] answers
        """
        prompts = self._build_prompts_from_input_ids(input_ids, attention_mask)
        responses: list[str] = []

        for idx, prompt in enumerate(prompts):
            logger.info(
                "GeminiEvalEngine: calling Gemini for sample %d/%d "
                "(model=%s, max_output_tokens=%d)",
                idx + 1,
                len(prompts),
                self.gemini_model_name,
                self.eval_config.answer_tokens,
            )
            answer = self._call_gemini_single(prompt)
            responses.append(answer)

        return responses

    def get_batch_size(self) -> int:
        """
        Batch size here is only for how many prompts per DataLoader batch.
        The engine itself calls Gemini one-by-one in generate_answers.
        """
        batch_size = self.eval_config.batch_size
        if batch_size is None:
            if self.eval_dataset is None:
                batch_size = 8
            else:
                batch_size = max(1, min(len(self.eval_dataset), 8))
            logger.info("Defaulting to batch size %s for Gemini backend", batch_size)
        return batch_size

    def free_model(self) -> None:
        """
        For API-based models there is no GPU model to free.
        We just drop the client reference so it can be GC'ed.
        """
        self.client = None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _build_prompts_from_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[str]:
        """
        Decode each row of input_ids -> text prompt, trimming padding using attention_mask.

        Handles both left-padding and right-padding correctly by using the attention_mask
        to identify valid token positions.
        """
        prompts: list[str] = []

        # Work on CPU for decoding
        input_ids = input_ids.cpu()
        attention_mask = attention_mask.cpu()

        for ids_row, mask_row in zip(input_ids, attention_mask, strict=False):
            # Use attention_mask to extract only valid (non-padding) tokens
            # This works correctly for both left-padding and right-padding
            valid_ids = ids_row[mask_row.bool()]
            prompt = self.tokenizer.decode(
                valid_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            prompts.append(prompt)

        return prompts

    def _call_gemini_single(self, prompt: str) -> str:
        """
        Issue a single Gemini API call.

        Uses:
          - eval_config.answer_tokens -> max_output_tokens
          - eval_config.sample -> temperature (0 if deterministic)
        """
        # If you ever add `temperature` to EvaluationConfig, wire it here.
        temperature = getattr(self.eval_config, "temperature", 0.7)
        if not self.eval_config.sample:
            temperature = 0.0

        if self.client is None:
            raise RuntimeError("Gemini client has been freed. Cannot generate answers.")

        cfg = GenerateContentConfig(
            max_output_tokens=self.eval_config.answer_tokens,
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
        except Exception as e:
            logger.error("Gemini call failed: %s", e, exc_info=True)
            return ""
