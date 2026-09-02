import logging
from typing import cast

import torch
from accelerate.utils.memory import find_executable_batch_size
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import set_seed
from transformers.data.data_collator import DataCollator

from .eval_config import EvaluationConfig
from .eval_engine import EvalEngine
from .harmony_output import (
    HarmonyOutputError,
    extract_harmony_final_answer,
    is_harmony_tokenizer,
)
from .max_batch_size import MAX_BATCH_SIZE
from .sampling_config import SamplingConfig
from .util_functions import load_transformers_model_and_tokenizer


def _truncate_at_eos(token_ids: list[int], eos_token_ids: set[int]) -> list[int]:
    """Drop the right padding ``generate`` appends after a finished sequence.

    Args:
        token_ids: One row of generated token IDs, excluding the prompt.
        eos_token_ids: Token IDs that end a generation.

    Returns:
        The row up to and including its first EOS token, or the row unchanged
        when it contains none (a row that hit the token limit is never padded).
    """
    for position, token_id in enumerate(token_ids):
        if token_id in eos_token_ids:
            return token_ids[: position + 1]
    return token_ids


class TransformersEvalEngine(EvalEngine):
    def __init__(
        self,
        data_collator: DataCollator,
        eval_config: EvaluationConfig,
        is_judge: bool = False,
    ) -> None:
        model_path_or_repo_id = self._get_model_path_or_repo_id(eval_config, is_judge)
        model_token = self._get_model_token(eval_config, is_judge)
        use_4bit = self._get_use_4bit(eval_config, is_judge)
        revision = (
            eval_config.judge_revision if is_judge else eval_config.model_revision
        )
        self.tokenizer, self.model = load_transformers_model_and_tokenizer(
            model_path_or_repo_id,
            model_token,
            use_4bit,
            eval_config.device_map,
            eval_config.trust_remote_code,
            revision=revision,
        )
        self._uses_harmony = is_harmony_tokenizer(self.tokenizer)
        self.data_collator = data_collator
        self.eval_config = eval_config
        self.is_judge = is_judge

    def set_dataset(self, eval_dataset: Dataset) -> None:
        self.eval_dataset = eval_dataset

    def _get_first_non_oom_batch_size(self, candidate_bs: int) -> int:
        logging.info(f"Trying batch size: {candidate_bs}")
        dl = DataLoader(
            cast("torch.utils.data.Dataset", self.eval_dataset),
            batch_size=candidate_bs,
            shuffle=False,
            collate_fn=self.data_collator,
        )
        it = iter(dl)
        batch = next(it)
        input_ids = batch["test_input_ids"].to(self.model.device)
        attention_mask = batch["test_attention_mask"].to(self.model.device)
        do_sample = self._get_sample_from_config(self.eval_config, self.is_judge)
        max_new_tokens = self._get_max_new_tokens(self.eval_config, self.is_judge)
        with torch.inference_mode():
            self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return candidate_bs

    def generate_answers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_config: SamplingConfig,
        repetition_penalty: float = 1.0,
    ) -> tuple[list[str], list[str | None]]:
        """Generate and decode one batch with Transformers.

        Args:
            input_ids: Token identifiers for the input prompts.
            attention_mask: Attention mask corresponding to the input tokens.
            sampling_config: Backend-independent decoding settings.
            repetition_penalty: Repetition penalty for this generation call.

        Returns:
            Generated answers and their inferred finish reasons.
        """
        if sampling_config.do_sample is None:
            do_sample = self._get_sample_from_config(self.eval_config, self.is_judge)
        else:
            do_sample = sampling_config.do_sample
        max_new_tokens = self._get_max_new_tokens(self.eval_config, self.is_judge)
        temperature = sampling_config.temperature
        if temperature is None:
            temperature = 1.0 if do_sample else 0.0
        top_p = sampling_config.top_p if sampling_config.top_p is not None else 1.0
        top_k = sampling_config.top_k if sampling_config.top_k is not None else 0
        seed = sampling_config.seed
        if seed is not None:
            set_seed(seed)
        device = self.model.device
        model_input_ids = input_ids.to(device)
        model_attention = attention_mask.to(device)
        eos_token_id = self.tokenizer.eos_token_id
        eos_token_ids: set[int]
        if eos_token_id is None:
            eos_token_ids = set()
        elif isinstance(eos_token_id, list):
            eos_token_ids = {int(token_id) for token_id in eos_token_id}
        else:
            eos_token_ids = {int(eos_token_id)}

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=model_input_ids,
                attention_mask=model_attention,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                return_dict_in_generate=True,
            )
        sequences = outputs.sequences
        generated_tokens = sequences[:, model_input_ids.shape[1] :].detach().cpu()
        completions = [
            _truncate_at_eos(
                [int(token_id) for token_id in row.tolist()], eos_token_ids
            )
            for row in generated_tokens
        ]
        finish_reasons: list[str | None] = [
            "stop" if completion and completion[-1] in eos_token_ids else "length"
            for completion in completions
        ]
        if self._uses_harmony:
            answers = []
            for index, completion in enumerate(completions):
                try:
                    answers.append(extract_harmony_final_answer(completion))
                except HarmonyOutputError:
                    answers.append("")
                    if finish_reasons[index] == "stop":
                        finish_reasons[index] = "harmony_parse_error"
        else:
            answers = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
        return answers, finish_reasons

    def ensure_test_model_ready(self) -> None:
        self.model.eval()

    def get_batch_size(self) -> int:
        batch_size = self._get_batch_size_from_config(self.eval_config, self.is_judge)

        if batch_size is None:
            starting_bs = max(1, min(len(self.eval_dataset), MAX_BATCH_SIZE))
            current_bs = starting_bs

            def halve_reducer():
                nonlocal current_bs
                current_bs = max(1, current_bs // 2)
                return current_bs

            wrapper = find_executable_batch_size(
                self._get_first_non_oom_batch_size,
                starting_batch_size=starting_bs,
                reduce_batch_size_fn=halve_reducer,
            )
            batch_size = cast("int", wrapper())
            logging.info("Selected batch size: %s", batch_size)
        return batch_size

    def free_model(self) -> None:
        self.model = self.model.cpu()
        del self.model
