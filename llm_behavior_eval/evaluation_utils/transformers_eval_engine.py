import logging
from typing import TYPE_CHECKING, cast

import torch
from accelerate.utils.memory import find_executable_batch_size
from transformers import set_seed
from transformers.data.data_collator import DataCollator

from .eval_config import EvaluationConfig
from .eval_engine import EvalDataset, JudgePrompt, PromptEvalEngine
from .max_batch_size import MAX_BATCH_SIZE
from .sampling_config import SamplingConfig
from .util_functions import (
    is_model_multimodal,
    load_transformers_model_and_tokenizer,
    safe_apply_chat_template,
)

if TYPE_CHECKING:
    from transformers.generation.utils import GenerationMixin


class TransformersEvalEngine(PromptEvalEngine):
    def __init__(
        self,
        data_collator: DataCollator,
        eval_config: EvaluationConfig,
        is_judge: bool = False,
    ) -> None:
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
        model_token = eval_config.judge_token if is_judge else eval_config.model_token
        use_4bit = eval_config.use_4bit_judge if is_judge else eval_config.use_4bit
        self.tokenizer, self.model = load_transformers_model_and_tokenizer(
            model_path_or_repo_id,
            model_token,
            use_4bit,
            eval_config.device_map,
            eval_config.trust_remote_code,
            tokenizer_name_or_path=tokenizer_path_or_repo_id,
        )
        self.tokenizer.padding_side = "left"
        self.data_collator = data_collator
        self.eval_config = eval_config
        self.is_judge = is_judge
        self.is_multimodal = is_model_multimodal(
            model_path_or_repo_id,
            eval_config.trust_remote_code,
            model_token,
        )

    def set_dataset(self, eval_dataset: EvalDataset) -> None:
        self.eval_dataset = eval_dataset

    def _get_first_non_oom_batch_size(self, candidate_bs: int) -> int:
        logging.info("Trying batch size: %s", candidate_bs)
        if not hasattr(self.eval_dataset, "__getitem__"):
            return candidate_bs

        first_sample = self.eval_dataset[0]
        if "test_messages" not in first_sample:
            # Some tests and legacy datasets are tokenized differently; skip probing.
            return candidate_bs

        prompt_count = min(candidate_bs, len(self.eval_dataset))
        prompts = [
            cast("JudgePrompt", self.eval_dataset[index]["test_messages"])
            for index in range(prompt_count)
        ]

        self.generate_answers_from_prompts(
            prompts,
            sampling_config=SamplingConfig(
                do_sample=(
                    self.eval_config.sample_judge
                    if self.is_judge
                    else self.eval_config.sample
                ),
                temperature=self.eval_config.sampling_config.temperature,
                top_p=self.eval_config.sampling_config.top_p,
                top_k=self.eval_config.sampling_config.top_k,
                seed=self.eval_config.sampling_config.seed,
            ),
        )
        return candidate_bs

    def ensure_test_model_ready(self) -> None:
        self.model.eval()

    def get_batch_size(self) -> int:
        batch_size = (
            self.eval_config.judge_batch_size
            if self.is_judge
            else self.eval_config.batch_size
        )

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

    def format_prompt(self, messages: list[dict[str, str]]) -> JudgePrompt:
        """Apply chat template to format messages into a tokenized prompt string."""
        max_answer_tokens = (
            self.eval_config.max_judge_tokens
            if self.is_judge
            else self.eval_config.max_answer_tokens
        )
        return safe_apply_chat_template(
            self.tokenizer,
            messages,
            is_multimodal=self.is_multimodal,
            max_answer_tokens=max_answer_tokens,
            reasoning=self.eval_config.reasoning,
            pass_max_answer_tokens=self.eval_config.pass_max_answer_tokens,
        )

    def generate_answers_from_prompts(
        self,
        prompts: list[JudgePrompt],
        sampling_config: SamplingConfig,
    ) -> list[str]:
        """Tokenize prompts and generate responses with the local transformers model."""
        if not prompts:
            return []
        string_prompts = self.normalize_prompts_to_strings(prompts)
        tokenized = self.tokenizer(
            string_prompts,
            truncation=True,
            return_tensors="pt",
            padding=True,
        )
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
        temperature = sampling_config.temperature
        if temperature is None:
            temperature = 1.0 if do_sample else 0.0
        top_p = sampling_config.top_p if sampling_config.top_p is not None else 1.0
        top_k = sampling_config.top_k if sampling_config.top_k is not None else 0
        seed = sampling_config.seed
        if seed is not None:
            set_seed(seed)

        input_ids = cast("torch.Tensor", tokenized["input_ids"]).to(self.model.device)
        attention_mask = cast("torch.Tensor", tokenized["attention_mask"]).to(
            self.model.device
        )
        with torch.inference_mode():
            outputs = cast("GenerationMixin", self.model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        generated_tokens = outputs[:, input_ids.shape[1] :].detach().cpu()
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
