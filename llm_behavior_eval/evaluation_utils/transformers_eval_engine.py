import logging
from typing import TYPE_CHECKING, cast

from accelerate.utils.memory import find_executable_batch_size
from torch.utils.data import DataLoader
from datasets import Dataset
import torch

from .eval_config import EvaluationConfig
from .util_functions import (
    load_transformers_model_and_tokenizer,
)
from .eval_engine import EvalEngine
from .max_batch_size import MAX_BATCH_SIZE


if TYPE_CHECKING:
    from transformers.data.data_collator import DataCollator
    from transformers.generation.utils import GenerationMixin


class TransformersEvalEngine(EvalEngine):
    def __init__(
        self,
        eval_dataset: Dataset,
        data_collator: DataCollator,
        eval_config: EvaluationConfig,
    ) -> None:
        self.tokenizer, self.model = load_transformers_model_and_tokenizer(
            eval_config.model_path_or_repo_id,
            eval_config.use_4bit,
            eval_config.device_map,
        )
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.eval_config = eval_config

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
        with torch.inference_mode():
            cast("GenerationMixin", self.model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.eval_config.answer_tokens,
                do_sample=self.eval_config.sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return candidate_bs

    def generate_answers(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> list[str]:
        device = self.model.device
        model_input_ids = input_ids.to(device)
        model_attention = attention_mask.to(device)
        with torch.inference_mode():
            outputs = cast("GenerationMixin", self.model).generate(
                input_ids=model_input_ids,
                attention_mask=model_attention,
                max_new_tokens=self.eval_config.answer_tokens,
                do_sample=self.eval_config.sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_tokens = outputs[:, model_input_ids.shape[1] :].detach().cpu()
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def ensure_test_model_ready(self) -> None:
        self.model.eval()

    def get_batch_size(self) -> int:
        batch_size = self.eval_config.batch_size
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
            batch_size = cast(int, wrapper())
            logging.info("Selected batch size: %s", batch_size)
        return batch_size

    def free_model(self) -> None:
        self.model = self.model.cpu()
        del self.model
