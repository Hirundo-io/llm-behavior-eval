from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import torch
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollator  # noqa: TCH002

try:
    from hirundo_core.h_core.debias.methods.plugin_model import PluginModelForCausalLM
except ImportError as err:
    raise ImportError("Plugin model not found. `use_plugin` is not available.") from err
from transformers import AutoConfig, AutoTokenizer

from llm_behavior_eval.evaluation_utils.transformers_eval_engine import (
    TransformersEvalEngine,
)
from llm_behavior_eval.evaluation_utils.util_functions import pick_best_dtype

if TYPE_CHECKING:
    from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig


class PluginEvalEngine(TransformersEvalEngine):
    def __init__(
        self,
        data_collator: DataCollator,  # type: ignore[type-arg]
        eval_config: EvaluationConfig,
    ) -> None:
        plugin_path = eval_config.model_path_or_repo_id
        cfg = AutoConfig.from_pretrained(plugin_path)
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = pick_best_dtype(device)
        self.model = PluginModelForCausalLM.from_pretrained(
            plugin_path,
            device_map=eval_config.device_map,
            torch_dtype=dtype,
            token=eval_config.model_token,
        )
        self.data_collator = data_collator
        self.eval_config = eval_config
        self.device = next(self.model.parameters()).device
        print(f"PluginEvalEngine initialized with backend: {self.eval_config.plugin_backend}")

    def _get_first_non_oom_batch_size(self, candidate_bs: int) -> int:
        """
        Override to pass the correct backend parameter when probing batch sizes.
        Without this, the plugin model defaults to 'gemini' backend during batch size probing.
        """
        logging.info(f"Trying batch size: {candidate_bs}")
        dl = DataLoader(
            cast("torch.utils.data.Dataset", self.eval_dataset),  # type: ignore[type-arg]
            batch_size=candidate_bs,
            shuffle=False,
            collate_fn=self.data_collator,
        )
        it = iter(dl)
        batch = next(it)
        input_ids = batch["test_input_ids"].to(self.model.device)
        attention_mask = batch["test_attention_mask"].to(self.model.device)
        with torch.inference_mode():
            model = cast("PluginModelForCausalLM", self.model)
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.eval_config.answer_tokens,
                do_sample=self.eval_config.sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                backend=self.eval_config.plugin_backend,  # Use the configured backend
            )
        return candidate_bs

    def generate_answers(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> list[str]:
        """
        Overrides the standard generation to use the custom, fused generate logic.
        """
        model = cast("PluginModelForCausalLM", self.model)
        tokenizer = self.tokenizer
        device = self.device
        model_input_ids = input_ids.to(device)
        model_attention = attention_mask.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=cast("torch.LongTensor", model_input_ids),
                attention_mask=cast("torch.LongTensor", model_attention),
                max_new_tokens=self.eval_config.answer_tokens,
                do_sample=self.eval_config.sample,
                # temperature=self.eval_config.temperature, # TODO: add temperature
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                backend=self.eval_config.plugin_backend,
            )
        generated_tokens = outputs[:, model_input_ids.shape[1] :].detach().cpu()
        return tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

