from typing import cast

import torch

try:
    from hirundo_core.h_core.debias.methods.plugin_model import PluginModelForCausalLM
except ImportError as err:
    raise ImportError("Plugin model not found. `use_plugin` is not available.") from err
from transformers import AutoConfig, AutoTokenizer
from transformers.data.data_collator import DataCollator

from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.transformers_eval_engine import (
    TransformersEvalEngine,
)
from llm_behavior_eval.evaluation_utils.util_functions import pick_best_dtype


class PluginEvalEngine(TransformersEvalEngine):
    def __init__(
        self,
        data_collator: DataCollator,
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
            dtype=dtype,
            token=eval_config.model_token,
        )
        self.data_collator = data_collator
        self.eval_config = eval_config
        self.device = next(self.model.parameters()).device

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
            )
        generated_tokens = outputs[:, model_input_ids.shape[1] :].detach().cpu()
        return tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

